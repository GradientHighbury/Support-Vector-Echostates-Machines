import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import numpy as np
import matplotlib.pyplot as plt


def nrmse(y_true:torch.Tensor,y_pred:torch.Tensor)->float:
    mse = torch.mean((y_true-y_pred)**2)
    variance = torch.var(y_true)
    return torch.sqrt(mse/(variance + 1e-8)).item()


def train_model(model:nn.Module,train_loader:DataLoader,val_loader:DataLoader,
                epochs:int =10,lr:float=1e-3,weight_deacy:float=1e-4,device:str="cpu") ->None:
    model.to(device)
    criterion= nn.MSELoss()
    optimizer = torch.optim.Adam(model.read_out.parameters(),lr=lr,weight_decay=weight_deacy)
    
    train_losse=[]
    val_losse=[]
    for epoch  in range (1,epochs+1):
        model.train()
        train_losses =[]
        val_losses=[]
        for x_batch,y_batch in train_loader:
            x_batch = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            y_batch = torch.as_tensor(y_batch, dtype=torch.float32, device=device).unsqueeze(1)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred,y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        train_losse.append(np.mean(train_losses))
        model.eval()
        val_true,val_pred=[],[]
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
                y_batch = torch.as_tensor(y_batch, dtype=torch.float32, device=device).unsqueeze(1)
                y_hat   = model(x_batch)
                val_loss = criterion(y_hat, y_batch) 
                val_true.append(y_batch.cpu())
                val_pred.append(y_hat.cpu())
                val_losses.append(val_loss.item())
        val_losse.append(np.mean(val_losses))
        val_true  = torch.cat(val_true,dim=0)
        val_pred  = torch.cat(val_pred,dim=0)
        val_score = nrmse(val_true,val_pred)
        

        if epoch==1 or epoch % 10 ==0:
            print(f"Epoch {epoch:03d}"
                  f"Train MSE: {np.mean(train_losses):6f} |"
                  f"Val NRMSE: {val_score:6f}")
        
            
    # ---------- Plot loss curves ----------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losse, label="Train Loss")
    plt.plot(val_losse, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train and Validation Loss ESN")
    plt.legend()
    plt.grid(True)
