from src.dataset.mackeyglass.mackeyglass import generate_mackey_glass_rk2
from src.dataloader.embedding_dataset import DelayEmbeddingDataset,sequential_split
from src.model.echostatenetwork import EchoStateNetwork
from src.model.fixedreservoir import FixedReservoir
from src.training.training import train_model,nrmse
from src.training.fitsvronreservoir import fit_svr_on_reservoir

import torch
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

def plot_all(final_val_true,final_val_pred):
    # ---------- Plot all predictions ----------
    y_true_np = final_val_true.numpy().flatten()
    y_pred_np = final_val_pred.numpy().flatten()

    plt.subplot(1, 2, 2)
    plt.plot(y_true_np, label="True")
    plt.plot(y_pred_np, label="Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("All Validation Predictions")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

class InputData:
    def __init__(self):
        self.length=8000
        self.embedded_dim=4
        self.delay=6
        self.horizon=84
        self.reservior_dim=200
        self.spectral_radius=0.9
        self.input_scale=0.5
        self.sparsity=0.9
        self.leak_rate=1.0
        self.seed=42
        self.epochs=100
        self.lr=1e-4
        self.weight_deacy=1e-4
        self.washout=50



def main_esm(input_data:InputData):

    #Choose device if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)

    # delay embedding + direct h-step prediction.
    series = generate_mackey_glass_rk2(length=input_data.length)

    #Normalise
    mean = series.mean()
    std = series.std()+1e-8

    series = (series-mean)/std

    #Paper Style idea
    # d(k) = [x(k), x(k-tau), x(k-2tau), x(k-3tau)]^T
    
    dataset = DelayEmbeddingDataset(series=series,embedded_dim=input_data.embedded_dim,delay=input_data.delay,horizon=input_data.horizon)

    n = len(dataset)
    n_train = int(0.7*n)
    n_val   = int(0.15*n)
    n_test=n-n_train-n_val

    train_set,val_set,test_set=torch.utils.data.random_split(dataset,[n_train,n_val,n_test],generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    val_loader   = DataLoader(val_set,batch_size=256,shuffle=False)
    test_loader  = DataLoader(test_set,batch_size=256,shuffle=False)

    model = EchoStateNetwork(input_dim=input_data.embedded_dim,reservior_dim=input_data.reservior_dim,
                             output_dim=1,spectral_radius=input_data.spectral_radius,input_scale=input_data.input_scale,
                             sparsity=input_data.sparsity,leak_rate=input_data.leak_rate,seed=input_data.seed)
    
    train_model(model=model,train_loader=train_loader,
                val_loader=val_loader,epochs=input_data.epochs,lr=input_data.lr,
                weight_deacy=input_data.weight_deacy,device=device)
    
    #Test 
    model.eval()
    y_true_all=[]
    y_pred_all=[]

    with torch.no_grad():
        for x_batch,y_batch in test_loader:
            x_batch = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            y_batch = torch.as_tensor(y_batch, dtype=torch.float32, device=device).unsqueeze(1)

            y_hat= model(x_batch)
            y_true_all.append(y_batch)
            y_pred_all.append(y_hat)

    
    y_true_all = torch.cat(y_true_all,dim=0)
    y_pred_all = torch.cat(y_pred_all,dim=0)


    test_score = nrmse(y_true_all,y_pred_all)

    print(f"\nTest NRMSE: {test_score:.6f}")

    # Show a few predictions
    print("\nSample predictions:")
    for i in range(10):
        print(
            f"true={y_true_all[i].item(): .4f} | "
            f"pred={y_pred_all[i].item(): .4f}"
        )

    plot_all(y_true_all,y_pred_all)

def main_svesm(input_data:InputData):


    #Choose device if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)


    #paper-like Mackey-Glass setup
    series = generate_mackey_glass_rk2(length=input_data.length,alpha=0.2,beta=-0.1,delta=17.0,h=0.1)

    make_embed=DelayEmbeddingDataset(series=series,embedded_dim=input_data.embedded_dim,delay=input_data.delay,horizon=input_data.horizon)

    X,y= make_embed.build_delay_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = sequential_split(X, y, train_ratio=0.7, val_ratio=0.15)

    #Fixed reservoir
    reservoir=FixedReservoir(input_dim=input_data.embedded_dim,reservoir_dim=input_data.reservior_dim,spectral_radius=input_data.spectral_radius,
                             input_scale=input_data.input_scale,sparsity=input_data.sparsity,leak_rate=input_data.leak_rate,
                             seed=input_data.seed)
    
    # Fit SVR on reservoir states
    svr_model, state_scaler, best_cfg, best_val_nrmse = fit_svr_on_reservoir(reservoir=reservoir,X_train=X_train,
                                                                             y_train=y_train,X_val=X_val,
                                                                             y_val=y_val,wash_out=input_data.washout,
                                                                             device=device)
    
    print("Best SVR config:", best_cfg)
    print(f"Best Validation NRMSE: {best_val_nrmse:.6f}")

    #Transform the test inputs into reseroir states
    H_test = reservoir.transform_sequence(X_test, wash_out=input_data.washout, device="cpu")
    y_test_eff = y_test[input_data.washout:]

    H_test_scaled = state_scaler.transform(H_test)
    y_test_pred = svr_model.predict(H_test_scaled)

    test_nrmse = nrmse(torch.as_tensor(y_test_eff, dtype=torch.float32, device=device), 
                       torch.as_tensor(y_test_pred, dtype=torch.float32, device=device))
    print(f"Test NRMSE: {test_nrmse:.6f}")
    print(f"Number of support vectors: {len(svr_model.support_)}")

    # Plot predictions
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_eff[:400], label="True")
    plt.plot(y_test_pred[:400], label="Predicted")
    plt.title("ESN + Linear SVR on Reservoir States")
    plt.xlabel("Test sample index")
    plt.ylabel("Normalized target")
    plt.legend()
    plt.grid(True)
    plt.show()












if __name__=="__main__":
    input_data=InputData()
    value=input("Please choose a model : \n1.ESM\n2.SVESM\nchoice ->:")
    if str(value)=="1":
        main_esm(input_data)
    elif str(value)=="2":
        main_svesm(input_data)
    else:
        print("Please enter a valid input")





