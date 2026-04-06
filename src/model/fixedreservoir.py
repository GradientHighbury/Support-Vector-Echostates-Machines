import torch.nn as nn
import torch
import numpy as np


class FixedReservoir(nn.Module):
    def __init__(self,input_dim:int,reservoir_dim:int,spectral_radius:float=0.9,input_scale:float=0.5,sparsity:float=0.9,
                 leak_rate:float=1.0,seed:int=42) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)

        self.input_dim=input_dim
        self.reservoir_dim=reservoir_dim
        self.leak_rate=leak_rate

        #Input to reservoir Weights
        W_in = (torch.rand(reservoir_dim,input_dim,generator=g)*2-1)* input_scale

        #Random recuurent Weight
        W_res=torch.rand(reservoir_dim,reservoir_dim)*2.0-1.0

        #Sparsity mask:keep only a small fraction of connections
        mask = (torch.rand(reservoir_dim,reservoir_dim,generator=g)>sparsity).float()
        W_res=W_res*mask

        #Scale to desired spectral radius
        eigvals= torch.linalg.eigvals(W_res).abs()
        maxeg=torch.max(eigvals).real

        if maxeg>0:
            W_res=W_res*(spectral_radius/maxeg)

        self.register_buffer("W_in",W_in)
        self.register_buffer("W_res",W_res)

    
    def reservoir_step(self,u_t:torch.Tensor,h_prev:torch.Tensor)->torch.Tensor:
        # u_t:   (batch, input_dim)
        # h_prev:(batch, reservoir_dim)
        preact = u_t @ self.W_in.T + h_prev @ self.W_res.T
        h_tilde = torch.tanh(preact)

        h= (1-self.leak_rate)*h_prev+self.leak_rate*h_tilde

        return(h)
    
    def transform_sequence(self, X:np.ndarray,wash_out:int=50,device:str="cpu")->np.ndarray:
        """
        Drive the reservoir sequentially through X and return the state matrix H.
        This uses true recurrence across the ordered samples.
        
        """
          
        self.to(device)
        self.eval()

        X_t = torch.as_tensor(X,dtype=torch.float32,device=device)

        h=torch.zeros(1,self.reservoir_dim,device=device)
        states=[]

        with torch.no_grad():
            for i in range(X_t.shape[0]):
                u = X_t[i:i+1] # shape: (1, input_dim)
                h = self.reservoir_step(u,h)
                states.append(h.squeeze(0).cpu().numpy())
        
        H = np.asarray(states,dtype=np.float32)

        if wash_out>0:
            H = H[wash_out:]

        return H
