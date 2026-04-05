
import torch.nn as nn
import torch



class EchoStateNetwork(nn.Module):
    """
    Fixed random reservoir + trainable linear readout.
    This matches the ESN-style structure used before the SVR formulation.
    """

    def __init__(self,input_dim:int,reservior_dim:int,output_dim:int=1,spectral_radius:float=0.9,
                 input_scale:float=0.5,sparsity:float=0.9,leak_rate:float=1.0,seed:int=42) -> None:
        super().__init__()

        self.input_dim=input_dim
        self.reservior_dim=reservior_dim
        self.output_dim=output_dim
        self.leak_rate=leak_rate

        #To generate random numbers
        g = torch.Generator()
        g.manual_seed(seed)

        #input Weights:Win
        w_in= (torch.rand(reservior_dim,input_dim,generator=g)*2-1.0)*input_scale 

        #Reservior Weights
        w_res=torch.rand(reservior_dim,reservior_dim,generator=g)*2-1

        #Apply sparsity mask
        mask= (torch.rand(reservior_dim,reservior_dim,generator=g)>sparsity).float()
        w_res=w_res*mask


        #Scale to desired spectral radius
        eigen_vals=torch.linalg.eigvals(w_res).abs()
        max_eig=torch.max(eigen_vals).real

        if max_eig>0:
            w_res=w_res*(spectral_radius/max_eig)

        #Register as buffers so that they are fixed, not trained
        self.register_buffer("W_in",w_in)
        self.register_buffer("W_res",w_res)

        #Trainable readout
        self.read_out=nn.Linear(reservior_dim,output_dim)

    def reservior_step(self,u_t:torch.Tensor,h_prev:torch.Tensor)->torch.Tensor:
        """
        u_t:(batch, input_dim)
        h_prev:(batch,reservior_dim)"""

        #This computes the raw next hidden activation before tanh.
        preact = u_t @ self.W_in.T + h_prev @ self.W_res.T

        #Apply the nonlinear activation function
        h_tilde = torch.tanh(preact)

        #This is the leaky update
        h = (1.0-self.leak_rate)*h_prev+self.leak_rate*h_tilde

        return h
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        x: (batch, input_dim)

        Since delay embedding already compresses the history into one vector,
        we treat each embedded vector as one reservoir input step.
        """

        batch_size = x.size(0)
        h_0=torch.zeros(batch_size,self.reservior_dim,device=x.device)
        h=self.reservior_step(x,h_0)
        y = self.read_out(h)

        return y     