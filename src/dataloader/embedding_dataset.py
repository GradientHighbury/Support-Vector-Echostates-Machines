from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch



class DelayEmbeddingDataset(Dataset):
    """
    Builds the training pairs:
            d(k) = [x(k),x(k-tau),x(k-2*tau),---]
            target =  x (k + horizon)
    """

    def __init__(self,series:np.ndarray,embedded_dim:int,delay:int,horizon:int) -> None:
        super().__init__()
        self.inputs =[]
        self.target =[]


        max_back=(embedded_dim-1)*delay

        start= max_back
        end = len(series)-horizon

        for k in range(start,end):
            d_k = [series[k-i*delay ] for i in range(embedded_dim)]
            target=  series[k+horizon]
            self.inputs.append(d_k)
            self.target.append(target)

    def __len__(self) ->int:
        return (len(self.inputs))
    
    def __getitem__(self, key:int):
        x = self.inputs[key]
        y = self.target[key]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return(x,y)