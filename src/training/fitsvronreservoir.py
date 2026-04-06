from src.model.fixedreservoir import FixedReservoir
from src.training.training import nrmse


import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler



def fit_svr_on_reservoir(reservoir:FixedReservoir,X_train:np.ndarray,y_train:np.ndarray,
                         X_val:np.ndarray,y_val:np.ndarray,wash_out:int=50,device:str="cpu"):
    #transform inputs into reservoir states
    H_Train= reservoir.transform_sequence(X_train,wash_out=wash_out,device=device)
    H_val = reservoir.transform_sequence(X_val,wash_out=wash_out,device=device)

    y_train_eff = y_train[wash_out:]
    y_val_eff = y_val[wash_out:]


    #Scale the reservoir states before SVR
    state_scalar= StandardScaler()
    H_train_scaled = state_scalar.fit_transform(H_Train)
    H_val_scaled   = state_scalar.transform(H_val)


    #Small validation Search
    C_grid = [0.1, 1.0, 10.0, 50.0]
    eps_grid = [0.01, 0.05, 0.1]


    best_model = None
    best_cfg = None
    best_score = float('inf')

    Total_iterations=len(C_grid)*len(eps_grid)
    i=0
    for C in C_grid:
        for eps in eps_grid:
            i+=1
            print(f"Running Iteration:->  {i}/{Total_iterations}")
            model = SVR(kernel="linear",C=C,epsilon=eps)
            model.fit(H_train_scaled,y_train_eff)

            val_pred = model.predict(H_val_scaled)
            score = nrmse(torch.as_tensor(y_val_eff, dtype=torch.float32, device=device), torch.as_tensor(val_pred, dtype=torch.float32, device=device))
            if score < best_score:
                best_score = score
                best_model = model
                best_cfg = {"C": C, "epsilon": eps}

    return(best_model,state_scalar,best_cfg,best_score)


