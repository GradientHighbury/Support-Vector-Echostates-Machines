# -------------------------------------------------
# 1) Mackey-Glass generator (Runge-Kutta method with a stepsize of 0.1
# -------------------------------------------------

import numpy as np

def generate_mackey_glass_rk2(length:int=7000,alpha:float=0.2,beta:float=-0.1,delta:int=17,power:int=10,h:float=0.1,seed:int=42)  -> np.ndarray:
    rng = np.random.default_rng(seed)  #Random number generation

    #delay buffer
    history_len=int(delta/h) +1 
    total_len= length+history_len +1

    x= np.zeros(total_len,dtype=np.float64)
    x[:history_len] = 1.2 + 0.2 * rng.random(history_len)
    # print(x)

    def f(x_t,x_delta):
        return(beta * x_t + (alpha * x_delta) / (1 + x_delta**10))

    for t in range(history_len,total_len-1):
        x_t=x[t]
        x_delta = x[t - history_len + 1]   # delayed value x(t - δ)
        k1 = f(x_t, x_delta)
        x_mid = x_t + 0.5 * h * k1
        k2 = f(x_mid, x_delta)

        x[t + 1] = x_t + h * k2
    
    return x[history_len:history_len + length].astype(np.float32)

 