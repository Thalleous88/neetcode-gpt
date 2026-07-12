import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)

        n = len(y_true)
        w1 = len(W1)
        w2 = len(W1[0])
        out = W1 @ x.T + b1
        act = np.array([max(0, o) for o in out])
        y_pred = W2 @ act + b2
        loss = np.mean(np.square(y_pred - y_true))
        
        dmse_ypred = (2/n) * (y_pred - y_true)
        dypred_act = W2
        dypred_W2 = act
        dact_out = np.array([1 if o > 0 else 0 for o in out])
        dout_dW1 = x
        dypred_out = (dmse_ypred * dypred_act).flatten() * dact_out
        
        dW1 = dypred_out.reshape(w1, 1) @ dout_dW1.reshape(1, len(x))
        db1 = np.squeeze(dypred_act * dact_out * dmse_ypred)
        dW2 = np.array([dmse_ypred * dypred_W2])
        db2 = dmse_ypred

        res = {'loss': np.round(loss, 4), 'dW1': dW1.round(decimals=4), 'db1': np.round(db1, 4), 'dW2': dW2.round(decimals=4), 'db2': np.round(db2, 4)}

        return res

