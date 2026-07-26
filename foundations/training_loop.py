import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        sample, feature = X.shape
        w = np.zeros(feature)
        b = 0

        for i in range(epochs):
            y_hat = X @ w + b
            loss = (1/sample) * np.sum(np.square(y_hat-y))

            dloss_dw = (2/sample) * X.T @ (y_hat-y)
            dloss_db = (2/sample) * np.sum(y_hat-y)

            w -= (lr * dloss_dw)
            b -= (lr * dloss_db)

        return (np.round(w, 5), np.round(b, 5))



        

