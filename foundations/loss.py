import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)

        return np.round((-1/n) * sum(np.array([(y_true[i] * np.log(y_pred[i] + 1e-7) + (1-y_true[i]) * np.log(1-y_pred[i])) for i in range(n)])), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        s, c = y_true.shape
        res = 0
        for iter in range(s):
            res += sum(np.array([y_true[iter, i] * np.log(y_pred[iter, i]) for i in range(c)]))

        return np.round((-1/s) * res, 4)
