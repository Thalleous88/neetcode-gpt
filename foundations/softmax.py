import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        tot = sum(np.exp(z - max(z)))
        return np.array([np.round((np.exp(i - max(z))/tot), 4) for i in z])
