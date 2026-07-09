class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        def f(x):
            return x**2

        def df(x):
            return 2 * x


        for i in range(iterations):
            init -= (learning_rate * df(init))

        return round(init, 5)
            
