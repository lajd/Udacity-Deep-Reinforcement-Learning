from typing import Callable, Optional


class ParameterDecay:
    def __init__(self, initial: float, lambda_fn: Callable[[int], float], final: Optional[float] = None):
        self.initial = initial
        self.final = final
        self.lambda_fn = lambda_fn

        self.value = initial

    def get_param(self, i):
        self.value = self.value * self.lambda_fn(i)
        if self.final:
            self.value = max(self.value, self.final)
        return self.value
