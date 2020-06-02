from typing import Callable, Optional


class ParameterScheduler:
    def __init__(self, initial: float, lambda_fn: Callable[[int], float], final: Optional[float] = None, ):
        self.initial = initial
        self.final = final
        self.lambda_fn = lambda_fn

        self.value = initial

    def get_param(self, i):
        if self.final:
            asc = self.final > self.initial
            if asc:
                return min(self.final, self.lambda_fn(i))
            else:
                return max(self.final, self.lambda_fn(i))
        else:
            return self.lambda_fn(i)
