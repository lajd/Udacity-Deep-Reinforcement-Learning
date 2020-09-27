from typing import Callable, Optional


class ParameterScheduler:
    def __init__(self, initial: float, lambda_fn: Optional[Callable[[int], float]] = None, final: Optional[float] = None, ):
        self.initial = initial
        self.final = final
        self.lambda_fn = lambda_fn

        self.value = initial

    def get_param(self, i):
        if not i:
            return self.initial

        if self.final:
            asc = self.final > self.initial
            if asc:
                return min(self.final, self.lambda_fn(i))
            else:
                return max(self.final, self.lambda_fn(i))
        else:
            return self.lambda_fn(i)


class LinearDecaySchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
