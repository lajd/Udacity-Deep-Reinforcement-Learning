import numpy as np
import random
from copy import copy
from typing import Tuple, Optional
from abc import abstractmethod
import torch


class Noise:
    def __init__(self):
        pass

    def reset(self, *args):
        pass

    def sample(self, *args):
        raise NotImplementedError

    def sample_like(self, *args):
        raise NotImplementedError


class OUNoise(Noise):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size: int, seed: int, mu: float = 0., theta: float = 0.15, sigma: float = 0.2, noise_clip: Tuple[int, int] = (-0.5, 0.5)):
        """Initialize parameters and noise process."""
        super().__init__()
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.noise_clip = noise_clip
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy(self.mu)

    def sample(self, *args):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class GaussianNoise(Noise):
    def __init__(self, scale: float = 0.2, clip: Optional[Tuple[float, float]] = (-0.5, 0.5)):
        super().__init__()
        self.scale = scale
        self.clip = clip

    def sample(self, x: torch.Tensor):
        output_as_numpy = False
        if isinstance(x, np.ndarray):
            output_as_numpy = True
            x = torch.from_numpy(x)
        noise = torch.randn_like(x) * self.scale
        if self.clip:
            noise = noise.clamp(self.clip[0], self.clip[1])
        if output_as_numpy:
            noise = noise.numpy()
        return noise


class RandomProcess(object):
    def reset_states(self):
        pass


class GaussianProcess(RandomProcess):
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        return np.random.randn(*self.size) * self.std()


class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)