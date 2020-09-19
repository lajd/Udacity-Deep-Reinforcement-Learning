import torch
from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class SoftmaxSelection(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, input):
        bsize = len(input)
        x = self.softmax(input)
        x = torch.argmax(x, dim=-1).view(bsize, 1)
        return x


class BoundVectorNorm(nn.Module):
    def __init__(self, max_vector_norm: float = 10):
        super().__init__()
        self.max_vector_norm = max_vector_norm

    def forward(self, x):
        norm = torch.norm(x)
        x = self.max_vector_norm * (F.tanh(norm)) * x / norm if norm > 0 else self.max_vector_norm * x
        return x

