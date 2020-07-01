import math
import torch
from typing import Optional, Type
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NoisyMLP(nn.Module):
    """ Helper module for creating noisy fully connected layers """
    def __init__(
            self,
            layer_sizes: tuple,
            activation_function: torch.nn.Module = nn.ReLU(),
            output_function: Optional[torch.nn.Module] = None,
            dropout: Optional[float] = None,
    ):
        super().__init__()

        layers = torch.nn.ModuleList([NoisyLinear(layer_sizes[0], layer_sizes[1])])

        previous_output = layer_sizes[1]
        for n_out in layer_sizes[2:]:
            layers.append(activation_function)
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(NoisyLinear(previous_output, n_out))
            previous_output = n_out

        if output_function:
            layers.append(output_function)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def reset_noise(self):
        for module in self.model.modules():
            if hasattr(module, 'reset_noise'):
                module.reset_noise()


class NoisyLinear(nn.Module):
    """Create a noisy linear layer

    Adapted from https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
    """
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        weight_epsilon = self.weight_epsilon.to(device)
        bias_epsilon = self.bias_epsilon.to(device)

        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
