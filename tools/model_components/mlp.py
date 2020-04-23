import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MLP(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, linear_layer_sizes: Tuple[int, ...], activation_function='relu', output_layer=None,
                 layer_dropout=0.25):
        super(MLP, self).__init__()
        mlp_layers = torch.nn.ModuleList([torch.nn.Linear(linear_layer_sizes[0], linear_layer_sizes[1])])

        if activation_function == 'relu':
            activation = torch.nn.ReLU()
        else:
            raise ValueError('Only relu activation implemented')

        previous_output = linear_layer_sizes[1]
        for n_out in linear_layer_sizes[2:]:
            mlp_layers.append(activation)
            mlp_layers.append(torch.nn.Dropout(layer_dropout))
            mlp_layers.append(torch.nn.Linear(previous_output, n_out))
            previous_output = n_out

        if output_layer:
            if output_layer == 'sigmoid':
                output = nn.Sigmoid()
            else:
                raise ValueError('Only sigmoid output implemented')
            mlp_layers.append(output)

        self.mlp_layers = torch.nn.Sequential(*mlp_layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.mlp_layers.forward(state)
