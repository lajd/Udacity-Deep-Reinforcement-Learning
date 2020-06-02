import torch
import torch.nn as nn
from typing import Tuple
from typing import Optional
from tools.misc import set_seed


class MLP(torch.nn.Module):
    """ Helper module for creating fully connected layers"""
    def __init__(
            self,
            layer_sizes: Tuple[int, ...],
            activation_function: torch.nn.Module = nn.ReLU(),
            output_function: Optional[torch.nn.Module] = None,
            dropout: Optional[float] = None,
            seed: int = None
    ):
        super(MLP, self).__init__()
        if seed:
            set_seed(seed)
        mlp_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[0], layer_sizes[1])])

        previous_output = layer_sizes[1]
        for n_out in layer_sizes[2:]:
            mlp_layers.append(activation_function)
            if dropout:
                mlp_layers.append(torch.nn.Dropout(dropout))
            mlp_layers.append(torch.nn.Linear(previous_output, n_out))
            previous_output = n_out

        if output_function:
            mlp_layers.append(output_function)

        self.mlp_layers = torch.nn.Sequential(*mlp_layers)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.mlp_layers.forward(x)
        return x
