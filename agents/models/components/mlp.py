import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
from agents.models.components import BaseComponent


class MLP(BaseComponent):
    """ Helper module for creating fully connected layers"""
    def __init__(
            self,
            layer_sizes: Tuple[int, ...],
            activation_function: torch.nn.Module = nn.ReLU(True),
            output_function: Optional[torch.nn.Module] = None,
            dropout: Optional[float] = None,
            seed: int = None,
            hidden_layer_initialization_fn: Optional[Callable] = None,
            output_layer_initialization_fn: Optional[Callable] = None,
            with_batchnorm: bool = False
    ):
        super().__init__()
        if seed:
            self.set_seed(seed)

        mlp_layers = torch.nn.ModuleList([])

        # Apply batchnorm to inputs of each layer
        if with_batchnorm:
            mlp_layers.append(torch.nn.BatchNorm1d(layer_sizes[0]))

        first_layer = torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        if hidden_layer_initialization_fn:
            first_layer.weight.data.uniform_(*hidden_layer_initialization_fn(first_layer))

        mlp_layers.append(first_layer)

        if len(layer_sizes) == 2:
            mlp_layers.append(activation_function)

        previous_output = layer_sizes[1]
        for n_out in layer_sizes[2:]:
            if with_batchnorm:
                mlp_layers.append(torch.nn.BatchNorm1d(previous_output))

            mlp_layers.append(activation_function)
            if dropout:
                mlp_layers.append(torch.nn.Dropout(dropout))

            next_layer = torch.nn.Linear(previous_output, n_out)
            if hidden_layer_initialization_fn:
                next_layer.weight.data.uniform_(*hidden_layer_initialization_fn(next_layer))

            mlp_layers.append(next_layer)

            previous_output = n_out

        if output_layer_initialization_fn:
            mlp_layers[-1].weight.data.uniform_(*output_layer_initialization_fn(mlp_layers[-1]))
            mlp_layers[-1].bias.data.uniform_(*output_layer_initialization_fn(mlp_layers[-1]))

        if output_function:
            mlp_layers.append(output_function)

        self.mlp_layers = torch.nn.Sequential(*mlp_layers)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.mlp_layers.forward(x)
        return x
