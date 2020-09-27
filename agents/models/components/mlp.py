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
        """

        :param layer_sizes: Size for each linear layer
        :param activation_function: Activation between layers
        :param output_function: Any output torch.nn.Module to be applied at the head
        :param dropout: Dropout for linear layers
        :param seed: Random seed
        :param hidden_layer_initialization_fn: How to initialize hidden linear layers
        :param output_layer_initialization_fn: How to initialize the last layer of the MLP defined my layer_sizes
        :param with_batchnorm: Apply batchnorm between linear layers

        Order is always (input_bn)->FC->BN->Activation->Dropout->FC
        """
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("Must provide at least 2 layer sizes")
        if seed:
            self.set_seed(seed)

        mlp_layers = torch.nn.ModuleList([])

        # Input BN
        if with_batchnorm:
            mlp_layers.append(torch.nn.BatchNorm1d(layer_sizes[0]))

        # HL 1
        first_layer = torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        if hidden_layer_initialization_fn:
            first_layer.weight.data.uniform_(*hidden_layer_initialization_fn(first_layer))

        mlp_layers.append(first_layer)

        # HL 2-N
        previous_output = layer_sizes[1]
        for n_out in layer_sizes[2:]:
            # BN
            if with_batchnorm:
                mlp_layers.append(torch.nn.BatchNorm1d(previous_output))

            # Activation
            mlp_layers.append(activation_function)

            # Dropout
            if dropout:
                mlp_layers.append(torch.nn.Dropout(dropout))

            # Next FC
            next_layer = torch.nn.Linear(previous_output, n_out)
            if hidden_layer_initialization_fn:
                next_layer.weight.data.uniform_(*hidden_layer_initialization_fn(next_layer))
            mlp_layers.append(next_layer)

            previous_output = n_out

        if output_layer_initialization_fn:
            mlp_layers[-1].weight.data.uniform_(*output_layer_initialization_fn(mlp_layers[-1]))
            mlp_layers[-1].bias.data.uniform_(*output_layer_initialization_fn(mlp_layers[-1]))

        # Apply output function -- Can be an Activation or a module
        if output_function:
            mlp_layers.append(output_function)

        # Stack
        self.mlp_layers = torch.nn.Sequential(*mlp_layers)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.mlp_layers.forward(x)
        return x
