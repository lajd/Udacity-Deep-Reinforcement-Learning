import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple
from agents.models.components.misc import Flatten
from agents.models.components import BaseComponent


class CNN(BaseComponent):
    """ Helper module for creating CNNs """
    def __init__(
            self,
            image_shape,
            num_stacked_frames: int,
            grayscale: bool,
            filters: tuple = (32, 64, 64),
            kernel_sizes: Union[Tuple[int, ...], Tuple[tuple, ...]] = ((1, 8, 8), (1, 4, 4), (4, 3, 3)),
            stride_sizes=((1, 4, 4), (1, 2, 2), (1, 1, 1)),
            output_layer: Optional[nn.Module] = None,
            **kwargs
    ):
        super().__init__()
        self.grayscale = grayscale
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.filters = filters
        self.num_stacked_frames = num_stacked_frames
        self.output_layer = output_layer

        self.activation = nn.ReLU()
        self.features = self.get_featurizer()
        self.output = None

        assert len(kernel_sizes) == len(stride_sizes) == len(filters), "Must be the same number of kernels, strides and filters"

        if self.grayscale:
            state_shape = (1, self.num_stacked_frames, image_shape[0], image_shape[1])
        else:
            state_shape = (1, 3, self.num_stacked_frames, image_shape[0], image_shape[1])

        self.output_size = self.output_feature_size(state_shape)

    def set_output(self, output: nn.Module):
        self.output = output

    def get_featurizer(self):
        layers = []
        if self.grayscale:
            layers.append(nn.Conv2d(self.num_stacked_frames, self.filters[0], kernel_size=self.kernel_sizes[0], stride=self.stride_sizes[0]))
            layers.append(nn.BatchNorm2d(self.filters[0]))
            layers.append(self.activation)
            for i in range(1, len(self.filters)):
                layers.append(
                    nn.Conv2d(self.filters[i-1], self.filters[i], kernel_size=self.kernel_sizes[i], stride=self.stride_sizes[i])
                )
                layers.append(nn.BatchNorm2d(self.filters[i]))
                layers.append(self.activation)

        else:  # RGB
            # For 3D convolutions, set input channels to RGB
            layers.append(nn.Conv3d(3, self.filters[0], kernel_size=self.kernel_sizes[0], stride=self.stride_sizes[0]))
            layers.append(nn.BatchNorm3d(self.filters[0]))
            layers.append(self.activation)
            for i in range(1, len(self.filters)):
                layers.append(nn.Conv3d(self.filters[i-1], self.filters[i], kernel_size=self.kernel_sizes[i],
                          stride=self.stride_sizes[i]))
                layers.append(nn.BatchNorm3d(self.filters[i]))
                layers.append(self.activation)

        layers.append(Flatten())

        if self.output_layer:
            layers.append(self.output_layer)

        featurizer = nn.Sequential(*layers)

        return featurizer

    def output_feature_size(self, shape):
        x = torch.rand(shape)
        x = self.features(x)
        return x.data.view(1, -1).size(1)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        if self.output:
            x = self.output(x)
        return x
