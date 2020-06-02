import torch
import torch.nn as nn
from typing import List
from .misc import Flatten

class CNN(torch.nn.Module):
    """ Helper module for creating CNNs """
    def __init__(
            self,
            state_shape,
            num_stacked_frames: int,
            grayscale: bool,
            nfilters: tuple = (32, 64, 64),
            kernel_sizes: List[tuple] = ((1, 8, 8), (1, 4, 4), (4, 3, 3)),
            stride_sizes=((1, 4, 4), (1, 2, 2), (1, 1, 1)),
            **kwargs
    ):
        super().__init__()
        self.state_shape = state_shape
        self.grayscale = grayscale
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.nfilters = nfilters
        self.num_stacked_frames = num_stacked_frames

        self.features = self.get_featurizer()

        if self.grayscale:
            state_shape = (1, self.num_stacked_frames, 84, 84)
        else:
            state_shape = (1, 3, self.num_stacked_frames, 84, 84)

        self.output_size = self.output_feature_size(state_shape)

    def get_featurizer(self):
        if self.grayscale:  # Single channel
            conv1 = nn.Conv2d(self.num_stacked_frames, self.nfilters[0], kernel_size=self.kernel_sizes[0], stride=self.stride_sizes[0])
            bn1 = nn.BatchNorm2d(self.nfilters[0])
            conv2 = nn.Conv2d(self.nfilters[0], self.nfilters[1], kernel_size=self.kernel_sizes[1], stride=self.stride_sizes[1])
            bn2 = nn.BatchNorm2d(self.nfilters[1])
            conv3 = nn.Conv2d(self.nfilters[1], self.nfilters[2], kernel_size=self.kernel_sizes[2], stride=self.stride_sizes[2])
            bn3 = nn.BatchNorm2d(self.nfilters[2])
        else:  # RGB
            conv1 = nn.Conv3d(3, self.nfilters[0], kernel_size=self.kernel_sizes[0], stride=self.stride_sizes[0])
            bn1 = nn.BatchNorm3d(self.nfilters[0])
            conv2 = nn.Conv3d(self.nfilters[0], self.nfilters[1], kernel_size=self.kernel_sizes[1],
                              stride=self.stride_sizes[1])
            bn2 = nn.BatchNorm3d(self.nfilters[1])
            conv3 = nn.Conv3d(self.nfilters[1], self.nfilters[2], kernel_size=self.kernel_sizes[2],
                              stride=self.stride_sizes[2])
            bn3 = nn.BatchNorm3d(self.nfilters[2])

        class Featurizer(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.model = nn.Sequential(
                    conv1,
                    bn1,
                    nn.ReLU(),
                    conv2,
                    bn2,
                    nn.ReLU(),
                    conv3,
                    bn3,
                    nn.ReLU(),
                    Flatten()
                )

            def forward(self, state: torch.Tensor):
                return self.model.forward(state)

        return Featurizer()

    def output_feature_size(self, shape):
        x = torch.rand(shape)
        x = self.features(x)
        return x.data.view(1, -1).size(1)

    def forward(self, x: torch.Tensor):
        return self.features(x)
