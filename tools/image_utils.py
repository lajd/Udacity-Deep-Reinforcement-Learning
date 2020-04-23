import torch
from torch import nn
import numpy as np
from skimage import transform
from collections import deque

from typing import Tuple


class CNN2D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,  # Number of filters
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )

    def forward(self, inp: torch.Tensor):
        b_size, w_in, h_in, d_in = inp.shape
        w_out = (w_in - self.kernel_size + 2*self.padding) / self.stride + 1
        h_out = (h_in - self.kernel_size + 2*self.padding) / self.stride + 1
        d_out = self.out_channels
        print("w_in: {}".format(w_in))
        print("h_in: {}".format(h_in))
        print("d_in: {}".format(d_in))
        print("W_out: {}".format(w_out))
        print("h_out: {}".format(h_out))
        print("d_out: {}".format(d_out))
        return self.cnn(inp)

class Image:
    def __init__(self, data: torch.Tensor):
        if data.dim() < 2:
            raise ValueError("Data must be at least 2 dimensional")
        elif data.dim() == 2:
            data = data.unsqueeze(2)
        data = data.unsqueeze(0)
        self.data = data

    def rgb_to_grayscale(self, img):
        """
        DeepMind took the maximum pixel value over subsequent frames to reduce flickering caused by the limitations
        of the Atari platform and then scale it from its current  210×160×3  resolution to  84×84 . We’ll do something
        similar, except convert the colors to grayscale rather than adjust based on the maximimum pixel value.

        To convert this, we will take the luminance channel (denoted as  Y ) from the image, which is the our
        RGB channel, and apply linear weights to the channel to transform it according to the relative luminance.
        """
        return np.dot(img[..., :3], [0.299, 0.587, 0.114])

    def resize_2d_image(self, img2d: torch.Tensor, new_size: Tuple[int, int]):
        img2d = transform.resize(img2d, new_size)
        return img2d


def image_list_to_cnn_input(images: list):
    """Images are expected to be tensors of shape (width, height, channels)"""
    image_tensor = torch.cat(images, dim=0)
    return image_tensor.permute(0, image_tensor.dim() - 1, *list(range(1, image_tensor.dim() - 1)))


class DeepCNN(torch.nn.Module):
    def __init__(self, tau: int = 4):
        super().__init__()
        self.tau = tau  # Number of image frames to stack
        self.state_buffer = deque(maxlen=tau)
        self.next_state_buffer = deque(maxlen=tau)

    def construct_model(self, ):
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.tau, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        pass


image = Image(torch.rand((84, 84)))
image_list = [image.data]
model = CNN2D(in_channels=1, out_channels=32, kernel_size=8, stride=1, padding=1)
cnn_input = image_list_to_cnn_input(image_list)

output = model.forward(cnn_input)