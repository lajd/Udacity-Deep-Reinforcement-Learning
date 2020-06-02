import torch
from torch import nn
from skimage import transform
from skimage.color import rgb2hsv
from typing import Tuple
from matplotlib.pyplot import imshow, imsave

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RGBImage:
    def __init__(self, data: torch.Tensor):
        if data.dim() < 3:
            raise ValueError("Data must be at least 3 dimensional")
        elif data.dim() == 3:
            data = data.unsqueeze(0)
        self.data = data.float().to(device)

    def to_gray(self):
        """Convert an RGB (3 channel) image to grayscale (1 channel)

        https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
        """
        relative_luminance_tensor = torch.FloatTensor([0.299, 0.587, 0.114]).to(device)
        return torch.matmul(self.data[..., :3], relative_luminance_tensor)

    def to_hsv(self):
        # https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_rgb_to_hsv.html
        hsv_img = rgb2hsv(self.data.cpu().numpy())
        return torch.FloatTensor(hsv_img).to(device)

    def to_hue(self):
        hsv = self.to_hsv()
        hue = hsv[:, :, :, 0]
        return hue

    def to_value(self):
        hsv = self.to_hsv()
        value = hsv[:, :, :, 2]
        return value

    def resize_2d(self, img2d: torch.Tensor, new_size: Tuple[int, int]):
        img2d = transform.resize(img2d, new_size)
        return img2d

    def show(self, data: torch.Tensor = None, cmap: str = None):
        if data is None:
            data = self.data
        data = data.cpu().numpy().squeeze()
        imshow(data, cmap)

    def save(self, data: torch.Tensor = None, path: str = '', cmap: str = None):
        if data is None:
            data = self.data
        data = data.cpu().numpy().squeeze()
        imsave(path, data, cmap=cmap)
