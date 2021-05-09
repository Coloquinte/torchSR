import math

import PIL
import torch
import torch.nn as nn

from typing import Any, Callable, List, Optional, Tuple, Union


__all__ = ['BoxDownscaler', 'BilinearDownscaler', 'BicubicDownscaler', 'LanczosDownscaler', 'DownscaledDataset', 'RandomDownscaledDataset']

class BoxDownscaler:
    def __call__(self, img, size):
        return img.resize(size, PIL.Image.BOX)


class BilinearDownscaler:
    def __call__(self, img, size):
        return img.resize(size, PIL.Image.BILINEAR)


class BicubicDownscaler:
    def __call__(self, img, size):
        return img.resize(size, PIL.Image.BICUBIC)


class LanczosDownscaler:
    def __call__(self, img, size):
        return img.resize(size, PIL.Image.LANCZOS)


def rescale_image(img, scale: Union[int, float], downscaler) -> List[Any]:
    if isinstance(img, list):
        if len(img) != 1:
            raise ValueError(f"Expecting a single image to downscale")
        img = img[0]
    # TODO: crop the HR before rescaling for some speedup
    hr = img
    # Rescale
    target_size = (int(math.ceil(hr.width / scale)), int(math.ceil(hr.height / scale)))
    lr = downscaler(hr, size=target_size)
    return [hr, lr]


class DownscaledDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 scale,
                 downscaler,
                 transform=None):
        self.dataset = dataset
        self.scale = scale
        self.downscaler = downscaler
        self.transform = transform

    def __getitem__(self, index: int) -> List[Any]:
        ret = rescale_image(self.dataset[index], self.scale, self.downscaler)
        if self.transform is not None:
            ret = self.transform(ret)
        return ret

    def __len__(self) -> int:
        return len(self.dataset)


class RandomDownscaledDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 scale_range,
                 downscaler,
                 transform=None):
        if not isinstance(scale_range, tuple) or len(scale_range) != 2 or scale_range[0] >= scale_range[1]:
            raise ValueError(f"Expected an ordered 2-tuple for scale_range")
        self.dataset = dataset
        self.scale_range = scale_range
        self.downscaler = downscaler
        self.transform = transform

    def __getitem__(self, index: int) -> List[Any]:
        # Sample uniformly by log of scale
        min_log, max_log = math.log(self.scale_range[0]), math.log(self.scale_range[1])
        scale = math.exp(torch.Tensor(1).uniform_(min_log, max_log))
        ret = rescale_image(self.dataset[index], scale, self.downscaler)
        if self.transform is not None:
            ret = self.transform(ret)
        return ret

    def __len__(self) -> int:
        return len(self.dataset)
