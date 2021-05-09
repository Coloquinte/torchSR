import PIL

import torch
import torch.nn as nn

from typing import Any, Callable, List, Optional, Tuple, Union


class BoxDownscaler:
    def __call__(img, size):
        return img.resize(size, PIL.Image.BOX)


class BilinearDownscaler:
    def __call__(img, size):
        return img.resize(size, PIL.Image.BILINEAR)


class BicubicDownscaler:
    def __call__(img, size):
        return img.resize(size, PIL.Image.BICUBIC)


class LanczosDownscaler:
    def __call__(img, size):
        return img.resize(size, PIL.Image.LANCZOS)


class DownscaledDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 scale=None,
                 crop_size=None,
                 downscaler=None,
                 transform=None):
        self.dataset = dataset
        self.scale = scale
        self.crop_size = crop_size
        self.downscaler = downscaler
        self.transform = transform

    def __getitem__(self, index: int) -> List[Any]:
        img = self.dataset[index]
        # Crop HR to the scale we want
        if self.crop_size is None:
            hr = img
        else:
            hr = img # TODO
        target_size = (int(math.ceil(scale * hr.width)), int(math.ceil(scale * hr.height)))
        # Rescale
        lr = self.downscaler(hr, size=target_size)
        return [hr, lr]

    def __len__(self) -> int:
        return len(self.dataset)
