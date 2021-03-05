import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F


__all__ = ('ToTensor', 'ToPILImage', 'Compose', 'RandomHorizontalFlip', 'RandomVerticalFlip')


def apply_all(x, func):
    if isinstance(x, list) or isinstance(x, tuple):
        return [func(t) for t in x]
    else:
        return func(x)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    def __call__(self, x):
        return apply_all(x, F.to_tensor)


class ToPILImage:
    def __call__(self, x):
        return apply_all(x, F.to_pil_image)


class RandomCrop(nn.Module):
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        # TODO
        pass


class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        pass

    def forward(self, x):
        # TODO
        pass


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            x = apply_all(x, F.hflip)
        return x


class RandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            x = apply_all(x, F.vflip)
        return x

