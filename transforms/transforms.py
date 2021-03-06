import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import math
import PIL
import numbers


__all__ = ('ToTensor', 'ToPILImage', 'Compose', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomCrop')


def apply_all(x, func):
    if isinstance(x, list) or isinstance(x, tuple):
        return [func(t) for t in x]
    else:
        return func(x)


def first_image(x):
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) == 0:
            raise ValueError("Expected a non-empty image list")
        return x[0]
    else:
        return x


def smallest_image(x):
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) == 0:
            raise ValueError("Expected a non-empty image list")
        return x[0]
    else:
        return x


def to_tuple(sz, dim, name):
    if isinstance(sz, numbers.Number):
        return (sz,) * dim
    if isinstance(sz, tuple):
        if len(sz) == 1:
            return sz * dim
        elif len(sz) == dim:
            return sz
    raise ValueError(f"Expected a number of {dim}-tuple for {name}")


def get_image_size(img):
    if isinstance(img, PIL.Image.Image):
        return (img.width, img.height)
    if isinstance(img, torch.Tensor):
        return (img.shape[-1], img.shape[-2])
    raise ValueError("Unsupported image type")


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
    """Crop the given images at a common random location.
    The location is chosen so that all images are cropped at a pixel boundary,
    even if they have different resolutions.
    """

    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = to_tuple(size, 2, "RandomCrop.size")
        # TODO: other torchvision.transforms.RandomCrop options

    @staticmethod
    def gcd_size(x):
        if isinstance(x, list) or isinstance(x, tuple):
            w, h = (x[0].width, x[0].height)
            for img in x:
                w = math.gcd(w, img.width)
                h = math.gcd(h, img.height)
            return (w, h)
        else:
            return img.width, img.height

    def apply_crop(self, img, common_size, common_crop_region):
        i, j, th, tw = common_crop_region
        width, height = get_image_size(img)
        width_scale = width // common_size[0]
        height_scale = height // common_size[1]
        return F.crop(img, i * height_scale, j * width_scale, th * height_scale, tw * width_scale)

    def get_common_crop_size(self, hr_size, common_size):
        width_scale = hr_size[0] // common_size[0]
        height_scale = hr_size[1] // common_size[1]
        if self.size[0] % width_scale != 0:
            raise ValueError(f"Crop width {self.size[0]} is incompatible with the required scale {width_scale}")
        if self.size[1] % height_scale != 0:
            raise ValueError(f"Crop height {self.size[1]} is incompatible with the required scale {height_scale}")
        crop_width = self.size[0] // width_scale
        crop_height = self.size[1] // height_scale
        return (crop_width, crop_height)

    def forward(self, x):
        img = first_image(x)
        # This size determines a valid cropping region
        common_size = self.gcd_size(x)
        common_crop_size = self.get_common_crop_size(img.size, common_size)
        if common_crop_size[0] > common_size[0] or common_crop_size[1] > common_size[1]:
            raise ValueError(f"Crop size {self.size} is too large for {img.size}")
        w, h = common_size
        tw, th = common_crop_size
        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        common_crop_region = (i, j, th, tw)
        return apply_all(x, lambda y: self.apply_crop(y, common_size, common_crop_region))


class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
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

