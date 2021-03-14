import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import math
import PIL
import numbers


__all__ = ('ToTensor', 'ToPILImage', 'Compose', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomFlipTurn', 'RandomCrop', 'CenterCrop', 'ColorJitter', 'GaussianBlur')


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


def image_gcd_size(x):
    if isinstance(x, list) or isinstance(x, tuple):
        w, h = get_image_size(x[0])
        for img in x:
            img_w, img_h = get_image_size(img)
            w = math.gcd(w, img_w)
            h = math.gcd(h, img_h)
        return (w, h)
    else:
        return get_image_size(x)


def get_common_crop_size(crop_size, hr_size, common_size):
    width_scale = hr_size[0] // common_size[0]
    height_scale = hr_size[1] // common_size[1]
    if crop_size[0] % width_scale != 0:
        raise ValueError(f"Crop width {self.size[0]} is incompatible with the required scale {width_scale}")
    if crop_size[1] % height_scale != 0:
        raise ValueError(f"Crop height {self.size[1]} is incompatible with the required scale {height_scale}")
    crop_width = crop_size[0] // width_scale
    crop_height = crop_size[1] // height_scale
    common_crop_size = (crop_width, crop_height)
    if common_crop_size[0] > common_size[0] or common_crop_size[1] > common_size[1]:
        raise ValueError(f"Crop size {self.size} is too large for {img.size}")
    return common_crop_size


def apply_crop(img, common_size, common_crop_region):
    i, j, th, tw = common_crop_region
    width, height = get_image_size(img)
    assert width % common_size[0] == 0
    assert height % common_size[1] == 0
    width_scale = width // common_size[0]
    height_scale = height // common_size[1]
    return F.crop(img, i * height_scale, j * width_scale, th * height_scale, tw * width_scale)


def random_uniform(minval, maxval):
    return float(torch.empty(1).uniform_(minval, maxval))


def random_uniform_none(bounds):
    if bounds is None:
        return None
    return random_uniform(bounds[0], bounds[1])


def param_to_tuple(param, name, center=1.0, bounds=(0.0, float("inf"))):
    if isinstance(param, (list, tuple)):
        if len(param) != 2:
            raise ValueError(f"{name} must have two bounds")
        return (max(bounds[0], param[0]), min(bounds[1], param[1]))
    if not isinstance(param, numbers.Number):
        raise ValueError("f{name} must be a number or a pair")
    if param == 0:
        return None
    minval = max(center - param, bounds[0])
    maxval = min(center + param, bounds[1])
    return (minval, maxval)


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

    def forward(self, x):
        hr_img = first_image(x)
        # This size determines a valid cropping region
        common_size = image_gcd_size(x)
        common_crop_size = get_common_crop_size(self.size, get_image_size(hr_img), common_size)
        w, h = common_size
        tw, th = common_crop_size
        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        common_crop_region = (i, j, th, tw)
        return apply_all(x, lambda y: apply_crop(y, common_size, common_crop_region))


class CenterCrop(nn.Module):
    """Crop the center of the given images
    The location is chosen so that all images are cropped at a pixel boundary,
    even if they have different resolutions.
    """

    def __init__(self, size):
        super(CenterCrop, self).__init__()
        self.size = to_tuple(size, 2, "CenterCrop.size")
        # TODO: other torchvision.transforms.CenterCrop options

    def forward(self, x):
        hr_img = first_image(x)
        # This size determines a valid cropping region
        common_size = image_gcd_size(x)
        common_crop_size = get_common_crop_size(self.size, get_image_size(hr_img), common_size)
        w, h = common_size
        tw, th = common_crop_size
        i = (h - th) // 2
        j = (w - tw) // 2
        common_crop_region = (i, j, th, tw)
        return apply_all(x, lambda y: apply_crop(y, common_size, common_crop_region))


class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
        self.brightness = param_to_tuple(brightness, 'ColorJitter.brightness')
        self.contrast = param_to_tuple(contrast, 'ColorJitter.contrast')
        self.saturation = param_to_tuple(saturation, 'ColorJitter.saturation')
        self.hue = param_to_tuple(hue, 'ColorJitter.hue', center=0, bounds=[-0.5, 0.5])

    def get_params(self):
        brightness_factor = random_uniform_none(self.brightness)
        contrast_factor = random_uniform_none(self.contrast)
        saturation_factor = random_uniform_none(self.saturation)
        hue_factor = random_uniform_none(self.hue)
        return (brightness_factor, contrast_factor, saturation_factor, hue_factor)

    def apply_jitter(self, img, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        if brightness_factor is not None:
            img = F.adjust_brightness(img, brightness_factor)
        if contrast_factor is not None:
            img = F.adjust_contrast(img, contrast_factor)
        if saturation_factor is not None:
            img = F.adjust_saturation(img, saturation_factor)
        if hue_factor is not None:
            img = F.adjust_hue(img, hue_factor)
        return img

    def forward(self, x):
        brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params()
        return apply_all(x, lambda y: self.apply_jitter(y, brightness_factor, contrast_factor, saturation_factor, hue_factor))


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


class RandomFlipTurn(nn.Module):
    def __init__(self):
        super(RandomFlipTurn, self).__init__()

    def forward(self, x):
        if torch.rand(1) < 0.5:
            x = apply_all(x, F.vflip)
        if torch.rand(1) < 0.5:
            x = apply_all(x, F.hflip)
        if torch.rand(1) < 0.5:
            x = apply_all(x, lambda y: F.rotate(y, 90))
        return x


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=None, sigma=(0.1, 2.0), isotropic=False):
        super(GaussianBlur, self).__init__()
        self.kernel_size = None if kernel_size is None else to_tuple(kernel_size)
        self.sigma = param_to_tuple(sigma, 'GaussianBlur.sigma')
        self.isotropic = isotropic

    def forward(self, x):
        if self.isotropic:
            sigma_x = random_uniform(self.sigma[0], self.sigma[1])
            sigma_y = sigma_x
        else:
            sigma_x = random_uniform(self.sigma[0], self.sigma[1])
            sigma_y = random_uniform(self.sigma[0], self.sigma[1])
        sigma = (sigma_x, sigma_y)
        if self.kernel_size is not None:
            kernel_size = self.kernel_size
        else:
            k_x = max(2*int(math.ceil(3*sigma_x))+1, 3)
            k_y = max(2*int(math.ceil(3*sigma_y))+1, 3)
            kernel_size = (k_x, k_y)
        return apply_all(x, lambda y: F.gaussian_blur(y, kernel_size, sigma))



