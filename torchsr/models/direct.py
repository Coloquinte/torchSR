import torch.nn as nn


__all__ = ['nearest', 'bilinear', 'bicubic' ]


def nearest(scale, pretrained=None):
    return nn.Upsample(scale_factor=scale, mode='nearest')


def bilinear(scale, pretrained=None):
    return nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)


def bicubic(scale, pretrained=None):
    return nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)

