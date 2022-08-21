# Implementation from https://github.com/sanghyun-son/EDSR-PyTorch
# Accurate Image Super-Resolution Using Very Deep Convolutional Networks
# https://arxiv.org/abs/1511.04587

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [ 'vdsr', 'vdsr_r20f64', ]

url = {
}

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class VDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, scale, pretrained, map_location=None):
        super(VDSR, self).__init__()
        self.scale = scale

        kernel_size = 3 
        n_colors = 3
        rgb_range = 1
        conv=default_conv

        self.scale = scale
        url_name = 'r{}f{}'.format(n_resblocks, n_feats)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        m_body = []
        m_body.append(conv(n_colors, n_feats, kernel_size))
        m_body.append(nn.ReLU(True))
        for _ in range(n_resblocks - 2):
            m_body.append(conv(n_feats, n_feats, kernel_size))
            m_body.append(nn.ReLU(True))
        m_body.append(conv(n_feats, n_colors, kernel_size))

        self.body = nn.Sequential(*m_body)

        if pretrained:
            self.load_pretrained(map_location=map_location)

    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.sub_mean(x)
        res = self.body(x)
        res += x
        x = self.add_mean(res)
        return x 

    def load_pretrained(self, map_location=None):
        if self.url is None:
            raise KeyError("No URL available for this model")
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        state_dict = load_state_dict_from_url(self.url, map_location=map_location, progress=True)
        self.load_state_dict(state_dict)


def vdsr_r20f64(scale, pretrained=False):
    return VDSR(20, 64, scale, pretrained)


def vdsr(scale, pretrained=False):
    return vdsr_r20f64(scale, pretrained)
