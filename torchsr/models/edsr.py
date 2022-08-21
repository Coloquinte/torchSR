# Implementation from https://github.com/sanghyun-son/EDSR-PyTorch
# Enhanced Deep Residual Networks for Single Image Super-Resolution
# https://arxiv.org/abs/1707.02921

import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [ 'edsr_baseline', 'edsr', 'edsr_r16f64', 'edsr_r32f256', ]

url = {
    'r16f64x2': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr64_x2.pt',
    'r16f64x3': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr64_x3.pt',
    'r16f64x4': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr64_x4.pt',
    'r32f256x2': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr_x2.pt',
    'r32f256x3': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr_x3.pt',
    'r32f256x4': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr_x4.pt',
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


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, scale, res_scale, pretrained=False, map_location=None):
        super(EDSR, self).__init__()
        self.scale = scale

        kernel_size = 3 
        n_colors = 3
        rgb_range = 255
        conv=default_conv
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        if pretrained:
            self.load_pretrained(map_location=map_location)

    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = self.sub_mean(255 * x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x) / 255

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


def edsr_r16f64(scale, pretrained=False):
    return EDSR(16, 64, scale, 1.0, pretrained)


def edsr_r32f256(scale, pretrained=False):
    return EDSR(32, 256, scale, 0.1, pretrained)


def edsr_baseline(scale, pretrained=False):
    return edsr_r16f64(scale, pretrained)


def edsr(scale, pretrained=False):
    return edsr_r32f256(scale, pretrained)
