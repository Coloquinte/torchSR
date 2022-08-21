# Implementation from https://github.com/nmhkahn/CARN-pytorch
# Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
# https://arxiv.org/abs/1803.08664

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = [ 'carn', 'carn_m' ]


urls = {
    'carn': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/carn.pt',
    'carn_m': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/carn_m.pt',
}


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, multi_scale, 
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, 
				 n_channels, scale, 
				 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class CARNBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(CARNBlock, self).__init__()

        self.b1 = ResidualBlock(64, 64)
        self.b2 = ResidualBlock(64, 64)
        self.b3 = ResidualBlock(64, 64)
        self.c1 = BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class CARN(nn.Module):
    def __init__(self, scale, pretrained=False, map_location=None):
        super(CARN, self).__init__()
        
        multi_scale = True
        group = 1
        self.scale = scale

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = CARNBlock(64, 64)
        self.b2 = CARNBlock(64, 64)
        self.b3 = CARNBlock(64, 64)
        self.c1 = BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsample = UpsampleBlock(64, scale=scale, 
                                      multi_scale=multi_scale,
                                      group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

        if pretrained:
            self.load_pretrained(map_location=map_location)
                
    def forward(self, x, scale=None):
        if self.scale is not None:
            if scale is not None and scale != self.scale:
                raise ValueError(f"Network scale is {self.scale}, not {scale}")
            scale = self.scale
        else:
            if scale is None:
                raise ValueError(f"Network scale was not set")
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out

    def load_pretrained(self, map_location=None):
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        state_dict = load_state_dict_from_url(urls["carn"], map_location=map_location, progress=True)
        self.load_state_dict(state_dict)


class CARNMBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(CARNMBlock, self).__init__()

        self.b1 = EResidualBlock(64, 64, group=group)
        self.c1 = BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class CARNM(nn.Module):
    def __init__(self, scale, pretrained=False, map_location=None):
        super(CARNM, self).__init__()
        
        multi_scale = True
        group = 4
        self.scale = scale

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = CARNMBlock(64, 64, group=group)
        self.b2 = CARNMBlock(64, 64, group=group)
        self.b3 = CARNMBlock(64, 64, group=group)
        self.c1 = BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsample = UpsampleBlock(64, scale=scale, 
                                      multi_scale=multi_scale,
                                      group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

        if pretrained:
            self.load_pretrained(map_location=map_location)
                
    def forward(self, x, scale=None):
        if self.scale is not None:
            if scale is not None and scale != self.scale:
                raise ValueError(f"Network scale is {self.scale}, not {scale}")
            scale = self.scale
        else:
            if scale is None:
                raise ValueError(f"Network scale was not set")
        scale = self.scale
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out

    def load_pretrained(self, map_location=None):
        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        state_dict = load_state_dict_from_url(urls["carn_m"], map_location=map_location, progress=True)
        self.load_state_dict(state_dict)


def carn(scale, pretrained=False):
    return CARN(scale, pretrained)


def carn_m(scale, pretrained=False):
    return CARNM(scale, pretrained)
