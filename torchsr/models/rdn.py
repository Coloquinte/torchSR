# Implementation from https://github.com/sanghyun-son/EDSR-PyTorch
# Modified to match pretrained models from https://github.com/yjn870/RDN-pytorch
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [ 'rdn', 'rdn_a', 'rdn_b', ]

url = {
    'g64go64d16c8x2': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/rdn_x2.pt',
    'g64go64d16c8x3': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/rdn_x3.pt',
    'g64go64d16c8x4': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/rdn_x4.pt',
}

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.layers = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.lff = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.lff(self.layers(x)) + x


class RDN(nn.Module):
    def __init__(self, scale, G0, D, C, G, pretrained=False, map_location=None):
        super(RDN, self).__init__()
        self.scale = scale

        url_name = 'g{}go{}d{}c{}x{}'.format(G, G0, D, C, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        r = scale
        kSize = 3
        n_colors = 3
        self.D = D

        # Shallow feature extraction net
        self.sfe1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.sfe2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.rdbs = nn.ModuleList()
        for i in range(self.D):
            self.rdbs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.gff = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.upscale = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
            ])
        elif r == 4:
            self.upscale = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")
        self.output = nn.Conv2d(G, n_colors, kSize, padding=(kSize-1)//2, stride=1)

        if pretrained:
            self.load_pretrained(map_location=map_location)

    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        f__1 = self.sfe1(x)
        x  = self.sfe2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            RDBs_out.append(x)

        x = self.gff(torch.cat(RDBs_out,1))
        x += f__1

        return self.output(self.upscale(x))

    def load_pretrained(self, map_location=None):
        if self.url is None:
            raise KeyError("No URL available for this model")
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        state_dict = load_state_dict_from_url(self.url, map_location=map_location, progress=True)
        self.load_state_dict(state_dict)


def rdn_a(scale, pretrained=False):
    return RDN(scale, G0=64, D=20, C=6, G=32, pretrained=pretrained)


def rdn_b(scale, pretrained=False):
    return RDN(scale, G0=64, D=16, C=8, G=64, pretrained=pretrained)


def rdn(scale, pretrained=False):
    return rdn_b(scale, pretrained)
