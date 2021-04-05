
import math
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

__all__ = [ 'ninasr_b0', ]

url = {
    'r10f16x2': 'https://drive.google.com/uc?export=download&id=1WuR2uehZlTrp2Jv6KJxv1UWpTGOrX5sR',
    'r10f16x3': 'https://drive.google.com/uc?export=download&id=1YlfujNRg4Cw2A6FNmtEa2B6weK59o2TW',
    'r10f16x4': 'https://drive.google.com/uc?export=download&id=1EPUERFrR0eluSQflv73cFJ0XmpAQMF7e',
    'r10f16x8': 'https://drive.google.com/uc?export=download&id=1I-qS7fDGGBgpuA6TNz_eBvSKf5AWGfyI',
}

class AttentionBlock(nn.Module):
    """
    A typical Squeeze-Excite attention block, with a local pooling instead of global
    """
    def __init__(self, n_feats, reduction=4, stride=8):
        super(AttentionBlock, self).__init__()
        m = []
        m.append(nn.AvgPool2d(2*stride-1, stride=stride, padding=stride-1, count_include_pad=False))
        m.append(nn.Conv2d(n_feats, n_feats//reduction, 1, bias=True))
        m.append(nn.ReLU(True))
        m.append(nn.Conv2d(n_feats//reduction, n_feats, 1, bias=True))
        m.append(nn.Sigmoid())
        m.append(nn.Upsample(scale_factor=stride, mode='nearest'))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        if res.shape != x.shape:
            res = res[:, :, x.shape[2], x.shape[3]]
        return res * x


class ResBlock(nn.Module):
    def __init__(self, n_feats, mid_feats):
        super(ResBlock, self).__init__()

        m = []
        m.append(nn.Conv2d(n_feats, mid_feats, 3, bias=True, padding=1))
        m.append(nn.ReLU(True))
        m.append(AttentionBlock(mid_feats))
        m.append(nn.Conv2d(mid_feats, n_feats, 3, bias=True, padding=1))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Rescale(nn.Module):
    def __init__(self, sign):
        super(Rescale, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        bias = sign * torch.Tensor(rgb_mean).reshape(1, 3, 1, 1)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return x + self.bias


class NinaSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, expansion, scale, pretrained=False):
        super(NinaSR, self).__init__()

        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.filename = f"ninasr_{url_name}.pt"

        n_colors = 3
        self.head = NinaSR.make_head(n_colors, n_feats)
        self.body = NinaSR.make_body(n_resblocks, n_feats, expansion)
        self.tail = NinaSR.make_tail(n_colors, n_feats, scale)

        if pretrained:
            self.load_pretrained()

    @staticmethod
    def make_head(n_colors, n_feats):
        m_head = [
            Rescale(-1),
            nn.Conv2d(n_colors, n_feats, 3, padding=1, bias=False),
        ]
        return nn.Sequential(*m_head)

    @staticmethod
    def make_body(n_resblocks, n_feats, expansion):
        mid_feats = int(n_feats*expansion)
        m_body = [
            ResBlock(n_feats, mid_feats)
            for i in range(n_resblocks)
        ]
        return nn.Sequential(*m_body)

    @staticmethod
    def make_tail(n_colors, n_feats, scale):
        m_tail = [
            nn.Conv2d(n_feats, n_colors * scale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            Rescale(1)
        ]
        return nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

    def load_pretrained(self):
        if self.url is None:
            raise KeyError("No URL available for this model")
        state_dict = load_state_dict_from_url(self.url, progress=True, file_name=self.filename)
        self.load_state_dict(state_dict)


def ninasr_b0(scale, pretrained=False):
    model = NinaSR(10, 16, 2.0, scale, pretrained)
    return model

