
import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [ 'ninasr_b0', 'ninasr_b1', 'ninasr_b2' ]

url = {
    'r10f16x2': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x2.pt',
    'r10f16x3': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x3.pt',
    'r10f16x4': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x4.pt',
    'r10f16x8': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x8.pt',
    'r26f32x2': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x2.pt',
    'r26f32x3': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x3.pt',
    'r26f32x4': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x4.pt',
    'r26f32x8': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x8.pt',
    'r84f56x2': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x2.pt',
    'r84f56x3': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x3.pt',
    'r84f56x4': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x4.pt',
    'r84f56x8': 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x8.pt',
}

class AttentionBlock(nn.Module):
    """
    A typical Squeeze-Excite attention block, with a local pooling instead of global
    """
    def __init__(self, n_feats, reduction=4, stride=16):
        super(AttentionBlock, self).__init__()
        self.body = nn.Sequential(
            nn.AvgPool2d(2*stride-1, stride=stride, padding=stride-1, count_include_pad=False),
            nn.Conv2d(n_feats, n_feats//reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_feats//reduction, n_feats, 1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=stride, mode='nearest')
        )

    def forward(self, x):
        res = self.body(x)
        if res.shape != x.shape:
            res = res[:, :, :x.shape[2], :x.shape[3]]
        return res * x


class ResBlock(nn.Module):
    def __init__(self, n_feats, mid_feats, in_scale, out_scale):
        super(ResBlock, self).__init__()

        self.in_scale = in_scale
        self.out_scale = out_scale

        m = []
        conv1 = nn.Conv2d(n_feats, mid_feats, 3, padding=1, bias=True)
        nn.init.kaiming_normal_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        m.append(conv1)
        m.append(nn.ReLU(True))
        m.append(AttentionBlock(mid_feats))
        conv2 = nn.Conv2d(mid_feats, n_feats, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(conv2.weight)
        #nn.init.zeros_(conv2.weight)
        m.append(conv2)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x * self.in_scale) * (2*self.out_scale)
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
    def __init__(self, n_resblocks, n_feats, scale, pretrained=False, map_location=None, expansion=2.0):
        super(NinaSR, self).__init__()
        self.scale = scale

        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        n_colors = 3
        self.head = NinaSR.make_head(n_colors, n_feats)
        self.body = NinaSR.make_body(n_resblocks, n_feats, expansion)
        self.tail = NinaSR.make_tail(n_colors, n_feats, scale)

        if pretrained:
            self.load_pretrained(map_location=map_location)

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
        out_scale = 4 / n_resblocks
        expected_variance = 1.0
        m_body = []
        for i in range(n_resblocks):
            in_scale = 1.0/math.sqrt(expected_variance)
            m_body.append(ResBlock(n_feats, mid_feats, in_scale, out_scale))
            expected_variance += out_scale ** 2
        return nn.Sequential(*m_body)

    @staticmethod
    def make_tail(n_colors, n_feats, scale):
        m_tail = [
            nn.Conv2d(n_feats, n_colors * scale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            Rescale(1)
        ]
        return nn.Sequential(*m_tail)

    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
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


def ninasr_b0(scale, pretrained=False):
    model = NinaSR(10, 16, scale, pretrained)
    return model

def ninasr_b1(scale, pretrained=False):
    model = NinaSR(26, 32, scale, pretrained)
    return model

def ninasr_b2(scale, pretrained=False):
    model = NinaSR(84, 56, scale, pretrained)
    return model

