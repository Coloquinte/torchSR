
import math
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

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
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats//reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_feats//reduction, n_feats, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.body(x)
        return res * x


class ScaledConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 scale=1.0, eps=1.0e-6):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.scale = scale / self.weight[0].numel() ** 0.5
        self.gain = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_weight(self):
        mean = torch.mean(self.weight, dim=(1, 2, 3), keepdim=True)
        std = torch.std(self.weight, dim=(1, 2, 3), keepdim=True)
        return (self.scale * self.gain) * (self.weight - mean) / (std + self.eps)

    def forward(self, x):
        return nn.functional.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResBlock(nn.Module):
    def __init__(self, n_feats, mid_feats, in_scale, out_scale, attention):
        super(ResBlock, self).__init__()

        scale1 = in_scale * 1.7139588594436646
        scale2 = out_scale * 2.0 if attention else out_scale

        m = []
        conv1 = ScaledConv2d(n_feats, mid_feats, 3, padding=1, bias=True, scale=scale1)
        m.append(conv1)
        m.append(nn.ReLU(True))
        if attention:
            m.append(AttentionBlock(mid_feats))
        conv2 = ScaledConv2d(mid_feats, n_feats, 3, padding=1, bias=False, scale=scale2)
        m.append(conv2)

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
    def __init__(self, n_resblocks, n_feats, scale, pretrained=False, expansion=2.0, attention=True):
        super(NinaSR, self).__init__()
        self.scale = scale

        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        n_colors = 3
        self.head = NinaSR.make_head(n_colors, n_feats)
        self.body = NinaSR.make_body(n_resblocks, n_feats, expansion, attention)
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
    def make_body(n_resblocks, n_feats, expansion, attention):
        mid_feats = int(n_feats*expansion)
        out_scale = 0.2
        expected_variance = 1.0
        m_body = []
        for i in range(n_resblocks):
            in_scale = 1.0/math.sqrt(expected_variance)
            m_body.append(ResBlock(n_feats, mid_feats, in_scale, out_scale, attention))
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

    def load_pretrained(self):
        if self.url is None:
            raise KeyError("No URL available for this model")
        state_dict = load_state_dict_from_url(self.url, progress=True)
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

