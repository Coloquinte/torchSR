
import math
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

__all__ = [ 'ninasr_x0', 'ninasr_x1', 'ninasr_x2',]

url = {
}

class AttentionBlock(nn.Module):
    """
    A typical Squeeze-Excite attention block, with a local pooling instead of global
    """
    def __init__(self, n_feats, reduction=4, stride=8):
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
        nn.init.zeros_(conv2.weight)
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


def spatial_features_1d(size, scale):
    # This is more natural than taking (0, size), as it doesn't align corners
    margin = 0.5 / scale
    return torch.linspace(margin, size-margin, int(round(scale*size)))


def spatial_features(shape, scale):
    # Construct the coordinates at each position
    width, height = shape[-2:]
    new_width, new_height = int(round(scale * width)), int(round(scale * height))
    x_features = spatial_features_1d(width, scale).unsqueeze(1).expand(new_width, new_height)
    y_features = spatial_features_1d(height, scale).unsqueeze(0).expand(new_width, new_height)
    return torch.stack([x_features, y_features], 0).unsqueeze(0)


class EmbeddingUpsampler(nn.Module):
    def __init__(self, n_colors, n_feats, n_resblocks, expansion):
        super(EmbeddingUpsampler, self).__init__()
        self.spatial_embed = nn.Conv2d(2, n_feats, 1)
        self.feature_embed = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.output = nn.Conv2d(n_feats, n_colors, 1)
        mid_feats = int(n_feats*expansion)
        self.resblocks = nn.ModuleList()
        for i in range(n_resblocks):
            seq = nn.Sequential(
                nn.Conv2d(2 * n_feats, mid_feats, 1),
                nn.ReLU(True),
                nn.Conv2d(mid_feats, n_feats, 1),
            )
            self.resblocks.append(seq)
                

    def forward(self, x, scale):
        # Feature embedding: simple 3x3 conv
        features = torch.nn.functional.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        features = self.feature_embed(features)
        # Spatial embedding: learnable cosinus coefficients
        spatial = spatial_features(x.shape, scale).to(features.device)
        spatial = self.spatial_embed(spatial)
        spatial = torch.cos(spatial)
        spatial = spatial.expand(features.shape[0], *spatial.shape[1:])
        for resblock in self.resblocks:
            f = torch.cat([spatial, features], dim=1)
            features += resblock(f)
        features = self.output(features)
        return features


class NinaSR_X(nn.Module):
    def __init__(self, n_resblocks, n_feats, pretrained=False, expansion=2.0):
        super(NinaSR_X, self).__init__()

        url_name = 'r{}f{}'.format(n_resblocks, n_feats)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        n_colors = 3
        self.head = NinaSR_X.make_head(n_colors, n_feats)
        self.body = NinaSR_X.make_body(n_resblocks, n_feats, expansion)
        self.tail = NinaSR_X.make_tail(n_colors, n_resblocks, n_feats, expansion)

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
        out_scale = 4 / n_resblocks
        expected_variance = 1.0
        m_body = []
        for i in range(n_resblocks):
            in_scale = 1.0/math.sqrt(expected_variance)
            m_body.append(ResBlock(n_feats, mid_feats, in_scale, out_scale))
            expected_variance += out_scale ** 2
        return nn.Sequential(*m_body)

    @staticmethod
    def make_tail(n_colors, n_resblocks, n_feats, expansion):
        return EmbeddingUpsampler(n_colors, n_feats, n_resblocks // 2, expansion)

    def forward(self, x, scale):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res, scale)
        return x

    def load_pretrained(self):
        if self.url is None:
            raise KeyError("No URL available for this model")
        state_dict = load_state_dict_from_url(self.url, progress=True)
        self.load_state_dict(state_dict)


def ninasr_x0(scale=None, pretrained=False):
    model = NinaSR_X(10, 16, pretrained)
    return model

def ninasr_x1(scale=None, pretrained=False):
    model = NinaSR_X(26, 32, pretrained)
    return model

def ninasr_x2(scale=None, pretrained=False):
    model = NinaSR_X(84, 56, pretrained)
    return model

