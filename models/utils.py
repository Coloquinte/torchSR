import torch
import torch.nn as nn

__all__ = [ 'ChoppedModel', 'SelfEnsembleModel', 'ZeroPaddedModel', 'ReplicationPaddedModel', 'ReflectionPaddedModel' ]


class WrappedModel(nn.Module):
    """
    Generic model wrapper, overriding state_dict to avoid issues during save/restore
    """
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict)


def get_windows(tot_size, chop_size, chop_overlap):
    stride = chop_size - chop_overlap
    starts = list(range(0, tot_size - chop_overlap, stride))
    starts[-1] = min(starts[-1], tot_size - stride)  # Right-side
    starts[-1] = max(starts[-1], 0)  # Left-side, if there's only one element
    return starts


def chop_and_forward(model, x, scale, chop_size, chop_overlap):
    if x.ndim != 4:
        raise ValueError("Super-Resolution models expect a tensor with 4 dimensions")
    width = x.shape[2]
    height = x.shape[3]
    if chop_overlap > chop_size / 2:
        raise ValueError(f"Chop size {chop_size} is too small for overlap {chop_overlap}")
    if width <= chop_size and height <= chop_size:
        return model(x)
    x_starts = get_windows(width, chop_size, chop_overlap)
    y_starts = get_windows(height, chop_size, chop_overlap)
    result_shape = (x.shape[0], x.shape[1], scale*x.shape[2], scale*x.shape[3])
    result = torch.zeros(result_shape, device=x.device)
    for i, x_s in enumerate(x_starts):
        for j, y_s in enumerate(y_starts):
            # Range (saturated for when only one tile fits)
            x_e = min(x_s + chop_size, width)
            y_e = min(y_s + chop_size, height)
            # Run model on the tile
            out = model(x[:, :, x_s:x_e, y_s:y_e])
            # Compute margins
            l_margin = 0 if i == 0 else chop_overlap // 2
            r_margin = 0 if i == len(x_starts)-1 else chop_overlap - chop_overlap // 2
            b_margin = 0 if j == 0 else chop_overlap // 2
            t_margin = 0 if j == len(y_starts)-1 else chop_overlap - chop_overlap // 2
            l_margin *= scale
            r_margin *= scale
            b_margin *= scale 
            t_margin *= scale
            # Compute bounds for result
            x_a = scale*x_s + l_margin
            x_b = scale*x_e - r_margin
            y_a = scale*y_s + b_margin
            y_b = scale*y_e - t_margin
            # Update the result
            assert x_b > x_a and y_b > y_a
            r_margin = None if r_margin == 0 else -r_margin
            t_margin = None if t_margin == 0 else -t_margin
            tile = out[:, :, l_margin:r_margin, b_margin:t_margin]
            result[:, :, x_a:x_b, y_a:y_b] = tile
    return result


class ChoppedModel(WrappedModel):
    """
    Wrapper to run a model on small image tiles in order to use less memory

    Args:
        model (torch.nn.Module): The super-resolution model to wrap
        scale (int): the scaling factor
        chop_size (int): the size of the tiles, in pixels
        chop_overlap (int): the overlap between the tiles, in pixels
    """
    def __init__(self, model, scale, chop_size, chop_overlap):
        super(ChoppedModel, self).__init__(model)
        self.scale = scale
        self.chop_size = chop_size
        self.chop_overlap = chop_overlap

    def forward(self, x):
        return chop_and_forward(self.model, x, self.scale, self.chop_size, self.chop_overlap)


class SelfEnsembleModel(WrappedModel):
    """
    Wrapper to run a model with the self-ensemble method

    Args:
        model (torch.nn.Module): The super-resolution model to wrap
        median (boolean, optional): Use the median of the runs instead of the mean
    """
    def __init__(self, model, median=False):
        super(SelfEnsembleModel, self).__init__(model)
        self.median = median

    def forward_transformed(self, x, hflip, vflip, rotate):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = self.model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x

    def forward(self, x):
        t = []
        for hflip in [False, True]:
            for vflip in [False, True]:
                for rot in [False, True]:
                    t.append(self.forward_transformed(x, hflip, vflip, rot))
        t = torch.stack(t)
        if self.median:
            return torch.quantile(t, 0.5, dim=0)
        else:
            return torch.mean(t, dim=0)


def run_and_remove_pad(x, model, padding):
    w, h = x.shape[-2:]
    x = model(x)
    sw, sh = x.shape[-2:]
    if padding * sw % w != 0:
        raise ValueError(f"Padding {padding} is incompatible with scaling {w} --> {sw}")
    if padding * sh % h != 0:
        raise ValueError(f"Padding {padding} is incompatible with scaling {h} --> {sh}")
    px = padding * sw // w
    py = padding * sh // h
    return x[..., px:-px, py:-py]


class ZeroPaddedModel(WrappedModel):
    """
    Wrapper to add zero-padding to the input images
    """
    def __init__(self, model, padding):
        super(ZeroPaddedModel, self).__init__(model)
        self.padding = padding
        self.padder = nn.ZeroPad2d(padding)

    def forward(self, x):
        x = self.padder(x)
        return run_and_remove_pad(x, self.model, self.padding)


class ReplicationPaddedModel(WrappedModel):
    """
    Wrapper to add replication-padding to the input images
    """
    def __init__(self, model, padding):
        super(ReplicationPaddedModel, self).__init__(model)
        self.padding = padding
        self.padder = nn.ReplicationPad2d(padding)

    def forward(self, x):
        x = self.padder(x)
        return run_and_remove_pad(x, self.model, self.padding)


class ReflectionPaddedModel(WrappedModel):
    """
    Wrapper to add reflection-padding to the input images
    """
    def __init__(self, model, padding):
        super(ReflectionPaddedModel, self).__init__(model)
        self.padding = padding
        self.padder = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x = self.padder(x)
        return run_and_remove_pad(x, self.model, self.padding)
