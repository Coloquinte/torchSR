import torch
import torch.nn as nn

__all__ = [ 'ChoppedModel' ]

def get_windows(size, stride):
    starts = list(range(0, size, stride))
    starts[-1] = min(starts[-1], size - stride)
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
    stride = chop_size - chop_overlap
    x_starts = get_windows(width, stride)
    y_starts = get_windows(height, stride)
    result_shape = (x.shape[0], x.shape[1], scale*x.shape[2], scale*x.shape[3])
    result = torch.zeros(result_shape, device=x.device)
    for i, x_s in enumerate(x_starts):
        for j, y_s in enumerate(y_starts):
            x_e = x_s + chop_size
            y_e = y_s + chop_size
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
            r_margin = None if r_margin == 0 else -r_margin
            t_margin = None if t_margin == 0 else -t_margin
            tile = out[:, :, l_margin:r_margin, b_margin:t_margin]
            result[:, :, x_a:x_b, y_a:y_b] = tile
    return result


class ChoppedModel(nn.Module):
    """
    Wrapper to run a model on small image tiles in order to use less memory
    """
    def __init__(self, model, scale, chop_size, chop_overlap):
        super(ChoppedModel, self).__init__()
        self.model = model
        self.scale = scale
        self.chop_size = chop_size
        self.chop_overlap = chop_overlap

    def forward(self, x):
        return chop_and_forward(self.model, x, self.scale, self.chop_size, self.chop_overlap)

