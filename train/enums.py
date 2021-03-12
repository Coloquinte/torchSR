
from enum import Enum

class CEnum(Enum):
    def __str__(self):
        return self.value

class BlockType(CEnum):
    Residual = 'residual'
    Dense = 'dense'
    MBConv = 'mbconv'

class BackboneType(CEnum):
    Sequential = 'sequential'
    Dense = 'dense'

class UpsamplerType(CEnum):
    Direct = 'direct'
    Conv = 'conv'
    ShortConv = 'sconv'

class ActivationType(CEnum):
    ReLU = 'relu'
    LeakyReLU = 'leaky'
    SiLU = 'silu'

class SkipConnectionType(CEnum):
    No = 'no'
    Features = 'features'
    Nearest = 'nearest'
    Linear = 'linear'
    Bicubic = 'bicubic'

class DatasetType(CEnum):
    Div2KBicubic = 'div2k_bicubic'
    Div2KUnknown = 'div2k_unknown'
    Set5 = 'set5'
    Set14 = 'set14'
    B100 = 'b100'
    Urban100 = 'urban100'

class LossType(CEnum):
    L1 = "l1"
    SmoothL1 = "smooth_l1"
    L2 = "l2"
