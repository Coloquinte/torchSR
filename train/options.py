import argparse

from .enums import *


parser = argparse.ArgumentParser(description='Super-Resolution networks')
train = parser.add_argument_group('Training')
data = parser.add_argument_group('Data')
hw = parser.add_argument_group('Harware')
model = parser.add_argument_group('Model')

# Model specification: network
model.add_argument("--arch", type=str,
                   help='network architecture to use')
model.add_argument('--download-pretrained', action='store_true',
                   help='download pretrained model')
model.add_argument('--load-checkpoint', type=str,
                   help='load model checkpoint')
model.add_argument('--save-checkpoint', type=str,
                   help='save model checkpoint')
#model.add_argument('--network-width', type=int, default=64,
#                   help='number of feature maps')
#model.add_argument('--network-depth', type=int, default=16,
#                   help='number of blocks')
#model.add_argument('--kernel-size', type=int, default=3,
#                   help='size of the convolution kernels')
#
## Model specification: structure
#model.add_argument('--upsampler', type=UpsamplerType, default=UpsamplerType.Conv,
#                   choices=list(UpsamplerType),
#                   help='upsampler type')
#model.add_argument('--skip-connection', type=SkipConnectionType, default=SkipConnectionType.Features,
#                   choices=list(SkipConnectionType),
#                   help='type of skip connection')
#
## Model specification: block
#model.add_argument('--block-type', type=BlockType, default=BlockType.Residual,
#                   choices=list(BlockType),
#                   help='block type')
#model.add_argument('--block-depth', type=int, default=2,
#                   help='depth of the block')
#model.add_argument('--block-expansion', type=float, default=1.0,
#                   help='block expansion ratio')
#model.add_argument('--block-scale-in', type=float, default=1.0,
#                   help='residual scaling (block input)')
#model.add_argument('--block-scale-out', type=float, default=1.0,
#                   help='residual scaling (block output)')
#
## Model specification: activation/normalization
#model.add_argument('--activation', type=ActivationType, default=ActivationType.ReLU,
#                   choices=list(ActivationType),
#                   help='activation function')
#model.add_argument('--batch-norm', action='store_true',
#                   help='use batch normalization')
#
## Model specification: attention
#model.add_argument('--attention', action='store_true',
#                   help='use channel attention')
#model.add_argument('--attention-field', type=int,
#                   help='attention field of view')
#model.add_argument('--attention-squeeze', type=float,
#                   help='attention squeeze factor')


# Training specification
train.add_argument('--scale', type=int, nargs='+', default=[2],
                   help='upsampling scale')
train.add_argument('--batch-size', type=int, default=16,
                   help='batch size')
train.add_argument('--epochs', type=int, default=6000,
                   help='number of epochs')
train.add_argument('--loss', type=LossType, default=LossType.L1,
                    choices=list(LossType),
                    help='training loss')
train.add_argument('--test-every', type=int, default=20,
                   help='number of training epochs between tests')
train.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
train.add_argument('--adam-betas', type=float, nargs=2, default=[0.9, 0.999],
                    help='adam momentum coefficients', metavar=("ADAM_BETA1", "ADAM_BETA2"))
train.add_argument('--adam-epsilon', type=float, default=1e-8,
                    help='adam epsilon for stability')
train.add_argument('--weight-decay', type=float, default=0.,
                    help='weight decay coefficient')
train.add_argument('--gradient-clipping', type=float,
                    help='clip the gradient values')
train.add_argument('--lr-decay-steps', type=int, nargs='+', default=[4000],
                    help='steps for learning rate decay')
train.add_argument('--lr-decay-rate', type=float, default=10.0,
                    help='learning rate decay per step')


# Dataset specification
data.add_argument('--evaluate', action='store_true',
                  help='only evaluate the model')
data.add_argument('--download-dataset', action='store_true',
                  help='download the dataset')
data.add_argument('--dataset-root', type=str, default='./data',
                  help='root directory for datasets')
data.add_argument('--dataset-train', nargs='+', default=[DatasetType.Div2KBicubic],
                  type=DatasetType, choices=[DatasetType.Div2KBicubic, DatasetType.Div2KUnknown],
                  help='Training dataset')
data.add_argument('--dataset-val', nargs='+', default=[DatasetType.Div2KBicubic],
                  type=DatasetType, choices=list(DatasetType),
                  help='Validation dataset')
data.add_argument('--patch-size-train', type=int, default=96,
                  help='image patch size for training (HR)')
data.add_argument('--patch-size-val', type=int, default=384,
                  help='image patch size for validation (HR)')
data.add_argument('--preload-dataset', action='store_true',
                  help='load the whole dataset in memory')


# Hardware specification
hw.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
hw.add_argument('--gpu', type=int, help='specify GPU id to use')
hw.add_argument('--datatype', type=DataType, default=DataType.FP32,
                choices=list(DataType),
                help='specify floating-point format')
hw.add_argument('--workers', type=int, default=0,
                help='number of workers for data loaders')

args = parser.parse_args()
if args.workers != 0 and args.preload_dataset:
    raise ValueError("Dataset preloading is incompatible with multiprocessing. "
                     "--worker argument cannot be given with --preload-dataset")

