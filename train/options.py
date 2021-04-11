import argparse

from .enums import *

parser = argparse.ArgumentParser(description='Super-Resolution networks')
train = parser.add_argument_group('Training')
data = parser.add_argument_group('Data')
hw = parser.add_argument_group('Harware')
model = parser.add_argument_group('Model')


# Model specification: network
model.add_argument("--arch", type=str, required=True,
                   help='network architecture to use')
model.add_argument('--download-pretrained', action='store_true',
                   help='download pretrained model')
load_ckp = model.add_mutually_exclusive_group()
load_ckp.add_argument('--load-checkpoint', type=str,
                      help='load model checkpoint')
load_ckp.add_argument('--load-pretrained', type=str,
                      help='load pretrained model')
model.add_argument('--save-checkpoint', type=str,
                   help='save model checkpoint')


# Training specification
train.add_argument('--scale', type=int, nargs='+', default=[2],
                   help='upsampling scale')
train.add_argument('--batch-size', type=int, default=16,
                   help='batch size')
train.add_argument('--epochs', type=int, default=300,
                   help='number of epochs')
train.add_argument('--loss', type=LossType, default=LossType.L1,
                   choices=list(LossType),
                   help='training loss')
train.add_argument('--optimizer', type=OptimizerType, default=OptimizerType.ADAM,
                   choices=list(OptimizerType),
                   help='optimizer')
train.add_argument('--lr', type=float, default=1e-4,
                   help='learning rate')
train.add_argument('--momentum', type=float,
                   help='momentum coefficient for SGD and RMSprop')
train.add_argument('--rmsprop-alpha', type=float,
                   help='smoothing coefficient for RMSprop')
train.add_argument('--adam-betas', type=float, nargs=2,
                   help='smoothing coefficients for Adam, AdamW and Adamax')
train.add_argument('--weight-decay', type=float,
                   help='weight decay coefficient')
train.add_argument('--gradient-clipping', type=float,
                   help='clip the gradient values')
train.add_argument('--lr-decay-steps', type=int, nargs='+', default=[200],
                   help='steps for learning rate decay')
train.add_argument('--lr-decay-rate', type=float, default=10.0,
                   help='learning rate decay per step')
train.add_argument('--log-dir', type=str,
                   help='log directory for tensorboard')


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
data.add_argument('--dataset-repeat', type=int, default=20,
                  help='number of times to repeat the dataset per training epoch')
data.add_argument('--augment', nargs='*', default=[DataAugmentationType.FlipTurn],
                  type=DataAugmentationType, choices=list(DataAugmentationType),
                  help='Data augmentation')
data.add_argument('--patch-size-train', type=int, default=96,
                  help='image patch size for training (HR)')
data.add_argument('--patch-size-val', type=int, default=384,
                  help='image patch size for validation (HR)')
data.add_argument('--preload-dataset', action='store_true',
                  help='load the whole dataset in memory')


# Hardware specification
hw.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
hw.add_argument('--gpu', type=int, help='specify GPU id to use')
hw.add_argument('--tune-backend', action='store_true',
                help='allow performance tuning of the backend (cudnn.benchmark)')
hw.add_argument('--datatype', type=DataType, default=DataType.FP32,
                choices=list(DataType),
                help='specify floating-point format')
hw.add_argument('--workers', type=int, default=0,
                help='number of workers for data loaders')

args = parser.parse_args()
if args.workers != 0 and args.preload_dataset:
    raise ValueError("Dataset preloading is incompatible with multiprocessing. "
                     "--worker argument cannot be given with --preload-dataset")
