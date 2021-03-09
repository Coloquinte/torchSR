from datasets import *
from transforms import *
from train.enums import *
from train.options import args

import torch

def train(model, optimizer, dataset):
    pass

def test(model, dataset):
    pass

def name_to_dataset(name, split, transform):
    kwargs = {
        'root' : args.dataset_root,
        'scale' : args.scale,
        'split' : split,
        'transform' : transform,
        'download' : args.download_dataset
        }
    if name == DatasetType.Div2KBicubic:
        return Div2K(**kwargs, track='bicubic')
    if name == DatasetType.Div2KUnknown:
        return Div2K(**kwargs, track='unknown')
    if name == DatasetType.Set5:
        return Set5(**kwargs)
    if name == DatasetType.Set14:
        return Set14(**kwargs)
    if name == DatasetType.B100:
        return B100(**kwargs)
    if name == DatasetType.Urban100:
        return Urban100(**kwargs)
    raise ValueError("Unknown dataset")

def names_to_dataset(names, split, transform):
    datasets = []
    for d in names:
        datasets.append(name_to_dataset(d, split, transform))
    if len(datasets) == 0:
        return None
    return torch.utils.data.ConcatDataset(datasets)

def get_datasets():
    dataset_train = names_to_dataset(args.dataset_train, 'train',
        transform=Compose([RandomCrop(args.patch_size)]))
    dataset_val = names_to_dataset(args.dataset_val, 'val',
        transform=None)
    return dataset_train, dataset_val

def get_model():
    pass

dataset_train, dataset_train, dataset_test = get_datasets()
dataset_train[0][1].show()

