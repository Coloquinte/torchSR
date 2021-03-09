from datasets import *
from transforms import *
from train.enums import *
from train.options import args

import torch

def train(model, optimizer, dataset):
    pass

def test(model, dataset):
    pass

def name_to_dataset(name, split):
    if name == DatasetType.Div2K:
        return Div2K(root=args.dataset_root, scale=args.scale, split=split, download=args.download_dataset, track='bicubic')
    if name == DatasetType.Div2KUnknown:
        return Div2K(root=args.dataset_root, scale=args.scale, split=split, download=args.download_dataset, track='unknown')
    if name == DatasetType.Set5:
        return Set5(root=args.dataset_root, scale=args.scale, split=split, download=args.download_dataset)
    if name == DatasetType.Set14:
        return Set14(root=args.dataset_root, scale=args.scale, split=split, download=args.download_dataset)
    if name == DatasetType.B100:
        return B100(root=args.dataset_root, scale=args.scale, split=split, download=args.download_dataset)
    if name == DatasetType.Urban100:
        return Urban100(root=args.dataset_root, scale=args.scale, split=split, download=args.download_dataset)
    raise ValueError("Unknown dataset")

def names_to_dataset(names, split):
    datasets = []
    for d in names:
        datasets.append(name_to_dataset(d, split))
    if len(datasets) == 0:
        return None
    return torch.utils.data.ConcatDataset(datasets)

def get_datasets():
    dataset_train = names_to_dataset(args.dataset_train, 'train')
    dataset_val = names_to_dataset(args.dataset_val, 'val')
    dataset_test = names_to_dataset(args.dataset_test, 'test')
    return dataset_train, dataset_val, dataset_test

dataset_train, dataset_train, dataset_test = get_datasets()
dataset_train[0][1].show()
