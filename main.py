from datasets import *
from transforms import *
from train.enums import *
from train.options import args
import models

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from tqdm import tqdm
import skimage.metrics
import pdb


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0.0

    def update(self, val):
        self.sum += val
        self.count += 1

    def get(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def train(model, optimizer, loss_fn, loader, device):
    with torch.enable_grad():
        model.train()
        t = tqdm(loader)
        t.set_description("Train")
        l1_avg = AverageMeter()
        l2_avg = AverageMeter()
        for hr, lr in t:
            hr, lr = hr.to(device), lr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = loss_fn(sr, hr)
            loss.backward()
            optimizer.step()
            l1_loss = torch.nn.functional.l1_loss(sr, hr).item()
            l2_loss = torch.sqrt(torch.nn.functional.mse_loss(sr, hr)).item()
            l1_avg.update(l1_loss)
            l2_avg.update(l2_loss)
            t.set_postfix(L1=f'{l1_avg.get():.4f}', L2=f'{l2_avg.get():.4f}')


def test(model, loader, device):
    with torch.no_grad():
        model.eval()
        t = tqdm(loader)
        t.set_description("Test")
        psnr_avg = AverageMeter()
        ssim_avg = AverageMeter()
        for hr, lr in t:
            hr, lr = hr.to(device), lr.to(device)
            sr = model(lr)
            for i in range(sr.shape[0]):
                img_hr = np.array(F.to_pil_image(hr[i].cpu()))
                img_sr = np.array(F.to_pil_image(sr[i].cpu()))
                psnr = skimage.metrics.peak_signal_noise_ratio(img_hr, img_sr)
                ssim = skimage.metrics.structural_similarity(img_hr, img_sr, gaussian_weights=True, multichannel=True)
                psnr_avg.update(psnr)
                ssim_avg.update(ssim)
                t.set_postfix(PSNR=f'{psnr_avg.get():.2f}', SSIM=f'{ssim_avg.get():.4f}')


def load_checkpoint(path, model):
    if path is None:
        return
    ckp = torch.load(path)
    model.load_state_dict(ckp, strict=False)


def save_checkpoint(path, model):
    if path is None:
        return
    torch.save(model.state_dict(), path)


def name_to_dataset(name, split, transform):
    kwargs = {
        'root' : args.dataset_root,
        'scale' : args.scale,
        'split' : split,
        'transform' : transform,
        'download' : args.download_dataset,
        'predecode' : True
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
    transform_train = Compose([
        RandomCrop(args.patch_size_train),
        RandomFlipTurn(),
        ColorJitter(brightness=0.2, contrast=0.05, saturation=0.05),
        ToTensor()
        ])
    transform_val = Compose([
        # Full images are too big: only validate on a centered patch
        CenterCrop(args.patch_size_val),
        ToTensor()
        ])
    dataset_train = names_to_dataset(args.dataset_train, 'train',
        transform=transform_train)
    dataset_val = names_to_dataset(args.dataset_val, 'val',
        transform=transform_val)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=not args.cpu)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=not args.cpu)
    return loader_train, loader_val


def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=args.lr)


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_decay_steps,
        gamma=1.0/args.lr_decay_rate)


def get_loss():
    if args.loss == LossType.L1:
        return nn.L1Loss()
    if args.loss == LossType.SmoothL1:
        return nn.SmoothL1Loss(beta=0.1)
    if args.loss == LossType.L2:
        return nn.MseLoss()
    raise ValueError("Unknown loss")


def get_model():
    if args.arch is None:
        raise ValueError("No model is specified")
    if args.arch not in models.__dict__:
        raise ValueError(f"Unknown model {args.arch}")
    model = models.__dict__[args.arch](scale=args.scale, pretrained=args.download_pretrained)
    return model


def get_device():
    if args.cpu:
        return 'cpu'
    else:
        return 'cuda'


loader_train, loader_val = get_datasets()
device = get_device()
model = get_model().to(device)
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer)
loss_fn = get_loss()
load_checkpoint(args.load_checkpoint, model)

if args.evaluate:
    test(model, loader_val, device)
else:
    for epoch in range(args.epochs):
        train(model, optimizer, loss_fn, loader_train, device)
        if (epoch+1) % args.test_every == 0:
            test(model, loader_val, device)
            save_checkpoint(args.save_checkpoint, model)
        scheduler.step()

