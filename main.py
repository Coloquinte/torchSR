from datasets import *
from transforms import *
from train.enums import *
from train.options import args
import models

import numpy as np
import os
import piq
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from tqdm import tqdm


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


def to_image(t):
    """Workaround a bug in torchvision
    The conversion of a tensor to a PIL image causes overflows, which result in huge errors"""
    t = torch.round(255 * t) / 255
    t = torch.clamp(t, 0, 1)
    return F.to_pil_image(t.cpu())


def report_model(model, name):
    n_parameters = 0
    for p in model.parameters():
        n_parameters += p.nelement()
    print(f"Training model {name} with {n_parameters} parameters")


class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, loader_train, loader_val, device, dtype):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.device = device
        self.dtype = dtype
        self.best_psnr = None
        self.best_epoch = None

    def train_iter(self, epoch):
        with torch.enable_grad():
            self.model.train()
            t = tqdm(self.loader_train)
            t.set_description(f"Epoch {epoch} train ")
            loss_avg = AverageMeter()
            l1_avg = AverageMeter()
            l2_avg = AverageMeter()
            for hr, lr in t:
                hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                self.optimizer.zero_grad()
                sr = self.model(lr)
                loss = self.loss_fn(sr, hr)
                loss.backward()
                if args.gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                optimizer.step()
                l1_loss = torch.nn.functional.l1_loss(sr, hr).item()
                l2_loss = torch.sqrt(torch.nn.functional.mse_loss(sr, hr)).item()
                l1_avg.update(l1_loss)
                l2_avg.update(l2_loss)
                args_dic = {
                    'L1' : f'{l1_avg.get():.4f}',
                    'L2' : f'{l2_avg.get():.4f}'
                }
                if not isinstance(self.loss_fn, (nn.L1Loss, nn.MSELoss)):
                    loss_avg.update(loss.item())
                    args_dic['Loss'] = f'{loss_avg.get():.4f}'
                t.set_postfix(**args_dic)

    def val_iter(self, epoch=None):
        with torch.no_grad():
            self.model.eval()
            t = tqdm(self.loader_val)
            if epoch is None:
                t.set_description("Validation")
            else:
                t.set_description(f"Epoch {epoch} val   ")
            psnr_avg = AverageMeter()
            ssim_avg = AverageMeter()
            for hr, lr in t:
                hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                sr = self.model(lr).clamp(0, 1)
                for i in range(sr.shape[0]):
                    psnr = piq.psnr(hr[i], sr[i])
                    ssim = piq.ssim(hr[i], sr[i])
                    psnr_avg.update(psnr)
                    ssim_avg.update(ssim)
                    t.set_postfix(PSNR=f'{psnr_avg.get():.2f}', SSIM=f'{ssim_avg.get():.4f}')
            return psnr_avg.get(), ssim_avg.get()

    def evaluate(self):
        self.val_iter()

    def train(self):
        t = tqdm(range(1, args.epochs+1))
        t.set_description("Epochs")
        for epoch in t:
            self.train_iter(epoch)
            if epoch % args.test_every == 0:
                psnr, ssim = self.val_iter(epoch)
                save_checkpoint(args.save_checkpoint, model)
                if self.best_psnr is None or psnr > self.best_psnr:
                    self.best_psnr = psnr
                    self.best_epoch = epoch
                    save_checkpoint(args.save_checkpoint, model, best=True)
                    t.set_postfix(best=epoch, PSNR=f'{psnr:.2f}', SSIM=f'{ssim:.4f}')
            scheduler.step()


def load_checkpoint(path, model):
    if path is None:
        return
    ckp = torch.load(path)
    model.load_state_dict(ckp, strict=False)


def save_checkpoint(path, model, best=False):
    if path is None:
        return
    if best:
        base, ext = os.path.splitext(path)
        path = base + "_best" + ext
    torch.save(model.state_dict(), path)


def name_to_dataset(name, split, transform):
    kwargs = {
        'root' : args.dataset_root,
        'scale' : args.scale,
        'split' : split,
        'transform' : transform,
        'download' : args.download_dataset,
        'predecode' : not args.preload_dataset,
        'preload' : args.preload_dataset,
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
        RandomCrop(args.patch_size_train, scales=[1,]+args.scale),
        RandomFlipTurn(),
        ColorJitter(brightness=0.2, contrast=0.05, saturation=0.05),
        ToTensor()
        ])
    transform_val = Compose([
        # Full images are too big: only validate on a centered patch
        CenterCrop(args.patch_size_val, allow_smaller=True, scales=[1,]+args.scale),
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
    if args.evaluate:
        return None
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.adam_betas,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
        )


def get_scheduler(optimizer):
    if args.evaluate:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_decay_steps,
        gamma=1.0/args.lr_decay_rate)


class PIQLoss(nn.Module):
    def __init__(self, loss, l1_epsilon=0.01):
        super(PIQLoss, self).__init__()
        self.l1_epsilon = l1_epsilon
        self.loss = loss

    def forward(self, input, target):
        # Clamp the values to the acceptable range for PIQ
        input_c = input.clamp(0, 1)
        target_c = target.clamp(0, 1)
        l1_loss = self.l1_epsilon * (input_c - input).abs().mean()
        return self.loss(input_c, target_c) + l1_loss


def get_loss():
    if args.loss == LossType.L1:
        return nn.L1Loss()
    if args.loss == LossType.SmoothL1:
        return nn.SmoothL1Loss(beta=0.01)
    if args.loss == LossType.L2:
        return nn.MSELoss()
    if args.loss == LossType.SSIM:
        return PIQLoss(piq.SSIMLoss())
    if args.loss == LossType.VIF:
        return PIQLoss(piq.VIFLoss())
    if args.loss == LossType.LPIPS:
        return PIQLoss(piq.LPIPS())
    if args.loss == LossType.DISTS:
        return PIQLoss(piq.DISTS())
    raise ValueError("Unknown loss")


def get_model():
    if args.arch is None:
        raise ValueError("No model is specified")
    if args.arch not in models.__dict__:
        raise ValueError(f"Unknown model {args.arch}")
    if len(args.scale) != 1:
        raise ValueError("Multiscale superresolution is not supported")
    model = models.__dict__[args.arch](scale=args.scale[0], pretrained=args.download_pretrained)
    return model


def get_device():
    if args.tune_backend:
        torch.backends.cudnn.benchmark = True
    if args.cpu:
        return 'cpu'
    elif args.gpu is not None:
        return 'cuda:{}'.format(args.gpu)
    else:
        return 'cuda'


def get_dtype():
    if args.datatype is DataType.FP16:
        return torch.float16
    elif args.datatype is DataType.BFLOAT:
        return torch.bfloat16
    else:
        return torch.float32


loader_train, loader_val = get_datasets()
device = get_device()
dtype = get_dtype()
model = get_model().to(dtype).to(device)
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer)
loss_fn = get_loss()
load_checkpoint(args.load_checkpoint, model)

trainer = Trainer(model, optimizer, scheduler, loss_fn, loader_train, loader_val, device, dtype)

if args.evaluate:
    trainer.evaluate()
else:
    report_model(model, args.arch)
    trainer.train()

