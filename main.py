import os

import numpy as np
import piq
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from tqdm import tqdm

import models
from datasets import *
from train.enums import *
from train.options import args
from transforms import *


class AverageMeter:
    def __init__(self, smoothing=0.0):
        self.count = 0
        self.avg = 0.0
        self.smoothing = smoothing

    def update(self, val, count=1):
        self.count += count
        alpha = max(count / self.count, self.smoothing)
        self.avg = self.avg * (1.0 - alpha) + val * alpha

    def get(self):
        if self.count == 0:
            return 0.0
        return self.avg


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
        self.epoch = 0
        self.best_psnr = None
        self.best_ssim = None
        self.best_epoch = None
        self.load_checkpoint()
        self.load_pretrained()
        self.writer = None
        if not args.evaluate:
            try:
                # Only if tensorboard is present
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(args.log_dir, purge_step=self.epoch)
            except ImportError:
                if args.log_dir is not None:
                    raise ImportError("tensorboard is required to use --log-dir")

    def train_iter(self):
        with torch.enable_grad():
            self.model.train()
            t = tqdm(range(len(self.loader_train) * args.dataset_repeat))
            t.set_description(f"Epoch {self.epoch} train ")
            loss_avg = AverageMeter()
            l1_avg = AverageMeter(0.05)
            l2_avg = AverageMeter(0.05)
            for i in range(args.dataset_repeat):
                for hr, lr in self.loader_train:
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
                        'L1': f'{l1_avg.get():.4f}',
                        'L2': f'{l2_avg.get():.4f}'
                    }
                    if not isinstance(self.loss_fn, (nn.L1Loss, nn.MSELoss)):
                        loss_avg.update(loss.item())
                        args_dic['Loss'] = f'{loss_avg.get():.4f}'
                    t.update()
                    t.set_postfix(**args_dic)

    def val_iter(self, final=True):
        with torch.no_grad():
            self.model.eval()
            t = tqdm(self.loader_val)
            if final:
                t.set_description("Validation")
            else:
                t.set_description(f"Epoch {self.epoch} val   ")
            psnr_avg = AverageMeter()
            ssim_avg = AverageMeter()
            l1_avg = AverageMeter()
            l2_avg = AverageMeter()
            for hr, lr in t:
                hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                sr = self.model(lr).clamp(0, 1)
                l1_loss = torch.nn.functional.l1_loss(sr, hr).item()
                l2_loss = torch.sqrt(torch.nn.functional.mse_loss(sr, hr)).item()
                psnr = piq.psnr(hr, sr)
                ssim = piq.ssim(hr, sr)
                l1_avg.update(l1_loss)
                l2_avg.update(l2_loss)
                psnr_avg.update(psnr)
                ssim_avg.update(ssim)
                t.set_postfix(PSNR=f'{psnr_avg.get():.2f}', SSIM=f'{ssim_avg.get():.4f}')
            if self.writer is not None:
                self.writer.add_scalar('PSNR', psnr_avg.get(), self.epoch)
                self.writer.add_scalar('SSIM', ssim_avg.get(), self.epoch)
                self.writer.add_scalar('L1', l1_avg.get(), self.epoch)
                self.writer.add_scalar('L2', l2_avg.get(), self.epoch)
            return psnr_avg.get(), ssim_avg.get()

    def evaluate(self):
        self.val_iter()

    def train(self):
        t = tqdm(total=args.epochs, initial=self.epoch)
        t.set_description("Epochs")
        if self.best_epoch is not None and self.best_psnr is not None and self.best_ssim is not None:
            t.set_postfix(best=self.best_epoch,
                          PSNR=f'{self.best_psnr:.2f}',
                          SSIM=f'{self.best_ssim:.4f}')
        while self.epoch < args.epochs:
            self.epoch += 1
            self.train_iter()
            psnr, ssim = self.val_iter(final=False)
            is_best = self.best_psnr is None or psnr > self.best_psnr
            if is_best:
                self.best_psnr = psnr
                self.best_ssim = ssim
                self.best_epoch = self.epoch
                t.set_postfix(best=self.epoch, PSNR=f'{psnr:.2f}', SSIM=f'{ssim:.4f}')
            self.save_checkpoint(best=True)
            t.update(1)
            scheduler.step()

    def load_checkpoint(self):
        if args.load_checkpoint is None:
            return
        ckp = torch.load(args.load_checkpoint)
        self.model.load_state_dict(ckp['state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(ckp['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckp['scheduler'])
        self.epoch = ckp['epoch']
        if 'best_epoch' in ckp:
            self.best_epoch = ckp['best_epoch']
        if 'best_psnr' in ckp:
            self.best_psnr = ckp['best_psnr']
        if 'best_ssim' in ckp:
            self.best_ssim = ckp['best_ssim']

    def load_pretrained(self):
        if args.load_pretrained is None:
            return
        ckp = torch.load(args.load_pretrained)
        self.model.load_state_dict(ckp, strict=False)

    def save_checkpoint(self, best=False):
        if args.save_checkpoint is None:
            return
        path = args.save_checkpoint
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_epoch': self.best_epoch,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
        }
        torch.save(state, path)
        if best:
            base, ext = os.path.splitext(path)
            best_path = base + "_best" + ext
            torch.save(state, best_path)
            model_path = base + "_model" + ext
            torch.save(self.model.state_dict(), model_path)


def name_to_dataset(name, split, transform):
    kwargs = {
        'root': args.dataset_root,
        'scale': args.scale,
        'split': split,
        'transform': transform,
        'download': args.download_dataset,
        'predecode': not args.preload_dataset,
        'preload': args.preload_dataset,
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


def get_transform_train():
    transforms = []
    transforms.append(RandomCrop(args.patch_size_train, scales=[1, ]+args.scale, margin=0.5))
    if DataAugmentationType.FlipTurn in args.augment:
        transforms.append(RandomFlipTurn())
    else:
        if DataAugmentationType.VFlip in args.augment:
            transforms.append(RandomVerticalFlip())
        if DataAugmentationType.HFlip in args.augment:
            transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    brightness = DataAugmentationType.BrightnessJitter in args.augment
    contrast = DataAugmentationType.ContrastJitter in args.augment
    saturation = DataAugmentationType.SaturationJitter in args.augment
    if brightness or contrast or saturation:
        transforms.append(ColorJitter(
            brightness=0.2 if brightness else 0,
            contrast=0.1 if contrast else 0,
            saturation=0.1 if saturation else 0
        ))
    return Compose(transforms)


def get_transform_val():
    transforms = []
    # Full images are too big: only validate on a centered patch
    transforms.append(CenterCrop(args.patch_size_val, allow_smaller=True, scales=[1, ]+args.scale))
    transforms.append(ToTensor())
    return Compose(transforms)


def get_datasets():
    dataset_train = names_to_dataset(args.dataset_train, 'train',
                                     transform=get_transform_train())
    dataset_val = names_to_dataset(args.dataset_val, 'val',
                                   transform=get_transform_val())
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

    kwargs = {}
    kwargs['lr'] = args.lr
    if args.weight_decay is not None:
        kwargs['weight_decay'] = args.weight_decay

    if args.optimizer in [OptimizerType.ADAM,
                          OptimizerType.ADAMW,
                          OptimizerType.ADAMAX]:
        if args.momentum is not None:
            raise ValueError("No momentum for Adam-like optimizers")
        if args.adam_betas is not None:
            kwargs['betas'] = args.adam_betas
        if args.optimizer is OptimizerType.ADAM:
            return torch.optim.Adam(model.parameters(), **kwargs)
        if args.optimizer is OptimizerType.ADAMW:
            return torch.optim.AdamW(model.parameters(), **kwargs)
        if args.optimizer is OptimizerType.ADAMAX:
            return torch.optim.Adamax(model.parameters(), **kwargs)
    elif args.optimizer in [OptimizerType.SGD,
                            OptimizerType.NESTEROV]:
        if args.momentum is not None:
            kwargs['momentum'] = args.momentum
        kwargs['nesterov'] = args.optimizer is OptimizerType.NESTEROV
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif args.optimizer is OptimizerType.RMSPROP:
        if args.momentum is not None:
            kwargs['momentum'] = args.momentum
        if args.rmsprop_alpha is not None:
            kwargs['alpha'] = args.rmsprop_alpha
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    assert False


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

trainer = Trainer(model, optimizer, scheduler, loss_fn, loader_train, loader_val, device, dtype)

if args.evaluate:
    trainer.evaluate()
else:
    report_model(model, args.arch)
    trainer.train()
