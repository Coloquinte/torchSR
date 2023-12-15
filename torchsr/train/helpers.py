import piq
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

import torchsr.models as models

from torchsr.datasets import *
from torchsr.transforms import *
from .enums import *
from .options import args


def _name_to_dataset(name, split, transform):
    kwargs = {
        "root": args.dataset_root,
        "scale": args.scale,
        "split": split,
        "transform": transform,
        "download": args.download_dataset,
        "predecode": not args.preload_dataset,
        "preload": args.preload_dataset,
    }
    if name == DatasetType.Div2KBicubic:
        return Div2K(**kwargs, track="bicubic")
    if name == DatasetType.Div2KUnknown:
        return Div2K(**kwargs, track="unknown")
    if name == DatasetType.Set5:
        return Set5(**kwargs)
    if name == DatasetType.Set14:
        return Set14(**kwargs)
    if name == DatasetType.B100:
        return B100(**kwargs)
    if name == DatasetType.Urban100:
        return Urban100(**kwargs)
    raise ValueError("Unknown dataset")


def _names_to_dataset(names, split, transform):
    datasets = []
    for d in names:
        datasets.append(_name_to_dataset(d, split, transform))
    if len(datasets) == 0:
        return None
    return torch.utils.data.ConcatDataset(datasets)


def _get_transform_train():
    transforms = []
    transforms.append(
        RandomCrop(args.patch_size_train, scales=[1, args.scale], margin=0.5)
    )
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
        transforms.append(
            ColorJitter(
                brightness=0.2 if brightness else 0,
                contrast=0.1 if contrast else 0,
                saturation=0.1 if saturation else 0,
            )
        )
    return Compose(transforms)


def _get_transform_val():
    transforms = []
    if not args.validation_only:
        # Full images are too big: only validate on a centered patch
        transforms.append(
            CenterCrop(args.patch_size_val, allow_smaller=True, scales=[1, args.scale])
        )
    else:
        transforms.append(AdjustToScale(scales=[1, args.scale]))
    transforms.append(ToTensor())
    return Compose(transforms)


def get_datasets():
    if args.images is not None:
        return None, None
    dataset_val = _names_to_dataset(
        args.dataset_val, "val", transform=_get_transform_val()
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=not args.cpu,
    )
    if not args.validation_only:
        dataset_train = _names_to_dataset(
            args.dataset_train, "train", transform=_get_transform_train()
        )
        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=not args.cpu,
        )
    else:
        loader_train = None
    return loader_train, loader_val


def get_optimizer(model):
    if args.validation_only:
        return None

    kwargs = {}
    kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        kwargs["weight_decay"] = args.weight_decay

    if args.optimizer in [
        OptimizerType.ADAM,
        OptimizerType.ADAMW,
        OptimizerType.ADAMAX,
    ]:
        if args.momentum is not None:
            raise ValueError("No momentum for Adam-like optimizers")
        if args.adam_betas is not None:
            kwargs["betas"] = args.adam_betas
        if args.optimizer is OptimizerType.ADAM:
            return torch.optim.Adam(model.parameters(), **kwargs)
        if args.optimizer is OptimizerType.ADAMW:
            return torch.optim.AdamW(model.parameters(), **kwargs)
        if args.optimizer is OptimizerType.ADAMAX:
            return torch.optim.Adamax(model.parameters(), **kwargs)
    elif args.optimizer in [OptimizerType.SGD, OptimizerType.NESTEROV]:
        if args.momentum is not None:
            kwargs["momentum"] = args.momentum
        kwargs["nesterov"] = args.optimizer is OptimizerType.NESTEROV
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif args.optimizer is OptimizerType.RMSPROP:
        if args.momentum is not None:
            kwargs["momentum"] = args.momentum
        if args.rmsprop_alpha is not None:
            kwargs["alpha"] = args.rmsprop_alpha
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    assert False


def get_scheduler(optimizer):
    if args.validation_only:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_decay_steps, gamma=1.0 / args.lr_decay_rate
    )


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


def load_pretrained(model):
    if args.load_pretrained is None:
        return
    ckp = torch.load(args.load_pretrained)
    state = model.state_dict()
    for name, param in ckp.items():
        if name in state:
            try:
                state[name].copy_(param)
            except Exception as e:
                if "tail" not in name and "upsampler" not in name:
                    raise e
        else:
            if "tail" not in name and "upsampler" not in name:
                raise KeyError(f'Unexpected key "{name}" in state_dict')


def get_model():
    if args.arch not in models.__dict__:
        raise ValueError(f"Unknown model {args.arch}")
    model = models.__dict__[args.arch](
        scale=args.scale, pretrained=args.download_pretrained
    )

    if args.freeze_backbone:
        if (
            args.download_pretrained is None
            and args.load_checkpoint is None
            and args.load_pretrained is None
        ):
            raise ValueError("A pretrained model is required to freeze the backbone")
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "upsampler"):
            for p in model.upsampler.parameters():
                p.requires_grad = True
        elif hasattr(model, "tail"):
            for p in model.tail.parameters():
                p.requires_grad = True
        else:
            raise ValueError("The model has no known upsampling module to unfreeze")

    if args.self_ensemble:
        model = models.utils.SelfEnsembleModel(model)

    if args.chop_size is not None:
        model = models.utils.ChoppedModel(
            model, args.scale, args.chop_size, args.chop_overlap
        )

    if args.zero_pad:
        model = models.utils.ZeroPaddedModel(model, args.zero_pad)
    elif args.replication_pad:
        model = models.utils.ReplicationPaddedModel(model, args.replication_pad)
    elif args.reflection_pad:
        model = models.utils.ReflectionPaddedModel(model, args.reflection_pad)

    load_pretrained(model)

    if args.weight_norm:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                m = nn.utils.weight_norm(m)

    return model


def get_device():
    if args.tune_backend:
        torch.backends.cudnn.benchmark = True
    if args.cpu:
        return "cpu"
    elif args.gpu is not None:
        return "cuda:{}".format(args.gpu)
    else:
        return "cuda"


def get_dtype():
    if args.datatype is DataType.FP16:
        return torch.float16
    elif args.datatype is DataType.BFLOAT:
        return torch.bfloat16
    else:
        return torch.float32


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


def report_model(model):
    n_parameters = 0
    for p in model.parameters():
        n_parameters += p.nelement()
    print(f"Model {args.arch} with {n_parameters} parameters")


def to_tensor(img):
    t = F.to_tensor(img)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t


def to_image(t):
    """Workaround a bug in torchvision
    The conversion of a tensor to a PIL image causes overflows, which result in huge errors
    """
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.mul(255).round().div(255).clamp(0, 1)
    return F.to_pil_image(t.cpu())


def to_luminance(t):
    coeffs = (
        torch.tensor([65.738, 129.057, 25.064]).reshape(1, 3, 1, 1).to(t.device) / 256
    )
    return t.mul(coeffs).sum(dim=1, keepdim=True)


def to_YCbCr(t):
    weights = (
        torch.tensor(
            [
                [65.738, 129.057, 25.064],
                [-37.945, -74.494, 112.439],
                [112.439, -94.154, -18.285],
            ]
        )
        .reshape(3, 3, 1, 1)
        .to(t.device)
        / 256
    )
    biases = torch.tensor([0.0, 0.5, 0.5]).to(t.device)
    return nn.functional.conv2d(t, weights, biases)
