import copy
import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import piq
from tqdm import tqdm
from PIL import Image

from .enums import *

from .helpers import AverageMeter
from .helpers import to_image
from .helpers import to_tensor
from .helpers import to_luminance
from .helpers import to_YCbCr
from .helpers import get_datasets
from .helpers import get_model
from .helpers import get_optimizer
from .helpers import get_scheduler
from .helpers import get_loss
from .helpers import get_device
from .helpers import get_dtype

from .options import args


class Trainer:
    def __init__(self):
        self.epoch = 0
        self.best_psnr = None
        self.best_ssim = None
        self.best_loss = None
        self.best_epoch = None

        self.setup_device()
        self.setup_datasets()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_loss()
        self.load_checkpoint()
        self.setup_tensorboard()

    def setup_device(self):
        self.device = get_device()
        self.dtype = get_dtype()

    def setup_datasets(self):
        self.loader_train, self.loader_val = get_datasets()

    def setup_model(self):
        self.model = get_model().to(self.device).to(self.dtype)

    def setup_optimizer(self):
        self.optimizer = get_optimizer(self.model)

    def setup_scheduler(self):
        self.scheduler = get_scheduler(self.optimizer)

    def setup_loss(self):
        self.loss_fn = get_loss()

    def setup_tensorboard(self):
        self.writer = None
        if not args.validation_only:
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
            loss_avg = AverageMeter(0.05)
            l1_avg = AverageMeter(0.05)
            l2_avg = AverageMeter(0.05)
            for i in range(args.dataset_repeat):
                for hr, lr in self.loader_train:
                    hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(
                        self.device
                    )
                    self.optimizer.zero_grad()
                    sr = self.model(lr)
                    sr = self.process_for_eval(sr)
                    hr = self.process_for_eval(hr)
                    loss = self.loss_fn(sr, hr)
                    loss.backward()
                    if args.gradient_clipping is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.gradient_clipping
                        )
                    self.optimizer.step()
                    l1_loss = nn.functional.l1_loss(sr, hr).item()
                    l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                    l1_avg.update(l1_loss)
                    l2_avg.update(l2_loss)
                    args_dic = {
                        "L1": f"{l1_avg.get():.4f}",
                        "L2": f"{l2_avg.get():.4f}",
                    }
                    if args.loss not in [LossType.L1, LossType.L2]:
                        loss_avg.update(loss.item())
                        args_dic[args.loss.name] = f"{loss_avg.get():.4f}"
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
            loss_avg = AverageMeter()
            for hr, lr in t:
                hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(
                    self.device
                )
                sr = self.model(lr).clamp(0, 1)
                if final:
                    # Round to pixel values
                    sr = sr.mul(255).round().div(255)
                sr = self.process_for_eval(sr)
                hr = self.process_for_eval(hr)
                loss = self.loss_fn(sr, hr)
                l1_loss = nn.functional.l1_loss(sr, hr).item()
                l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                psnr = piq.psnr(hr, sr)
                ssim = piq.ssim(hr, sr)
                loss_avg.update(loss.item())
                l1_avg.update(l1_loss)
                l2_avg.update(l2_loss)
                psnr_avg.update(psnr)
                ssim_avg.update(ssim)
                args_dic = {
                    "PSNR": f"{psnr_avg.get():.4f}",
                    "SSIM": f"{ssim_avg.get():.4f}",
                }
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    args_dic[args.loss.name] = f"{loss_avg.get():.4f}"
                t.set_postfix(**args_dic)
            if self.writer is not None:
                self.writer.add_scalar("PSNR", psnr_avg.get(), self.epoch)
                self.writer.add_scalar("SSIM", ssim_avg.get(), self.epoch)
                self.writer.add_scalar("L1", l1_avg.get(), self.epoch)
                self.writer.add_scalar("L2", l2_avg.get(), self.epoch)
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    self.writer.add_scalar(args.loss.name, loss_avg.get(), self.epoch)
            return loss_avg.get(), psnr_avg.get(), ssim_avg.get()

    def validation(self):
        loss, psnr, ssim = self.val_iter()
        if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
            print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, loss: {loss:.4f}")
        else:
            print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    def run_model(self):
        scale = args.scale
        with torch.no_grad():
            self.model.eval()
            input_images = []
            for f in args.images:
                if os.path.isdir(f):
                    for g in os.listdir(f):
                        n = os.path.join(f, g)
                        if os.path.isfile(n):
                            input_images.append(n)
                else:
                    input_images.append(f)
            if args.destination is None:
                raise ValueError("You should specify a destination directory")
            os.makedirs(args.destination, exist_ok=True)
            t = tqdm(input_images)
            t.set_description("Run")
            for filename in t:
                try:
                    img = Image.open(filename)
                    img.load()
                except:
                    print(f"Could not open {filename}")
                    continue
                img = to_tensor(img).to(self.device)
                sr_img = self.model(img)
                sr_img = to_image(sr_img)
                destname = (
                    os.path.splitext(os.path.basename(filename))[0] + f"_x{scale}.png"
                )
                sr_img.save(os.path.join(args.destination, destname))

    def train(self):
        t = tqdm(total=args.epochs, initial=self.epoch)
        t.set_description("Epochs")
        if self.best_epoch is not None:
            args_dic = {"best": self.best_epoch}
            if self.best_psnr is not None:
                args_dic["PSNR"] = f"{self.best_psnr:.2f}"
            if self.best_ssim is not None:
                args_dic["SSIM"] = f"{self.best_ssim:.2f}"
            if self.best_loss is not None:
                args_dic["loss"] = f"{self.best_loss:.2f}"
            t.set_postfix(**args_dic)
        while self.epoch < args.epochs:
            self.epoch += 1
            self.train_iter()
            loss, psnr, ssim = self.val_iter(final=False)
            is_best = self.best_loss is None or loss < self.best_loss
            if is_best:
                self.best_loss = loss
                self.best_psnr = psnr
                self.best_ssim = ssim
                self.best_epoch = self.epoch
                t.set_postfix(
                    best=self.epoch,
                    PSNR=f"{psnr:.2f}",
                    SSIM=f"{ssim:.4f}",
                    loss=f"{loss:.4f}",
                )
            self.save_checkpoint(best=is_best)
            t.update(1)
            self.scheduler.step()

    def get_model_state_dict(self):
        # Ensures that the state_dict is on the CPU and reverse model transformations
        self.model.to("cpu")
        model = copy.deepcopy(self.model)
        self.model.to(self.device)
        if args.weight_norm:
            for m in model.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    m = nn.utils.remove_weight_norm(m)
        return model.state_dict()

    def load_checkpoint(self):
        if args.load_checkpoint is None:
            return
        ckp = torch.load(args.load_checkpoint)
        self.model.load_state_dict(ckp["state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(ckp["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckp["scheduler"])
        self.epoch = ckp["epoch"]
        if "best_epoch" in ckp:
            self.best_epoch = ckp["best_epoch"]
        if "best_psnr" in ckp:
            self.best_psnr = ckp["best_psnr"]
        if "best_ssim" in ckp:
            self.best_ssim = ckp["best_ssim"]
        if "best_loss" in ckp:
            self.best_loss = ckp["best_loss"]

    def save_checkpoint(self, best=False):
        if args.save_checkpoint is None:
            return
        path = args.save_checkpoint
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_epoch": self.best_epoch,
            "best_psnr": self.best_psnr,
            "best_ssim": self.best_ssim,
            "best_loss": self.best_loss,
        }
        torch.save(state, path)
        base, ext = os.path.splitext(path)
        if args.save_every is not None and self.epoch % args.save_every == 0:
            torch.save(state, base + f"_e{self.epoch}" + ext)
        if best:
            torch.save(state, base + "_best" + ext)
            torch.save(self.get_model_state_dict(), base + "_model" + ext)

    def process_for_eval(self, img):
        if args.shave_border != 0:
            shave = args.shave_border
            img = img[..., shave:-shave, shave:-shave]
        if args.eval_luminance:
            img = to_luminance(img)
        elif args.scale_chroma is not None:
            img = to_YCbCr(img)
            chroma_scaling = torch.tensor([1.0, args.scale_chroma, args.scale_chroma])
            img = img * chroma_scaling.reshape(1, 3, 1, 1).to(img.device)
        return img
