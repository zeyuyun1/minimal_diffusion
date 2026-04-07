import argparse
import json
import os
import re

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid
# import sparselearning
# from sparselearning.core import Masking, CosineDecay, LinearDecay

from recurrent_diffusion_pkg.model import neural_sheet6, neural_sheet7, UNetSmall, UNet, UNetBig, ToySCDEQ

from torch.optim.lr_scheduler import LambdaLR
import wandb
import numpy as np
import math

def _to_wandb_image(chw: torch.Tensor):
    """Convert CHW [0,1] → something wandb.Image likes."""
    chw = chw.detach().cpu()
    C = chw.size(0)
    if C == 1:
        return wandb.Image(chw[0].numpy())              # H×W
    elif C in (3, 4):
        return wandb.Image(chw.permute(1, 2, 0).numpy())  # H×W×3/4
    else:
        # collapse funny channel counts (e.g., 16) to grayscale
        return wandb.Image(chw.mean(dim=0).numpy())


def _ensure_chw(img: torch.Tensor) -> torch.Tensor:
    """Accept HxW, CHW, or HWC and return CHW for downstream grid/logging."""
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    if img.dim() == 2:
        return img.unsqueeze(0)
    if img.dim() == 3 and img.shape[-1] in (1, 3, 4) and img.shape[0] not in (1, 3, 4):
        return img.permute(2, 0, 1).contiguous()
    return img

# Import from our new modules
from recurrent_diffusion_pkg.utils import warmup_fn, vis_patches
from recurrent_diffusion_pkg.loss import EDMLossNoCond, SimpleUniformNoiseLoss, EDMStyleXPredLoss, SimpleNoiseLoss, PyramidEDMLoss
from recurrent_diffusion_pkg.data import SubsetWithTransform, ImageDataModule, MNISTDataModule, STL10DataModule, ToyPacManDataModule


def add_noise(x, sigma, distribution: str = "uniform",
              P_mean: float = -1.2, P_std: float = 1.2):
    """
    x: (B,C,H,W)
    sigma:
      * if distribution == "uniform": tuple/list (lo, hi) or float
      * if distribution == "edm": tuple/list (noise_min, noise_max) or None
    P_mean/P_std: EDM log-normal parameters (must match training)
    """
    B = x.shape[0]
    device = x.device
    sigma_view_shape = [B] + [1] * (x.ndim - 1)

    if distribution == "edm":
        # σ ~ LogNormal(P_mean, P_std), then clamp to noise_range if given
        rnd = torch.randn(*sigma_view_shape, device=device)
        sigma_sample = (rnd * P_std + P_mean).exp()
        if isinstance(sigma, (tuple, list)) and len(sigma) == 2:
            lo, hi = sigma
            if lo is not None:
                sigma_sample = sigma_sample.clamp_min(lo)
            if hi is not None:
                sigma_sample = sigma_sample.clamp_max(hi)
        return x + torch.randn_like(x) * sigma_sample, sigma_sample.flatten()

    # default: uniform (your original behavior)
    if isinstance(sigma, (tuple, list)) and len(sigma) == 2:
        lo, hi = sigma
        sigma_sample = torch.rand(*sigma_view_shape, device=device) * (hi - lo) + lo
        return x + torch.randn_like(x) * sigma_sample, sigma_sample.flatten()
    else:
        sigma_sample = torch.full((B,), float(sigma), device=device, dtype=x.dtype)
        sigma_view = sigma_sample.view(*sigma_view_shape)
        return x + torch.randn_like(x) * sigma_view, sigma_sample


class EMACheckpointCallback(Callback):
    """Custom callback to save EMA model checkpoints."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def on_validation_end(self, trainer, pl_module):
        """Save EMA model checkpoint at the end of validation."""
        if trainer.is_global_zero:  # Only save on main process
            ema_checkpoint_path = os.path.join(self.save_dir, "ema_model.ckpt")
            torch.save(pl_module.ema_model.state_dict(), ema_checkpoint_path)
            print(f"Saved EMA model checkpoint to: {ema_checkpoint_path}")
    
    def on_train_end(self, trainer, pl_module):
        """Save final EMA model checkpoinz at the end of training."""
        if trainer.is_global_zero:  # Only save on main process
            ema_checkpoint_path = os.path.join(self.save_dir, "final_ema_model.ckpt")
            torch.save(pl_module.ema_model.state_dict(), ema_checkpoint_path)
            print(f"Saved final EMA model checkpoint to: {ema_checkpoint_path}")


class LitDenoiser(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_basis: list,
        eta_base: float,
        kernel_size: int,
        stride: int,
        lr: float = 1e-3,
        sigma=(0.1, 1.5),
        model_arch = "h_sparse",
        ema_halflife_kimg: float = 500.0,
        ema_rampup_ratio: float = 0.05,
        P_mean: float = -2.0,
        P_std: float = 0.5,
        edm_weighting: bool = False,
        jfb_no_grad_iters: tuple = (0, 6),
        jfb_with_grad_iters: tuple = (1, 3),
        jfb_reuse_solution: bool = False,
        learning_horizontal: bool = True,
        noise_embedding: bool = True,
        loss_type: str = "edm",
        bias: bool = True,
        relu_6: bool = False,
        T: float = 0.1,
        per_dim_threshold: bool = False,
        positive_threshold: bool = False,
        multiscale: bool = False,
        constraint_energy: str = "SC",
        intra: bool = True,
        k_inter: int = None,
        n_hid_layers: int = -1,
        tied_transpose: bool = True,
        film: bool = False,
        scalar_mul: bool = False,
        ff_scale: bool = False,
        control_groups: int = None,
        ignore_ff_control: bool = False,
        pyramid_loss: bool = False,
        pyramid_level_weights: list = None,
        img_size: int = 64,
        unet_variant: str = "small",
        unet_base_width: int = 32,
        unet_no_attention: bool = False,
        toy_dim: int = 128,
        toy_hidden_dim: int = 128,
        toy_learning_horizontal: bool = False,
        simple_control: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        if loss_type == "uniform":
            self.loss_obj = SimpleUniformNoiseLoss(noise_range=sigma)
        elif loss_type == "pyramid":
            self.loss_obj = PyramidEDMLoss(
                P_mean=P_mean,
                P_std=P_std,
                sigma_data=0.25,
                noise_range=sigma,
                edm_weighting=edm_weighting,
                level_weights=pyramid_level_weights,
            )
        else:
            self.loss_obj = EDMLossNoCond(noise_range=sigma,sigma_data=0.25,P_mean=P_mean,P_std=P_std,edm_weighting=edm_weighting)
        # self.loss_obj = SimpleUniformNoiseLoss(noise_range=sigma)
        # self.loss_obj = EDMLossNoCond(noise_range=sigma,sigma_data=0.15)
        # self.loss_obj = EDMLossNoCond(noise_range=sigma,sigma_data=0.15,P_mean=P_mean,P_std=P_std)
        # self.loss_obj = EDMStyleXPredLoss(noise_range=sigma,sigma_data=0.25)
        # self.loss_obj = SimpleNoiseLoss(noise_range=sigma,sigma_data=0.25,P_mean=-2,P_std=0.5,weighting='edm')
        model_cls_map = {
            # "neural_sheet2": neural_sheet2,
            # "neural_sheet3": neural_sheet3,
            # "neural_sheet4": neural_sheet4,
            # "neural_sheet5": neural_sheet5,
            "neural_sheet6": neural_sheet6,
            "neural_sheet7": neural_sheet7,
            "unet": None,
            "toy_sc_deq": None,
        }
        if model_arch not in model_cls_map:
            raise ValueError(
                f"Unsupported model_arch='{model_arch}'. "
                f"Supported: {list(model_cls_map.keys())}"
            )

        model_kwargs = dict(
            in_channels=in_channels,
            num_basis=num_basis,
            eta_base=eta_base,
            kernel_size=kernel_size,
            stride=stride,
            jfb_no_grad_iters=jfb_no_grad_iters,
            jfb_with_grad_iters=jfb_with_grad_iters,
            jfb_reuse_solution=jfb_reuse_solution,
            learning_horizontal=learning_horizontal,
            noise_embedding=noise_embedding,
            bias=bias,
            relu_6=relu_6,
            T=T,
            constraint_energy=constraint_energy,
            per_dim_threshold=per_dim_threshold,
            positive_threshold=positive_threshold,
            multiscale=multiscale,
            intra=intra,
            k_inter=k_inter,
            n_hid_layers=n_hid_layers,
            tied_transpose=tied_transpose,
            film=film,
            scalar_mul=scalar_mul,
            ff_scale=ff_scale,
            simple_control=simple_control,
        )
        if model_arch == "neural_sheet3":
            model_kwargs["control_groups"] = control_groups
            model_kwargs["ignore_ff_control"] = ignore_ff_control

        if model_arch == "unet":
            unet_factories = {
                "small": UNetSmall,
                "base": UNet,
                "big": UNetBig,
            }
            if unet_variant not in unet_factories:
                raise ValueError(
                    f"Unsupported unet_variant='{unet_variant}'. Supported: {list(unet_factories.keys())}"
                )
            unet_factory = unet_factories[unet_variant]
            self.model = unet_factory(
                image_size=img_size,
                in_channels=in_channels,
                out_channels=in_channels,
                base_width=unet_base_width,
                num_classes=None,
                use_attention=not unet_no_attention,
            )
            self.ema_model = unet_factory(
                image_size=img_size,
                in_channels=in_channels,
                out_channels=in_channels,
                base_width=unet_base_width,
                num_classes=None,
                use_attention=not unet_no_attention,
            )
        elif model_arch == "toy_sc_deq":
            self.model = ToySCDEQ(
                data_dim=toy_dim,
                hidden_dim=toy_hidden_dim,
                eta=eta_base,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                learning_horizontal=toy_learning_horizontal,
            )
            self.ema_model = ToySCDEQ(
                data_dim=toy_dim,
                hidden_dim=toy_hidden_dim,
                eta=eta_base,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                learning_horizontal=toy_learning_horizontal,
            )
        else:
            model_cls = model_cls_map[model_arch]
            self.model = model_cls(**model_kwargs)
            self.ema_model = model_cls(**model_kwargs)
        print(type(self.model))
        print("EMA",type(self.ema_model))
        # Copy initial parameters from main model to EMA model
        self.ema_model.load_state_dict(self.model.state_dict())
        # Freeze EMA model parameters (they will be updated via EMA, not gradients)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self._unused_param_debug_steps = 5

    def forward(self, x, noise_labels=None):
        if self.hparams.model_arch in ("unet", "toy_sc_deq"):
            return self.model(x, noise_labels)
        return self.model(x, noise_labels=noise_labels)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        if self.hparams.model_arch in ("neural_sheet7", "unet"):
            # sheet7/unet: plain denoising path
            noise_dist = "uniform" if self.hparams.loss_type == "uniform" else "edm"
            x_in, noise_labels = add_noise(
                x,
                self.hparams.sigma,
                distribution=noise_dist,
                P_mean=self.hparams.P_mean,
                P_std=self.hparams.P_std,
            )
            if self.hparams.model_arch == "neural_sheet7":
                x_hat = self.model(
                    x_in,
                    noise_labels=noise_labels,
                    measurement=None,
                    class_labels=labels,
                )
            else:
                x_hat = self.model(x_in, noise_labels)
            per_pix = (x_hat - x) ** 2
            if self.hparams.edm_weighting and noise_dist == "edm":
                sigma_data = 0.25
                sigma = noise_labels.view(-1, 1, 1, 1)
                w = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
                per_pix = per_pix * w
        else:
            if self.hparams.model_arch == "toy_sc_deq":
                per_pix = self.loss_obj(self.model, x, class_labels=None)
            else:
                per_pix = self.loss_obj(self.model, x, class_labels=labels)
        loss = per_pix.mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)        
        return loss


    def optimizer_step(self, *args, **kwargs):
        # perform the actual optimizer step first
        super().optimizer_step(*args, **kwargs)
        # then update EMA exactly once per optimizer step
        self.update_ema()

    def on_after_backward(self):
        # Track global grad L2 norm to diagnose instability/explosions.
        grad_sq_sum = None
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                g2 = torch.sum(p.grad.detach() * p.grad.detach())
                grad_sq_sum = g2 if grad_sq_sum is None else (grad_sq_sum + g2)
        if grad_sq_sum is not None:
            grad_norm = torch.sqrt(grad_sq_sum)
            self.log(
                "grad_norm",
                grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

        # Debug helper: list trainable params that didn't receive gradients in early steps.
        debug_all_ranks = os.environ.get("DEBUG_DDP_UNUSED", "0") == "1"
        if not debug_all_ranks and not self.trainer.is_global_zero:
            return
        if int(self.global_step) >= self._unused_param_debug_steps:
            return
        unused = []
        for name, p in self.named_parameters():
            if p.requires_grad and p.grad is None:
                unused.append(name)
        if unused:
            rank = int(os.environ.get("RANK", "0"))
            print(f"[rank {rank}] [unused params @ step {int(self.global_step)}] {len(unused)}", flush=True)
            for name in unused:
                print(f"  {name}", flush=True)
    
    @torch.no_grad()
    def update_ema(self):
        gb = self._global_batch_size()
        cur_nimg = self._cur_nimg()

        ema_halflife_nimg = int(self.hparams.ema_halflife_kimg * 1000)
        if self.hparams.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, max(cur_nimg, 1) * self.hparams.ema_rampup_ratio)

        beta = 0.5 ** (gb / max(ema_halflife_nimg, 1))
        for p_ema, p_net in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, beta))


    def validation_step(self, batch, batch_idx):
        x, _ = batch
        noise_dist = "uniform" if self.hparams.loss_type == "uniform" else "edm"
        x_in, noise_labels = add_noise(
            x,
            self.hparams.sigma,
            distribution=noise_dist,
            P_mean=self.hparams.P_mean,
            P_std=self.hparams.P_std,
        )
        measurement = None
        
        # Use EMA model for validation (better performance)
        with torch.no_grad():
            if self.hparams.model_arch == "neural_sheet7":
                x_hat = self.ema_model(
                    x_in,
                    noise_labels=noise_labels,
                    measurement=measurement,
                )
            elif self.hparams.model_arch == "unet":
                x_hat = self.ema_model(x_in, noise_labels)
            else:
                x_hat = self.ema_model(x_in, noise_labels=noise_labels)
    
        # compute batch signal and noise power
        sig_pow   = torch.sum(x ** 2)
        noise_pow = torch.sum((x - x_hat) ** 2)
    
        # compute SNR in dB and log
        snr_db = 10 * torch.log10(sig_pow / noise_pow)
        self.log("val_snr_db", snr_db, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:                     # only first batch each epoch
            self._log_images_rank0(x, x_in, x_hat, x_corrupt=None)   # will execute only on global rank 0
    
        return snr_db

    @rank_zero_only
    def _log_images_rank0(self, x, x_in, x_hat, x_corrupt=None):
        """
        x, x_in, x_hat, x_corrupt: (B, C, H, W), values can be in [-1,1] or
        arbitrary — we normalize in vis_patches.
        """
        if x.ndim != 4:
            return
        n_vis = min(4, x.shape[0])

        # 4 panels: clean | noisy | denoised | corrupt
        a = _ensure_chw(vis_patches(x[:n_vis].detach().cpu().flatten(2),     show=False, return_tensor=True))
        b = _ensure_chw(vis_patches(x_in[:n_vis].detach().cpu().flatten(2),  show=False, return_tensor=True))
        c = _ensure_chw(vis_patches(x_hat[:n_vis].detach().cpu().flatten(2), show=False, return_tensor=True))
        if x_corrupt is None:
            x_corrupt = x_in
        d = _ensure_chw(vis_patches(x_corrupt[:n_vis].detach().cpu().flatten(2), show=False, return_tensor=True))

        # stack as a batch and tile horizontally
        panel = make_grid(torch.stack([a, b, c, d], dim=0), nrow=4, padding=4)  # CHW
        # re-normalize just in case; safe even if already [0,1]
        panel = (panel - panel.min()) / (panel.max() - panel.min() + 1e-8)

        # Resolve your net for pulling weights
        net = getattr(self, "ema_model", None) or self.model
        root = getattr(net, "model", net)

        # ---- layer-1 filters ----
        # w2: [out_c, in_c, kh, kw]
        basis_img = None
        with torch.no_grad():
            try:
                w2 = root.levels[0].decoder.conv.weight.detach()
            except Exception:
                try:
                    w2 = root.encoder_nodes[0].decoder.conv.weight.detach()
                except Exception:
                    try:
                        w2 = root.encoder[0].decoder.conv.weight.detach()
                    except Exception:
                        w2 = None
            if w2 is not None:
                out_c, in_c, kh, kw = w2.shape
                patches = w2.view(out_c, in_c, kh * kw)   # (B=out_c, C=in_c, P)
                if in_c not in (1, 3):
                    patches = patches.mean(dim=1, keepdim=True)  # (B,1,P)
                basis_img = _ensure_chw(vis_patches(
                    patches, normalize=True, return_tensor=True,
                    ncol=int(math.ceil(math.sqrt(out_c)))
                ))

        logs = {"validation_panel": _to_wandb_image(panel)}
        if basis_img is not None:
            logs["filters/layer0"] = _to_wandb_image(basis_img)
        self.logger.experiment.log(logs, step=self.global_step)



    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=warmup_fn),
            'interval': 'step',
        }
        return [optimizer], [scheduler]

    def _world_size(self) -> int:
        # PL 2.x usually has .world_size; fall back to num_devices/1
        return int(getattr(self.trainer, "world_size",
                        getattr(self.trainer, "num_devices", 1)) or 1)

    def _accum(self) -> int:
        return int(getattr(self.trainer, "accumulate_grad_batches", 1))

    def _per_device_bs(self) -> int:
        # works with your DataModules
        return int(getattr(self.trainer.datamodule, "batch_size", 1))

    def _global_batch_size(self) -> int:
        return self._per_device_bs() * self._world_size() * self._accum()

    def _cur_nimg(self) -> int:
        # global_step increments per optimizer step in PL
        return int(self.global_step) * self._global_batch_size()
    


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Train a recurrent conv denoiser with PyTorch Lightning.")
    parser.add_argument("--project_name", type=str, default="main",
                        help="Name of the project")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Name of the experiment. Results will be saved under pretrained_model/<exp_name>.")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to a JSON config file. Overrides other args.")
    parser.add_argument("--config_dir", type=str, default="configs",
                        help="Directory to save generated config files.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for image datasets. If not set, dataset-specific defaults are used (celeba=~/celeba, berkeley=/home/zeyu/data/BSDS500, vh=...).")
    # General training args
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # Model hyperparameters
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_basis", type=lambda s: [int(item) for item in s.split(',')], default="64,64",
                        help="Comma-separated list for number of basis per layer, e.g. '64,64'.")
    parser.add_argument("--eta_base", type=float, default=0.25)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sigma", type=lambda s: tuple(map(float, s.split(','))), default="0.5,0.5",
                        help="Noise sigma as single float or range 'min,max'.")
    parser.add_argument(
        "--model_arch",
        type=str,
        choices=["neural_sheet2", "neural_sheet3", "neural_sheet5", "neural_sheet6", "neural_sheet7", "unet", "toy_sc_deq"],
        default="neural_sheet2",
    )
    parser.add_argument(
        "--unet_variant",
        type=str,
        choices=["small", "base", "big"],
        default="small",
        help="UNet size variant when --model_arch unet.",
    )
    parser.add_argument(
        "--unet_base_width",
        type=int,
        default=32,
        help="Base channel width for UNet variants.",
    )
    parser.add_argument(
        "--unet_no_attention",
        action="store_true",
        default=False,
        help="Disable attention blocks in UNet.",
    )
    parser.add_argument(
        "--toy_dim",
        type=int,
        default=128,
        help="Toy vector dimension for toy_pacman dataset and toy_sc_deq model.",
    )
    parser.add_argument(
        "--toy_num_samples",
        type=int,
        default=10000,
        help="Number of generated toy_pacman samples.",
    )
    parser.add_argument(
        "--toy_hidden_dim",
        type=int,
        default=128,
        help="Hidden dictionary size for toy_sc_deq.",
    )
    parser.add_argument(
        "--toy_learning_horizontal",
        action="store_true",
        default=False,
        help="Enable horizontal matrix M in toy_sc_deq (off keeps only dictionary encode/decode path).",
    )
    parser.add_argument("--dataset", type=str, choices=["mnist","celeba","vh","stl10","berkeley","toy_pacman"], default="celeba",
                        help="Dataset to train on: 'mnist' or 'celeba'.")
    parser.add_argument("--grayscale", action="store_true", default=False,
                        help="If set, convert images to grayscale (1 channel) in the dataloader.")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Target image size for resizing/cropping (square). Ignored if --no_resize.")
    parser.add_argument("--no_resize", action="store_true", default=False,
                        help="Disable resize/center-crop in the dataloader transforms.")
    parser.add_argument("--resize_size", type=int, default=512,
                        help="Resize size for the dataloader transforms.")
    parser.add_argument("--random_crop", action="store_true", default=False,
                        help="Use RandomCrop(img_size) instead of CenterCrop. If --no_resize is set, only crop.")
    # EDM noise distribution parameters
    parser.add_argument("--P_mean", type=float, default=-1.2,
                        help="Mean parameter for EDM log-normal noise distribution (default: -2.0)")
    parser.add_argument("--P_std", type=float, default=1.2,
                        help="Std parameter for EDM log-normal noise distribution (default: 0.5)")
    parser.add_argument("--edm_weighting", action="store_true", default=False,
                        help="Use EDM weighting for the loss function (default: False)")
    # EMA hyperparameters
    parser.add_argument("--ema_halflife_kimg", type=float, default=500.0,
                        help="EMA half-life in thousands of images (default: 500.0)")
    parser.add_argument("--ema_rampup_ratio", type=float, default=0.05,
                        help="EMA ramp-up ratio (default: 0.05)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint file to load for initialization (e.g., 'pretrained_model/main/00001_experiment/denoiser.ckpt')")
    # jfb_no_grad_iters
    parser.add_argument("--jfb_no_grad_iters", type=lambda s: tuple(map(int, s.split(','))), default="0,6",
                        help="Comma-separated list for number of no-grad iterations for each layer, e.g. '0,6'.")
    parser.add_argument("--jfb_with_grad_iters", type=lambda s: tuple(map(int, s.split(','))), default="1,3",
                        help="Comma-separated list for number of with-grad iterations for each layer, e.g. '1,3'.")
    parser.add_argument("--jfb_reuse_solution", action="store_true", default=False,
                        help="Use reuse solution for JFB (default: False)")
    parser.add_argument("--no_learning_horizontal", action="store_true", default=False,
                        help="No learning horizontal for gram (default: False)")
    parser.add_argument("--no_noise_embedding", action="store_true", default=False,
                        help="No noise embedding (default: False)")
    parser.add_argument("--loss_type", type=str, choices=["edm", "uniform", "pyramid"], default="edm",
                        help="Loss type: 'edm' (default) | 'uniform' | 'pyramid' (per-level Laplacian)")
    parser.add_argument("--pyramid_level_weights",
                        type=lambda s: [float(v) for v in s.split(',')],
                        default=None,
                        help="Comma-separated per-level loss weights for pyramid loss, "
                             "e.g. '0.5,1.0,2.0' to upweight coarse levels. "
                             "Defaults to equal weight 1.0 per level.")
    parser.add_argument("--no_bias", action="store_true", default=False,
                        help="No bias (default: False)")
    parser.add_argument("--relu_6", action="store_true", default=False,
                        help="Use ReLU6 (default: False)")
    parser.add_argument("--T", type=float, default=0.0,
                        help="Temperature for the model (default: 0.1)")
    parser.add_argument("--per_dim_threshold", action="store_true", default=False,
                        help="Per-dimension threshold (default: False)")
    parser.add_argument("--positive_threshold", action="store_true", default=False,
                        help="Positive threshold (default: False)")
    parser.add_argument("--multiscale", action="store_true", default=False,
                        help="Multiscale (default: False)")
    parser.add_argument("--constraint_energy", type=str, choices=["SC", "BM",], default="SC",
                        help="Constraint energy (default: SC)")
    parser.add_argument("--intra", action="store_true", default=False,
                        help="Intra (default: True)")
    parser.add_argument("--k_inter", type=int, default=None,
                        help="k for inter layer weight norm (default: None)")
    parser.add_argument("--n_hid_layers", type=int, default=1,
                        help="Number of hidden layers (default: 1)")
    parser.add_argument(
        "--no_tied_transpose",
        action="store_true",
        default=False,
        help="Disable tied transpose (default: enabled)"
    )
    parser.add_argument("--film", action="store_true", default=False,
                        help="Use FiLM (default: False)")
    parser.add_argument("--scalar_mul", action="store_true", default=False,
                        help="Use scalar multiplication (default: False)")
    parser.add_argument("--ff_scale", action="store_true", default=False,
                        help="Use FF scale (default: False)")
    parser.add_argument(
        "--control_groups",
        type=int,
        default=None,
        help="Grouped control for neural_sheet3. None keeps legacy behavior; 1 gives globally shared control.",
    )
    parser.add_argument(
        "--ignore_ff_control",
        action="store_true",
        default=False,
        help="For neural_sheet3: ignore level FF control gate noise_emb[0] in node updates.",
    )
    parser.add_argument("--simple_control", action="store_true", default=False,
                        help="Use simple control (default: False)")
    # sparselearning.core.add_sparse_args(parser)
    args = parser.parse_args()
    
    # Set up experiment directory with index to avoid collisions
    # Check if we're in a distributed environment using environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    project_dir = os.path.join("pretrained_model", args.project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Only rank 0 creates the directory and determines the index
    if local_rank == 0:
        # Find existing experiment directories and get the next index
        prev_run_dirs = []
        if os.path.isdir(project_dir):
            prev_run_dirs = [x for x in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        
        # Create experiment directory with index
        exp_dir_name = f"{cur_run_id:05d}_{args.exp_name}"
        base_dir = os.path.join(project_dir, exp_dir_name)
        os.makedirs(base_dir, exist_ok=True)
        
        # Write the directory name to a file for other processes to read
        with open(os.path.join(project_dir, ".current_exp"), 'w') as f:
            f.write(exp_dir_name)
    else:
        # Non-rank-0 processes wait a bit and then read the directory name
        import time
        time.sleep(0.1)  # Give rank 0 time to create the directory
        
        try:
            with open(os.path.join(project_dir, ".current_exp"), 'r') as f:
                exp_dir_name = f.read().strip()
            base_dir = os.path.join(project_dir, exp_dir_name)
        except (OSError, IOError):
            # Fallback if file doesn't exist
            exp_dir_name = f"00000_{args.exp_name}"
            base_dir = os.path.join(project_dir, exp_dir_name)

    # Load or save config
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config_file}")
    else:
        config = vars(args)

    # Save config in experiment folder
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to {config_path}")
    
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.exp_name,
        save_dir=base_dir,           # where to store wandb files
        log_model=True               # automatically upload checkpoints
    )
    
    # Prepare data and model
    # dm = MNISTDataModule(batch_size=args.batch_size)
    lr = args.lr*args.batch_size/64*args.gpus
    model = LitDenoiser(
        in_channels=args.in_channels,
        num_basis=args.num_basis,
        eta_base=args.eta_base,
        kernel_size=args.kernel_size,
        stride=args.stride,
        lr=lr,
        sigma=args.sigma,
        ema_halflife_kimg=args.ema_halflife_kimg,
        ema_rampup_ratio=args.ema_rampup_ratio,
        model_arch = args.model_arch,
        P_mean=args.P_mean,
        P_std=args.P_std,
        edm_weighting = args.edm_weighting,
        jfb_no_grad_iters=args.jfb_no_grad_iters,
        jfb_with_grad_iters=args.jfb_with_grad_iters,
        jfb_reuse_solution=args.jfb_reuse_solution,
        learning_horizontal=not args.no_learning_horizontal,
        noise_embedding=not args.no_noise_embedding,
        loss_type=args.loss_type,
        bias=not args.no_bias,
        relu_6=args.relu_6,
        T=args.T,
        per_dim_threshold=args.per_dim_threshold,
        positive_threshold=args.positive_threshold,
        multiscale=args.multiscale,
        constraint_energy=args.constraint_energy,
        intra=args.intra,
        k_inter=args.k_inter,
        n_hid_layers=args.n_hid_layers,
        tied_transpose=not args.no_tied_transpose,
        film=args.film,
        scalar_mul=args.scalar_mul,
        ff_scale=args.ff_scale,
        control_groups=args.control_groups,
        ignore_ff_control=args.ignore_ff_control,
        pyramid_level_weights=args.pyramid_level_weights,
        img_size=args.img_size,
        unet_variant=args.unet_variant,
        unet_base_width=args.unet_base_width,
        unet_no_attention=args.unet_no_attention,
        toy_dim=args.toy_dim,
        toy_hidden_dim=args.toy_hidden_dim,
        toy_learning_horizontal=args.toy_learning_horizontal,
        simple_control=args.simple_control,
    )
    
    # Load checkpoint if specified
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

        print(f"Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # For Lightning checkpoints, keep only model.* params and strip the prefix.
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                k[len("model."):]: v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }

        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Model load with strict=False. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        else:
            print("Loaded model weights.")

        ema_state = checkpoint.get("ema_model")
        if ema_state is not None:
            ema_missing, ema_unexpected = model.ema_model.load_state_dict(ema_state, strict=False)
            if ema_missing or ema_unexpected:
                print(f"EMA load with strict=False. Missing keys: {len(ema_missing)}, unexpected keys: {len(ema_unexpected)}")
            else:
                print("Loaded EMA model weights.")
        else:
            model.ema_model.load_state_dict(model.model.state_dict())
            print("No EMA weights found in checkpoint; copied model weights to EMA.")




    # Select DataModule based on dataset: mnist, stl10, celeba, berkeley, vh
    if args.dataset == "mnist":
        dm = MNISTDataModule(data_dir="./data", batch_size=args.batch_size)
    elif args.dataset == "stl10":
        dm = STL10DataModule(data_dir="~/data", batch_size=args.batch_size)
    elif args.dataset == "toy_pacman":
        dm = ToyPacManDataModule(
            num_samples=args.toy_num_samples,
            d_dim=args.toy_dim,
            batch_size=args.batch_size,
            num_workers=4,
            val_split=0.1,
            test_split=0.1,
            seed=42,
        )
    else:
        # celeba, berkeley, vh use ImageDataModule; per-dataset default data_dir unless --data_dir set
        if args.dataset not in ("celeba", "berkeley", "vh"):
            raise ValueError(f"Unknown dataset for ImageDataModule: {args.dataset}")
        if args.data_dir is not None:
            data_dir = os.path.expanduser(args.data_dir)
        else:
            _default_data_dirs = {
                "celeba": "~/celeba",
                "berkeley": "/home/zeyu/data/BSDS",
                "vh": "/home/zeyu/vanhateren_all/vh_patches256_train",
            }
            data_dir = os.path.expanduser(_default_data_dirs[args.dataset])
        dm = ImageDataModule(
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=4,
            img_size=args.img_size,
            val_split=0.1,
            test_split=0.1,
            seed=42,
            grayscale=args.grayscale,
            no_resize=args.no_resize,
            random_crop=args.random_crop,
        )


    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=base_dir,
        filename="denoiser",
        save_top_k=1,
        # monitor="val_loss",
        # mode="min",
        monitor="val_snr_db",
        mode="max",
        save_weights_only=True,
    )

    # EMA checkpoint callback
    ema_callback = EMACheckpointCallback(save_dir=base_dir)
    
    # Trainer setup
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.gpus if torch.cuda.is_available() else 1
    if accelerator == "gpu" and args.gpus > 1:
        strategy = "ddp_find_unused_parameters_true" if os.environ.get("DEBUG_DDP_UNUSED", "0") == "1" else "ddp"
    else:
        strategy = "auto"
    # strategy = 'auto'
    # import os
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        # strategy="ddp_find_unused_parameters_true",
        max_epochs=args.n_epochs,
        callbacks=[checkpoint_callback, ema_callback],
        logger=wandb_logger,
    )

    # Run training and testing
    trainer.fit(model, dm)
    trainer.test(model, dm)

    print(f"Best model weights saved at: {checkpoint_callback.best_model_path}")
