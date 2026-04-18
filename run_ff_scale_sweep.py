"""
run_ff_scale_sweep.py
=====================
Causal intervention: scale the feedforward (res_in) drive by ff_scale ∈ [0, 1]
at inference time, without adding any noise to the input image.

  ff_scale=1.0  →  normal operation (input fully drives activations)
  ff_scale=0.0  →  pure prior (network generates from learned distribution)

Works by temporarily patching forward_inter on the neural_sheet7 instance.
The model code is not modified.

Output: figures/ff_scale_sweep_<tag>.pdf
"""

import argparse
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from torchvision import transforms

from recurrent_diffusion_pkg.utils import load_lit_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str,
                   default="pretrained_model/scaling-VH-new-2/"
                           "00008_simple_sheet7_simple_control_small_noise_long_iter")
    p.add_argument("--clean_stimulus",   type=str, default="stimulus/shape_Circle_clean.png")
    p.add_argument("--corrupt_stimulus", type=str, default="stimulus/shape_Circle_perturbed.png")
    p.add_argument("--out_dir",  type=str, default="figures")
    p.add_argument("--tag",      type=str, default="00008_circle")
    p.add_argument("--n_iters",  type=int, default=8)
    p.add_argument("--noise_label", type=float, default=0.4,
                   help="Noise label passed to the model (no actual noise added to image)")
    p.add_argument("--ff_scales", type=float, nargs="+",
                   default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0])
    p.add_argument("--panel_size", type=float, default=3.0)
    return p.parse_args()


@contextmanager
def patch_ff_scale(net, ff_scale: float):
    """Temporarily scale res_in by ff_scale inside forward_inter."""
    original = net.forward_inter

    def patched_forward_inter(
        self, x, x_in, a, upstream_grad, decoded,
        noise_emb_ls=None, T=0, return_history=False,
        ablate_noi=None, ablate_mode="spatial_mean", measurement=None,
    ):
        if self.constraint_energy == "SC" and decoded[0] is not None:
            res_in = self.encoder_0(x - self.decoder_0(decoded))
        else:
            res_in = self.encoder_0(x)

        res_in = [r * ff_scale for r in res_in]

        self.forward_dynamics(a, res_in, upstream_grad, decoded, noise_emb_ls, T, ablate_noi, ablate_mode, reverse=False)
        snap1 = self._snapshot(a, decoded, upstream_grad) if return_history else None
        self.forward_dynamics(a, res_in, upstream_grad, decoded, noise_emb_ls, T, ablate_noi, ablate_mode, reverse=True)
        snap2 = self._snapshot(a, decoded, upstream_grad) if return_history else None

        if return_history:
            return [snap1, snap2]
        return None

    net.forward_inter = types.MethodType(patched_forward_inter, net)
    try:
        yield
    finally:
        net.forward_inter = original


def main():
    args = parse_args()
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {Path(args.model_dir).name} ...")
    _, lit_model = load_lit_model(Path(args.model_dir), device=DEVICE, ckpt_name=None)
    net = lit_model.ema_model.eval()
    for lvl in net.levels:
        lvl.reset_wnorm()

    import torch.nn as _nn
    n_channels = next(m for m in net.levels[0].modules() if isinstance(m, _nn.Conv2d)).in_channels

    # ── Load stimuli ──────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Grayscale(1) if n_channels == 1 else transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    stimuli = [
        ("clean circle",     transform(Image.open(args.clean_stimulus))),
        ("corrupted circle", transform(Image.open(args.corrupt_stimulus))),
    ]
    imgs = torch.stack([t for _, t in stimuli]).to(DEVICE)  # [2, C, H, W]
    sigma = torch.full((len(stimuli),), args.noise_label, device=DEVICE, dtype=imgs.dtype)

    # ── Sweep over ff_scale ───────────────────────────────────────────────────
    ff_scales = args.ff_scales
    results = {}   # ff_scale -> last denoised tensor [2, C, H, W]

    for ff_scale in ff_scales:
        print(f"  ff_scale={ff_scale:.2f} ...", end=" ", flush=True)
        with patch_ff_scale(net, ff_scale):
            with torch.no_grad():
                history = net(
                    imgs, noise_labels=sigma,
                    infer_mode=True, n_iters=args.n_iters, return_history=True,
                )
        results[ff_scale] = history[-1]["denoised"].cpu()
        print("done")

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / f"ff_scale_sweep_{args.tag}.pdf"
    print(f"\nWriting {out_path} ...")

    n_cols = len(ff_scales) + 1  # +1 for input

    with PdfPages(out_path) as pdf:
        for stim_idx, (stim_name, stim_tensor) in enumerate(stimuli):
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(n_cols * args.panel_size, args.panel_size),
                facecolor="black",
            )
            fig.patch.set_facecolor("black")
            for ax in axes:
                ax.set_facecolor("black")
                ax.axis("off")

            # col 0: input
            inp_arr = stim_tensor[0].numpy() if n_channels == 1 else stim_tensor.permute(1,2,0).numpy()
            axes[0].imshow(inp_arr, cmap="gray" if n_channels == 1 else None, vmin=0, vmax=1)
            axes[0].set_title("input", fontsize=7, color="white", pad=2)

            for col, ff_scale in enumerate(ff_scales):
                den = results[ff_scale][stim_idx]
                arr = den[0].float().numpy() if n_channels == 1 else den.permute(1,2,0).float().numpy()
                arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)
                axes[col + 1].imshow(arr, cmap="gray" if n_channels == 1 else None, vmin=0, vmax=1)
                axes[col + 1].set_title(f"ff={ff_scale:.1f}", fontsize=7, color="white", pad=2)

            fig.suptitle(
                f"{stim_name}  |  ff_scale sweep  |  noise_label={args.noise_label}  |  n_iters={args.n_iters}",
                fontsize=9, color="white", y=1.05,
            )
            plt.tight_layout(pad=0.3)
            pdf.savefig(fig, bbox_inches="tight", facecolor="black")
            plt.close(fig)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
