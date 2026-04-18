"""
run_noise_label_sweep.py
========================
Feed CLEAN artificial stimuli into the model while sweeping the noise LABEL.
No actual noise is added — the label alone controls how strongly the model
applies its learned prior vs. the feedforward input.

Layout: rows = stimuli, col 0 = clean input, cols 1..N = denoised at each σ.

Output: figures/noise_label_sweep_<tag>.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from recurrent_diffusion_pkg.utils import load_lit_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str,
                   default="pretrained_model/scaling-VH-new-2/"
                           "00008_simple_sheet7_simple_control_small_noise_long_iter")
    p.add_argument("--stim_dir",  type=str, default="stimulus",
                   help="Directory with shape_*.png stimuli")
    p.add_argument("--out_dir",   type=str, default="figures")
    p.add_argument("--tag",       type=str, default="00008")
    p.add_argument("--n_iters",   type=int, default=8)
    p.add_argument("--noise_labels", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    p.add_argument("--panel_size", type=float, default=1.6)
    return p.parse_args()


def to_display(t, n_channels):
    """(C, H, W) tensor → HxW or HxWx3 float [0,1]."""
    t = t.cpu().float()
    t = (t - t.min()) / max((t.max() - t.min()).item(), 1e-8)
    t = t.clamp(0, 1)
    if n_channels == 1:
        return t[0].numpy(), "gray"
    return t.permute(1, 2, 0).numpy(), None


def main():
    args = parse_args()
    OUT_DIR  = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {Path(args.model_dir).name} ...")
    config, lit_model = load_lit_model(Path(args.model_dir), device=DEVICE, ckpt_name=None)
    net = lit_model.ema_model.eval()
    for lvl in net.levels:
        lvl.reset_wnorm()

    import torch.nn as _nn
    _first_conv = next(m for m in net.levels[0].modules() if isinstance(m, _nn.Conv2d))
    n_channels = _first_conv.in_channels
    print(f"  n_channels={n_channels}")

    # ── Load all shape stimuli ────────────────────────────────────────────────
    stim_paths = sorted(Path(args.stim_dir).glob("shape_*.png"))
    if n_channels == 1:
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ])

    stimuli = [(p.stem, transform(Image.open(p))) for p in stim_paths]
    print(f"  {len(stimuli)} stimuli: {[n for n,_ in stimuli]}")

    noise_labels = args.noise_labels
    n_labels = len(noise_labels)
    n_stim   = len(stimuli)

    # ── Run model: batch all stimuli, one pass per noise label ────────────────
    imgs = torch.stack([t for _, t in stimuli]).to(DEVICE)   # (N, C, H, W)

    # results[label_idx] = history list (last entry has "denoised")
    results = []
    for sigma_val in noise_labels:
        sigma = torch.full((n_stim,), sigma_val, device=DEVICE, dtype=imgs.dtype)
        print(f"  σ_label={sigma_val:.3f} ...", end=" ", flush=True)
        with torch.no_grad():
            history = net(
                imgs, noise_labels=sigma,
                infer_mode=True, n_iters=args.n_iters, return_history=True,
            )
        results.append(history)
        print("done")

    # ── Build figure: rows=stimuli, cols=input+noise_labels ──────────────────
    n_cols   = n_labels + 1   # col 0 = input
    cell     = args.panel_size
    fig_w    = n_cols * cell
    fig_h    = n_stim * cell

    fig, axes = plt.subplots(n_stim, n_cols,
                              figsize=(fig_w, fig_h),
                              facecolor="white")
    if n_stim == 1:
        axes = axes[np.newaxis, :]

    for row, (stim_name, stim_tensor) in enumerate(stimuli):
        # Col 0: clean input
        arr, cmap = to_display(stim_tensor, n_channels)
        axes[row, 0].imshow(arr, cmap=cmap, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(stim_name.replace("shape_", ""),
                                fontsize=6, rotation=0, labelpad=55,
                                va="center", color="#333333")
        if row == 0:
            axes[row, 0].set_title("input", fontsize=7, pad=3, color="#333333")

        # Cols 1..N: denoised at each noise label
        for col, (sigma_val, history) in enumerate(zip(noise_labels, results)):
            denoised = history[-1]["denoised"][row]   # (C, H, W)
            arr, cmap = to_display(denoised, n_channels)
            axes[row, col + 1].imshow(arr, cmap=cmap, vmin=0, vmax=1)
            if row == 0:
                axes[row, col + 1].set_title(f"σ={sigma_val:.2f}",
                                              fontsize=7, pad=3, color="#333333")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#dddddd")
            sp.set_linewidth(0.5)

    fig.suptitle(
        f"Noise label sweep — {Path(args.model_dir).name}  |  n_iters={args.n_iters}  |  clean input",
        fontsize=9, y=1.002, color="#222222",
    )
    plt.tight_layout(pad=0.3)

    out_path = OUT_DIR / f"noise_label_sweep_{args.tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
