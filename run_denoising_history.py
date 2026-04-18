"""
run_denoising_history.py
========================
Generate denoising_history_dark_<tag>_nodcfilter.pdf.

For each stimulus, runs recurrent denoising with return_history=True and
renders a page of Gabor-overlay panels showing the intermediate steps.

Usage
-----
    python run_denoising_history.py [options]

Key defaults reproduce denoising_history_dark_00008_nodcfilter.pdf:
  --thr 0.5  --nw 0.1  --ff_thr 1.0  --n_drop 2  --n_stimuli 1
"""

import argparse
import io
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms

from recurrent_diffusion_pkg.utils import load_lit_model, build_loaders_from_config
from recurrent_diffusion_pkg.needle_plot import (
    fit_all_levels_gabor_bank,
    drop_top_norm_filters,
    extract_full_image_interactions,
    compute_per_neuron_thresholds,
    render_gabor_overlay,
    plot_gabor_diagnostic_grid,
)

# ── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate denoising history PDF")
    p.add_argument("--model_dir", type=str,
                   default="pretrained_model/scaling-VH-new-2/"
                           "00008_simple_sheet7_simple_control_small_noise_long_iter",
                   help="Path to pretrained model directory")
    p.add_argument("--stimulus_dir", type=str, default="stimulus",
                   help="Directory with shape_*.png and crop_*.png stimuli")
    p.add_argument("--out_dir", type=str, default="figures",
                   help="Output directory for the PDF")
    p.add_argument("--tag", type=str, default="00008",
                   help="Tag used in the output filename: denoising_history_dark_<tag>_nodcfilter.pdf")

    p.add_argument("--noise_level", type=float, default=0.4,
                   help="Noise sigma added to stimulus before denoising")
    p.add_argument("--n_iters", type=int, default=8,
                   help="Number of recurrent denoising iterations")
    p.add_argument("--n_stimuli", type=int, default=1,
                   help="Number of stimuli to process (0 = all)")
    p.add_argument("--shapes_only", action="store_true",
                   help="Only use shape_*.png stimuli (skip crop_*.png)")

    p.add_argument("--thr_fraction", type=float, default=0.03,
                   help="Per-neuron threshold fraction: thr = fraction × max(|activation|) per neuron")
    p.add_argument("--nw", type=float, default=0.1,
                   help="Needle width multiplier")
    p.add_argument("--intensity", type=float, default=1.0,
                   help="Needle intensity multiplier")
    p.add_argument("--ff_thr", type=float, default=1.0,
                   help="ff_threshold passed to fit_all_levels_gabor_bank")
    p.add_argument("--n_drop", type=int, default=2,
                   help="Number of top-norm (DC/Gaussian) filters to drop per level")

    p.add_argument("--n_plot_steps", type=int, default=4,
                   help="How many history steps to render per stimulus page")
    p.add_argument("--target_levels", type=int, nargs="+", default=[0, 1],
                   help="Feature levels to visualise (default: 0 1)")
    p.add_argument("--panel_size", type=float, default=4.0,
                   help="Inches per panel in the PDF")
    p.add_argument("--use_dataset", action="store_true",
                   help="Load stimuli from the dataset specified in model config instead of stimulus dir")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_to_rgb(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def main():
    args = parse_args()

    BASE_DIR     = Path(args.model_dir)
    STIMULUS_DIR = Path(args.stimulus_dir)
    OUT_DIR      = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {BASE_DIR.name} ...")
    config, lit_model = load_lit_model(BASE_DIR, device=DEVICE, ckpt_name=None)
    net = lit_model.ema_model.eval()
    for lvl in net.levels:
        lvl.reset_wnorm()

    import torch.nn as _nn
    _first_conv = next(m for m in net.levels[0].modules() if isinstance(m, _nn.Conv2d))
    n_channels = _first_conv.in_channels

    print(f"Fitting Gabor bank (ff_threshold={args.ff_thr}) ...")
    fit_by_level = fit_all_levels_gabor_bank(net, ff_threshold=args.ff_thr,levels = args.target_levels)
    if args.n_drop > 0:
        print(f"Dropping top-{args.n_drop} norm filters per level ...")
        drop_top_norm_filters(fit_by_level, n_drop=args.n_drop)

    diag_path = OUT_DIR / f"diag_grid_{args.tag}.png"
    print(f"Saving diagnostic grid → {diag_path} ...")
    plot_gabor_diagnostic_grid(
        net, fit_by_level,
        save_path=diag_path,
        n_show_per_level=16,
        needle_width_multiplier=args.nw,
    )

    # ── Load stimuli ──────────────────────────────────────────────────────────
    if args.use_dataset:
        print(f"Loading stimuli from dataset '{config.get('dataset')}' (config) ...")
        _, train_loader, _, _ = build_loaders_from_config(
            config, batch_size=max(args.n_stimuli, 1), num_workers=0,
        )
        batch = next(iter(train_loader))
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        n = args.n_stimuli if args.n_stimuli > 0 else imgs.shape[0]
        imgs = imgs[:n].to(DEVICE)
        stimuli = [(f"sample_{i:03d}", imgs[i].cpu()) for i in range(n)]
        print(f"Using {len(stimuli)} stimuli from dataset")
    else:
        shape_paths = sorted(STIMULUS_DIR.glob("shape_*.png"))
        if args.shapes_only:
            image_paths = shape_paths
        else:
            crop_paths  = sorted(STIMULUS_DIR.glob("crop_*.png"))
            image_paths = shape_paths + crop_paths

        if args.n_stimuli > 0:
            image_paths = image_paths[:args.n_stimuli]

        if n_channels == 1:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
            ])
        stimuli = [(p.stem, transform(Image.open(p))) for p in image_paths]
        print(f"Using {len(stimuli)} stimuli: {[n for n, _ in stimuli]}")

    # ── Run denoising ─────────────────────────────────────────────────────────
    clean = torch.stack([t for _, t in stimuli]).to(DEVICE)
    sigma = torch.full((len(clean),), args.noise_level,
                       device=DEVICE, dtype=clean.dtype)
    noisy = clean + torch.randn_like(clean) * sigma.view(-1, 1, 1, 1)

    print(f"Running denoising (n_iters={args.n_iters}, noise={args.noise_level}) ...")
    with torch.no_grad():
        history = net(
            noisy, noise_labels=sigma,
            infer_mode=True, n_iters=args.n_iters, return_history=True,
        )

    all_steps = len(history)
    plot_idx  = [int(round(i)) for i in
                 np.linspace(0, all_steps - 1, min(args.n_plot_steps, all_steps))]
    history   = [history[i] for i in plot_idx]
    n_steps   = len(history)
    print(f"History: {all_steps} total → plotting {n_steps} steps at {plot_idx}")

    # ── Per-neuron thresholds ─────────────────────────────────────────────────
    print(f"Computing per-neuron thresholds (fraction={args.thr_fraction}) ...")
    per_neuron_thr = [
        compute_per_neuron_thresholds(
            history, batch_idx=img_idx,
            fraction=args.thr_fraction, target_levels=args.target_levels,
        )
        for img_idx in range(len(stimuli))
    ]

    # ── Render helpers ────────────────────────────────────────────────────────
    def overlay_rgb(img_idx, step):
        snap  = history[step]
        specs = extract_full_image_interactions(
            a=snap["a"], batch_idx=img_idx,
            act_threshold=per_neuron_thr[img_idx],
            target_levels=args.target_levels,
        )
        fig = render_gabor_overlay(
            gabor_specs=specs, net=net, fit_by_level=fit_by_level,
            background_image=snap["denoised"][img_idx],
            dark_mode=True,
            intensity_multiplier=args.intensity,
            needle_width_multiplier=args.nw,
            return_fig=True,
        )
        rgb = fig_to_rgb(fig)
        plt.close(fig)
        return rgb

    def tensor_to_display(t):
        """Convert [C, H, W] tensor to numpy array for imshow."""
        t = t.cpu().float()
        if t.shape[0] == 1:
            return t[0].numpy(), "gray"
        arr = t.permute(1, 2, 0).numpy()
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)
        return np.clip(arr, 0, 1), None

    def make_page(img_idx, name):
        n_cols = n_steps + 1   # noisy/blank + n_steps
        fig = plt.figure(
            figsize=(n_cols * args.panel_size, 2 * args.panel_size),
            facecolor="white",
        )
        fig.patch.set_facecolor("white")
        gs = GridSpec(2, n_cols, figure=fig, hspace=0.06, wspace=0.03)

        _, cmap = tensor_to_display(clean[img_idx])
        noisy_arr, _ = tensor_to_display(noisy[img_idx])

        # Row 0: noisy | denoised at each step
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(noisy_arr, cmap=cmap, vmin=0, vmax=1)
        ax.set_title("noisy", fontsize=8, color="black", pad=3)
        ax.axis("off")
        ax.set_facecolor("white")

        for col in range(n_steps):
            arr, _ = tensor_to_display(history[col]["denoised"][img_idx])
            ax = fig.add_subplot(gs[0, col + 1])
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"step {plot_idx[col]}", fontsize=8, color="black", pad=3)
            ax.axis("off")
            ax.set_facecolor("white")

        # Row 1: blank | Gabor overlay at each step
        ax = fig.add_subplot(gs[1, 0])
        ax.axis("off")
        ax.set_facecolor("white")

        for col in range(n_steps):
            ax = fig.add_subplot(gs[1, col + 1])
            ax.imshow(overlay_rgb(img_idx, col))
            ax.axis("off")
            ax.set_facecolor("black")

        fig.suptitle(
            f"{name}   noise={args.noise_level}   n_iters={args.n_iters}   "
            f"thr_fraction={args.thr_fraction}   nw={args.nw}",
            fontsize=9, color="black", y=1.01,
        )
        return fig

    # ── Write PDF ─────────────────────────────────────────────────────────────
    out_path = OUT_DIR / f"denoising_history_{args.tag}_nodcfilter.pdf"
    print(f"\nWriting {out_path} ...")
    with PdfPages(out_path) as pdf:
        for img_idx, (name, _) in enumerate(stimuli):
            print(f"  [{img_idx+1}/{len(stimuli)}] {name}", end=" ", flush=True)
            fig = make_page(img_idx, name)
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print("done")

    print(f"\nSaved → {out_path}")
    print("All done.")


if __name__ == "__main__":
    main()
