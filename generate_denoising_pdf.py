"""
generate_denoising_pdf.py

For the first N_STIMULI stimuli in stimulus/, runs denoising with return_history=True.
Each stimulus gets one PDF page: a 2-row grid of render_gabor_overlay figures,
where row 1 = early history steps, row 2 = late history steps.

Saves:
  figures/denoising_history_dark.pdf
  figures/denoising_history_light.pdf

Run from recurrent_diffusion_minimal/:
    python generate_denoising_pdf.py
"""

import io
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
from recurrent_diffusion_pkg.needle_plot import (
    fit_all_levels_gabor_bank,
    drop_top_norm_filters,
    extract_full_image_interactions,
    compute_per_neuron_thresholds,
    render_gabor_overlay,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("pretrained_model/scaling-VH-new-2/"
                     "00008_simple_sheet7_simple_control_small_noise_long_iter")
STIMULUS_DIR  = Path("stimulus")
OUT_DIR       = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_LEVEL   = 0.4
N_ITERS       = 8
N_STIMULI     = 999    # process all stimuli
N_HISTORY_ROWS = 4    # split history across this many rows
THR_FRACTION  = 0.03   # per-neuron threshold = fraction × max(|activation|) over history
TARGET_LEVELS = [0, 1]
NW_MULT       = 0.8
N_PLOT_STEPS  = 4      # subsample history to this many evenly-spaced steps


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
config, lit_model = load_lit_model(BASE_DIR, device=DEVICE, ckpt_name=None)
net = lit_model.ema_model.eval()
for lvl in net.levels:
    lvl.reset_wnorm()

print("Fitting Gabor bank...")
fit_by_level = fit_all_levels_gabor_bank(net, ff_threshold=1.0)
print("Dropping top-2 norm (Gaussian) filters per level...")
drop_top_norm_filters(fit_by_level, n_drop=2)

# ── Load stimuli ──────────────────────────────────────────────────────────────
shape_paths = sorted(STIMULUS_DIR.glob("shape_*.png"))
crop_paths  = sorted(STIMULUS_DIR.glob("crop_*.png"))
image_paths = (shape_paths + crop_paths)[:N_STIMULI]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
stimuli = [(p.stem, transform(Image.open(p))) for p in image_paths]
print(f"Using {len(stimuli)} stimuli: {[n for n,_ in stimuli]}")

# ── Run model ─────────────────────────────────────────────────────────────────
clean = torch.stack([t for _, t in stimuli]).to(DEVICE)
sigma = torch.full((len(clean),), NOISE_LEVEL, device=DEVICE, dtype=clean.dtype)
noisy = clean + torch.randn_like(clean) * sigma.view(-1, 1, 1, 1)

print(f"Running denoising (n_iters={N_ITERS})...")
with torch.no_grad():
    history = net(
        noisy, noise_labels=sigma,
        infer_mode=True, n_iters=N_ITERS, return_history=True,
    )
_all_steps = len(history)
_plot_idx  = [int(round(i)) for i in
              __import__("numpy").linspace(0, _all_steps - 1, min(N_PLOT_STEPS, _all_steps))]
history    = [history[i] for i in _plot_idx]
n_steps    = len(history)
print(f"History: {_all_steps} total → plotting {n_steps} steps at {_plot_idx}")

# Per-neuron thresholds (computed over the full original history before subsampling)
print("Computing per-neuron thresholds...")
per_neuron_thr = [
    compute_per_neuron_thresholds(
        history, batch_idx=img_idx,
        fraction=THR_FRACTION, target_levels=TARGET_LEVELS,
    )
    for img_idx in range(len(stimuli))
]
print(f"  threshold fraction={THR_FRACTION},  "
      f"example L0-ch0: {per_neuron_thr[0].get((0,0), float('nan')):.4f}")

# Single row of N_PLOT_STEPS steps
n_cols    = n_steps
step_rows = [list(range(n_steps))]
print(f"Layout: 1 history row × {n_cols} cols  "
      f"({[len(r) for r in step_rows]} steps per row)")


# ── Render helper ─────────────────────────────────────────────────────────────
def fig_to_rgb(fig):
    """Render matplotlib figure → H×W×3 uint8 numpy via PNG round-trip."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def overlay_rgb(img_idx, step, dark):
    snap  = history[step]
    specs = extract_full_image_interactions(
        a=snap["a"], batch_idx=img_idx,
        act_threshold=per_neuron_thr[img_idx], target_levels=TARGET_LEVELS,
    )
    fig = render_gabor_overlay(
        gabor_specs=specs, net=net, fit_by_level=fit_by_level,
        background_image=snap["denoised"][img_idx],
        dark_mode=dark, intensity_multiplier=1.0,
        needle_width_multiplier=NW_MULT, return_fig=True,
    )
    rgb = fig_to_rgb(fig)
    plt.close(fig)
    return rgb


# ── Build one page per stimulus ───────────────────────────────────────────────
PANEL_SIZE = 4.0   # inches per panel (larger = more needle detail)

def make_page(img_idx, name, dark):
    bg = "black" if dark else "white"
    fg = "white" if dark else "black"

    from matplotlib.gridspec import GridSpec

    n_rows_total = 1 + len(step_rows)   # reference row + history rows
    fig = plt.figure(
        figsize=(n_cols * PANEL_SIZE, n_rows_total * PANEL_SIZE),
        facecolor=bg,
    )
    fig.patch.set_facecolor(bg)
    gs = GridSpec(
        n_rows_total, n_cols, figure=fig,
        hspace=0.06, wspace=0.03,
        height_ratios=[1] * n_rows_total,
    )

    # --- Row 0: clean / noisy / final denoised (spread evenly) ---
    final_denoised = history[-1]["denoised"][img_idx, 0].cpu().float().numpy()
    final_denoised = (final_denoised - final_denoised.min()) / max(
        final_denoised.max() - final_denoised.min(), 1e-8)

    ref_positions = np.linspace(0, n_cols - 1, 3, dtype=int).tolist()
    ref_data = [
        (clean[img_idx, 0].cpu().numpy(), "clean"),
        (noisy[img_idx, 0].cpu().numpy(), "noisy"),
        (final_denoised,                  "denoised (final)"),
    ]
    for col, (arr, title) in zip(ref_positions, ref_data):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8, color=fg, pad=3)
        ax.axis("off")
        ax.set_facecolor(bg)

    # --- History rows ---
    for row, steps in enumerate(step_rows):
        for col, step in enumerate(steps):
            direction = "fwd" if step % 2 == 0 else "rev"
            ax = fig.add_subplot(gs[row + 1, col])
            ax.imshow(overlay_rgb(img_idx, step, dark))
            ax.set_title(f"step {step} ({direction})", fontsize=7, color=fg, pad=2)
            ax.axis("off")
            ax.set_facecolor(bg)

    mode_str = "dark" if dark else "light"
    fig.suptitle(
        f"{name}   noise={NOISE_LEVEL}   n_iters={N_ITERS}   [{mode_str}]",
        fontsize=9, color=fg, y=1.01,
    )
    return fig


# ── Save PDFs ─────────────────────────────────────────────────────────────────
for dark in [True]:
    mode = "dark" if dark else "light"
    out  = OUT_DIR / f"denoising_history_{mode}.pdf"
    print(f"\nWriting {out}...")
    with PdfPages(out) as pdf:
        for img_idx, (name, _) in enumerate(stimuli):
            print(f"  [{img_idx+1}/{len(stimuli)}] {name}", end=" ", flush=True)
            fig = make_page(img_idx, name, dark)
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print("✓")
    print(f"Saved → {out}")

print("\nDone.")
