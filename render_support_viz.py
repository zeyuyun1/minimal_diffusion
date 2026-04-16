"""
Standalone script: renders sparse code activations as needle plots.
Run from recurrent_diffusion_minimal/:
    python render_support_viz.py
Saves figure to figures/render_support.png
"""
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from recurrent_diffusion_pkg.utils import (
    load_lit_model,
    fit_decoder_level_gabor_bank,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path("pretrained_model/scaling-VH-new-2/00005_simple_sheet7_simple_control_small_noise")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = Path("figures/render_support.png")

FF_THRESHOLD = 0.4
IMAGE_SIZE = (16, 16)
NUM_SPIKES = 10
MARGIN = 16
ACT_THRESHOLD = 0.1
NEEDLE_WIDTH_MULTIPLIER = 0.5   # linewidth = max(0.4, minor_sigma * this)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading model...")
config, lit_model = load_lit_model(BASE_DIR, device=DEVICE, ckpt_name=None)
net = lit_model.ema_model.eval()
print(f"n_levels: {net.n_levels}")

# ---------------------------------------------------------------------------
# Fit gabors
# ---------------------------------------------------------------------------
print("Fitting gabors...")
fit_by_level = {}
for level_idx in range(net.n_levels):
    fit_bank = fit_decoder_level_gabor_bank(
        model_net=net,
        level_idx=level_idx,
        ff_threshold=FF_THRESHOLD,
        unit_norm=True,
        use_color=True,
    )
    fit_by_level[level_idx] = fit_bank
    n_kept = len(fit_bank["kept_idx"])
    successful = sum(r.get("successful_fit", False) for r in fit_bank["results"])
    print(f"  L{level_idx}: kept={n_kept}, successful_fit={successful}")


# ---------------------------------------------------------------------------
# Clipping helper
# ---------------------------------------------------------------------------
def clip_line_segment(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Liang-Barsky line clipping. Returns (x0,y0,x1,y1) clipped, or None."""
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            u = qi / pi
            if pi < 0:
                u0 = max(u0, u)
            else:
                u1 = min(u1, u)
    if u0 > u1:
        return None
    return x0 + u0 * dx, y0 + u0 * dy, x0 + u1 * dx, y0 + u1 * dy


# ---------------------------------------------------------------------------
# render_support
# ---------------------------------------------------------------------------
def render_support(a, fit_by_level, net, batch_idx=0, ax=None,
                   act_threshold=0.01, exclude_levels=None,
                   needle_width_multiplier=0.5):
    """
    Render sparse code activations as needle glyphs.

    - Needle endpoints from result["x"]/result["y"] are in kernel pixel coords
      (integer centers 0..kw-1), so the valid clip window is [0, kw-1] x [0, kh-1].
    - Intensity = |sparse_code| * |weight_norm|, normalised globally and used as alpha.
    - Linewidth varies per neuron via the minor axis of the fitted Gabor.
    """
    if exclude_levels is None:
        exclude_levels = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.invert_yaxis()

    # --- First pass: collect all draw calls so we can normalise intensity globally ---
    draw_calls = []

    for level_idx, a_level in enumerate(a):
        if a_level is None or level_idx in exclude_levels:
            continue

        latent = a_level[batch_idx]
        fit_bank = fit_by_level[level_idx]
        kept_idxs = fit_bank["kept_idx"]
        results = fit_bank["results"]

        conv = net.levels[level_idx].decoder.conv
        dictionary = conv.weight_v if hasattr(conv, "weight_v") else conv.weight
        _, _, kh, kw = dictionary.shape

        stride_factor = 2 ** level_idx
        cx_kernel = (kw - 1) / 2.0
        cy_kernel = (kh - 1) / 2.0

        # The fitting uses pixel centers 0..kw-1 / 0..kh-1 as coordinate grid,
        # so the valid range for needle endpoints is exactly [0, kw-1] x [0, kh-1].
        xmin, xmax = 0.0, float(kw - 1)
        ymin, ymax = 0.0, float(kh - 1)

        # Weight norms for this level (used for intensity)
        enc = net.levels[level_idx].encoder_node.encoder
        weight_g = enc.weight_g.detach().cpu().float().flatten() if hasattr(enc, "weight_g") else None

        n_clipped = 0
        for i, channel_idx in enumerate(kept_idxs):
            result = results[i]
            if not result.get("successful_fit", False):
                continue

            x_arr = np.asarray(result["x"], dtype=np.float32)
            y_arr = np.asarray(result["y"], dtype=np.float32)
            if x_arr.ndim == 0 or y_arr.ndim == 0:
                continue  # scalar (center-surround), skip

            act_map = latent[channel_idx]
            yy, xx = torch.where(torch.abs(act_map) > act_threshold)
            if len(yy) == 0:
                continue

            # Clip needle to the kernel pixel grid [0, kw-1] x [0, kh-1]
            raw_len = np.hypot(x_arr[1] - x_arr[0], y_arr[1] - y_arr[0])
            clip_res = clip_line_segment(
                x_arr[0], y_arr[0], x_arr[1], y_arr[1],
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            )
            if clip_res is None:
                continue
            x_arr = np.array([clip_res[0], clip_res[2]], dtype=np.float32)
            y_arr = np.array([clip_res[1], clip_res[3]], dtype=np.float32)
            if np.hypot(x_arr[1] - x_arr[0], y_arr[1] - y_arr[0]) < raw_len - 1e-4:
                n_clipped += 1

            disp_x = x_arr - cx_kernel
            disp_y = y_arr - cy_kernel

            minor = min(float(result.get("sigma_x", 1.0)), float(result.get("sigma_y", 1.0)))
            lw = max(0.4, minor * needle_width_multiplier)

            w_norm = float(abs(weight_g[channel_idx])) if weight_g is not None else 1.0

            for y, x in zip(yy, xx):
                val = act_map[y, x].item()
                intensity = abs(val) * w_norm
                cx_global = x.item() * stride_factor
                cy_global = y.item() * stride_factor
                plot_x = cx_global + disp_x * stride_factor
                plot_y = cy_global + disp_y * stride_factor
                color = "darkred" if val > 0 else "navy"
                draw_calls.append((plot_x, plot_y, color, lw, intensity))

        print(f"  L{level_idx}: {n_clipped} needles clipped to [{xmin},{xmax}]x[{ymin},{ymax}]")

    if not draw_calls:
        return ax

    # Normalise intensity globally → alpha
    max_intensity = max(d[4] for d in draw_calls)
    min_intensity = min(d[4] for d in draw_calls)
    rng = max(max_intensity - min_intensity, 1e-8)

    for plot_x, plot_y, color, lw, intensity in draw_calls:
        alpha = 0.15 + 0.85 * (intensity - min_intensity) / rng
        ax.plot(plot_x, plot_y, color=color, alpha=float(alpha),
                linewidth=lw, solid_capstyle='round', zorder=3)

    return ax


# ---------------------------------------------------------------------------
# Test: random spike activations
# ---------------------------------------------------------------------------
def make_random_activations(net, image_size, num_spikes, allowed_levels, fit_by_level):
    device = next(net.parameters()).device
    a = []
    for i in range(net.n_levels):
        h_i = image_size[0] // (2 ** i)
        w_i = image_size[1] // (2 ** i)
        a.append(torch.zeros(1, net.levels[i].num_basis, h_i, w_i, device=device))

    print(f"Firing {num_spikes} random neurons...")
    for _ in range(num_spikes):
        lvl = random.choice(allowed_levels)
        fit_bank = fit_by_level[lvl]
        valid = [
            ki for ki, res in zip(fit_bank["kept_idx"], fit_bank["results"])
            if res.get("successful_fit", False) and "x" in res and "y" in res
        ]
        if not valid:
            continue
        c = random.choice(valid)
        _, _, h, w = a[lvl].shape
        y, x = random.randint(2, h - 3), random.randint(2, w - 3)
        strength = random.uniform(0.8, 1.2) * random.choice([-1.0, 1.0])
        a[lvl][0, c, y, x] = strength
        print(f"  L{lvl} | ch={c:03d} | pos=({y},{x}) | strength={strength:.2f}")
    return a


net = net.to(DEVICE)
for lvl in net.levels:
    lvl.reset_wnorm()

allowed_levels = [0, 1]
exclude = [i for i in range(net.n_levels) if i not in allowed_levels]
SAVE_PATH.parent.mkdir(exist_ok=True)

SEEDS = [42, 7, 123, 999, 2025, 314, 888, 17, 55, 1001]

for seed in SEEDS:
    print(f"\n--- seed={seed} ---")
    random.seed(seed)
    a_random = make_random_activations(net, IMAGE_SIZE, NUM_SPIKES, allowed_levels, fit_by_level)

    with torch.no_grad():
        decoded_ls = [net.levels[i].decoder(a_random[i]) for i in range(net.n_levels)]
        recon = net.decoder_0(decoded_ls)
        img = recon[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img, cmap='gray', extent=[0, IMAGE_SIZE[1], IMAGE_SIZE[0], 0])
    axes[0].set_facecolor('gray')
    axes[0].set_title("Network decoder output", fontsize=14)
    axes[1].set_facecolor('whitesmoke')
    axes[1].set_title(f"Needle plot (seed={seed})", fontsize=14)

    for ax in axes:
        ax.set_xlim(-MARGIN, IMAGE_SIZE[1] + MARGIN)
        ax.set_ylim(IMAGE_SIZE[0] + MARGIN, -MARGIN)
        ax.set_aspect('equal')

    render_support(a_random, fit_by_level, net, batch_idx=0, ax=axes[1],
                   act_threshold=ACT_THRESHOLD, exclude_levels=exclude,
                   needle_width_multiplier=NEEDLE_WIDTH_MULTIPLIER)

    plt.tight_layout()
    out = Path(f"figures/render_support_seed{seed}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")
