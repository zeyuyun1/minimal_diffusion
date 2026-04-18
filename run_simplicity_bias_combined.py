"""
run_simplicity_bias_combined.py
================================
Mix training-set faces with 3 OOD stimulus shapes (circle, square, triangle)
and rank all images together by estimated log likelihood.

Estimator (MSE proxy)
---------------------
score(x) = -E_{σ, ε}[ ||D(x + σε, σ) - x||² ]

Simpler images are easier to denoise → lower MSE → higher score.
This is a forward-pass-only estimator (no Jacobian needed), much faster.

Two models compared side-by-side:
  Left  — UNet   00033_layer1_unet_small_edm_noatt
  Right — Sheet7 00008_simple_sheet7_simple_control_small_noise_long_iter

Visualization: full sorted strip showing the overall transition from
simple (high score) to complex (low score).  Shape images get an orange
border; faces get a thin blue border.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from recurrent_diffusion_pkg.utils import load_lit_model, build_loaders_from_config


def parse_args():
    p = argparse.ArgumentParser(description="Simplicity bias: rank images by MSE log-likelihood")
    p.add_argument("--unet_dir",  type=str,
                   default="pretrained_model/scaling-VH-new-2/00033_layer1_unet_small_edm_noatt")
    p.add_argument("--sheet_dir", type=str,
                   default="pretrained_model/scaling-VH-new-2/"
                           "00008_simple_sheet7_simple_control_small_noise_long_iter")
    p.add_argument("--stim_dir",  type=str, default="stimulus")
    p.add_argument("--out_dir",   type=str, default="figures")
    p.add_argument("--unet_only", action="store_true",
                   help="Only score/plot the UNet panel (figure 1.B)")
    p.add_argument("--n_train_imgs",   type=int,   default=300)
    p.add_argument("--n_iters_sheet",  type=int,   default=4)
    return p.parse_args()


# populated in main() after parse_args
UNET_DIR  = None
SHEET_DIR = None
STIM_DIR  = None
OUT_DIR   = None

SHAPE_FILES = [
    "shape_Circle.png",
    "shape_Square.png",
    "shape_Triangle_Equilateral.png",
]

SIGMA_MIN    = 0.05
SIGMA_MAX    = 2.0
N_SIGMA      = 8
N_NOISE_SAMP = 3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── MSE score estimator ───────────────────────────────────────────────────────

@torch.no_grad()
def score_image_mse(model_fn, x_clean, sigmas, n_samples=3):
    """
    score(x) = -E_{σ,ε}[ ||D(x + σε) - x||² ]
    Higher (less negative) = easier to denoise = simpler.
    """
    total_mse = 0.0
    count = 0
    for sigma in sigmas:
        for _ in range(n_samples):
            noise   = torch.randn_like(x_clean) * sigma
            x_noisy = x_clean + noise
            x_hat   = model_fn(x_noisy)
            mse     = ((x_hat - x_clean) ** 2).mean().item()
            total_mse += mse
            count += 1
    return -(total_mse / count)


# ── Load models ───────────────────────────────────────────────────────────────

def load_unet(model_dir, device):
    config, lit = load_lit_model(model_dir, device=device)
    net = lit.ema_model.eval().to(device)
    def make_fn(sigma):
        s = torch.full((1,), sigma, device=device, dtype=torch.float32)
        def fn(x): return net(x, s.expand(x.shape[0]))
        return fn
    sigmas = np.exp(np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), N_SIGMA)).tolist()
    def model_fn(x_noisy):
        # called per-sigma inside score_image_mse via closure; reuse sigma from caller
        raise RuntimeError("use make_fn")
    return config, net, make_fn, sigmas


def load_sheet7(model_dir, device, n_iters=4):
    config, lit = load_lit_model(model_dir, device=device)
    net = lit.ema_model.eval().to(device)
    for lvl in net.levels:
        lvl.reset_wnorm()
    def make_fn(sigma):
        s = torch.full((1,), sigma, device=device, dtype=torch.float32)
        def fn(x):
            return net(x, noise_labels=s.expand(x.shape[0]),
                       infer_mode=True, n_iters=n_iters, n_iters_grad=2)
        return fn
    sigmas = np.exp(np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), N_SIGMA)).tolist()
    return config, net, make_fn, sigmas


# ── Scoring loop ──────────────────────────────────────────────────────────────

def score_all(make_fn, sigmas, all_imgs, device, label):
    scores = []
    N = len(all_imgs)
    for i, x in enumerate(all_imgs):
        x = x.to(device)
        total = 0.0
        cnt   = 0
        for sigma in sigmas:
            fn = make_fn(sigma)
            for _ in range(N_NOISE_SAMP):
                noise   = torch.randn_like(x) * sigma
                x_noisy = x + noise
                with torch.no_grad():
                    x_hat = fn(x_noisy)
                total += ((x_hat - x) ** 2).mean().item()
                cnt   += 1
        scores.append(-(total / cnt))
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{N}  last={scores[-1]:.5f}")
    return np.array(scores)


# ── Load stimuli ──────────────────────────────────────────────────────────────

def load_stimuli(stim_dir, img_size=128):
    images, names = [], []
    for fname in SHAPE_FILES:
        p   = stim_dir / fname
        img = Image.open(p).convert("L").resize((img_size, img_size), Image.BILINEAR)
        t   = torch.from_numpy(np.array(img)).float() / 255.0
        images.append(t.unsqueeze(0).unsqueeze(0))   # (1,1,H,W)
        names.append(Path(fname).stem.replace("shape_", ""))
    return images, names


# ── Plotting: full transition strip ───────────────────────────────────────────

def _show(ax, t):
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = t.cpu().float().numpy()
    if arr.shape[0] == 1:
        arr = arr[0]
    else:
        arr = arr.transpose(1, 2, 0)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(arr.clip(0, 1))
    ax.set_xticks([]); ax.set_yticks([])


def build_band_sequences(scores, is_shape, n_cols, n_band_rows,
                         band_pct_edges=None):
    """
    Split the full ranking into percentile bands defined by band_pct_edges.
    band_pct_edges: list of percentile breakpoints, e.g. [0, 10, 40, 100]
      → bands [0-10%), [10-40%), [40-100%]

    For each band, return n_cols*n_band_rows images:
      - All shapes that fall in that band (inserted at their true rank position)
      - Remaining slots filled with uniformly sampled faces from that band

    Returns:
      bands      : list of lists of original indices (rank-ordered within band)
      rank_of    : dict orig_idx -> 0-based global rank
      band_ranges: list of (lo_pct, hi_pct) for labelling
    """
    if band_pct_edges is None:
        band_pct_edges = [0, 33, 67, 100]

    N_total    = len(scores)
    order      = np.argsort(scores)[::-1]           # best → worst
    rank_of    = {idx: r for r, idx in enumerate(order)}

    shape_idxs = [i for i, s in enumerate(is_shape) if s]
    N_slots    = n_cols * n_band_rows

    n_bands    = len(band_pct_edges) - 1
    bands      = []
    band_ranges = []

    for b in range(n_bands):
        lo_pct  = band_pct_edges[b]
        hi_pct  = band_pct_edges[b + 1]
        lo_rank = int(round(lo_pct / 100 * (N_total - 1)))
        hi_rank = int(round(hi_pct / 100 * (N_total - 1)))
        band_ranges.append((lo_pct, hi_pct))

        # All indices (shapes + faces) in this rank band
        band_order = order[lo_rank: hi_rank + 1]
        band_shape = [i for i in band_order if is_shape[i]]
        band_face  = [i for i in band_order if not is_shape[i]]

        n_face_show = max(0, N_slots - len(band_shape))
        if len(band_face) > 0:
            sample_pos = np.round(
                np.linspace(0, len(band_face) - 1, min(n_face_show, len(band_face)))
            ).astype(int)
            sel_faces = [band_face[p] for p in sample_pos]
        else:
            sel_faces = []

        combined = sorted(band_shape + sel_faces, key=lambda i: rank_of[i])
        bands.append(combined)

    return bands, rank_of, band_ranges


def plot_band_panel(fig, inner_gs, images, bands, rank_of,
                    is_shape, shape_name_map, N_total, band_ranges,
                    n_cols, n_band_rows, panel_x_center):
    """
    Fill a 6-row × n_cols grid split into 3 pairs of rows (bands).
    Adds percentile range labels between bands.
    """
    BAND_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]   # top / mid / bottom
    BAND_LABELS = [
        f"Top {band_ranges[0][1]}%  (simplest)",
        f"{band_ranges[1][0]}–{band_ranges[1][1]}%  (middle)",
        f"Bottom {100 - band_ranges[2][0]}%  (most complex)",
    ]

    for b, (band_idxs, (lo_pct, hi_pct), bcolor, blabel) in enumerate(
            zip(bands, band_ranges, BAND_COLORS, BAND_LABELS)):

        for slot, idx in enumerate(band_idxs):
            r = b * n_band_rows + slot // n_cols
            c = slot % n_cols
            ax = fig.add_subplot(inner_gs[r, c])
            _show(ax, images[idx])

            rank = rank_of[idx] + 1
            pct  = 100 * (rank - 1) / (N_total - 1)

            if is_shape[idx]:
                color, lw = "#FF8C00", 4
                ax.set_title(shape_name_map.get(idx, ""),
                             fontsize=6, color="#FF8C00", pad=2)
            else:
                color, lw = "#4A90D9", 0.8
                ax.set_title(f"{pct:.0f}%", fontsize=5, color="gray", pad=1)

            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color)
                sp.set_linewidth(lw)

        # Hide unused slots in band
        for slot in range(len(band_idxs), n_cols * n_band_rows):
            r = b * n_band_rows + slot // n_cols
            c = slot % n_cols
            ax = fig.add_subplot(inner_gs[r, c])
            ax.axis("off")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(0); np.random.seed(0)

    unet_dir  = Path(args.unet_dir)
    sheet_dir = Path(args.sheet_dir)
    stim_dir  = Path(args.stim_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Load stimuli ──────────────────────────────────────────────────────────
    print("Loading stimuli...")
    stim_imgs, stim_names = load_stimuli(stim_dir)
    print(f"  {len(stim_imgs)} shapes: {stim_names}")

    # ── Load training faces ───────────────────────────────────────────────────
    print("Loading training faces...")
    cfg0 = json.load(open(unet_dir / "config.json"))
    _, train_loader, _, _ = build_loaders_from_config(cfg0, batch_size=8, num_workers=2)
    train_imgs = []
    for batch, _ in train_loader:
        for img in batch:
            train_imgs.append(img.unsqueeze(0))
            if len(train_imgs) >= args.n_train_imgs: break
        if len(train_imgs) >= args.n_train_imgs: break
    print(f"  {len(train_imgs)} faces")

    all_imgs       = stim_imgs + train_imgs
    is_shape       = [True] * len(stim_imgs) + [False] * len(train_imgs)
    shape_name_map = {i: stim_names[i] for i in range(len(stim_imgs))}
    N_total        = len(all_imgs)
    print(f"  Total pool: {N_total}")

    # ── Score models ──────────────────────────────────────────────────────────
    models_to_run = [("UNet", lambda: load_unet(unet_dir, DEVICE))]
    if not args.unet_only:
        models_to_run.append(
            ("Sheet7", lambda: load_sheet7(sheet_dir, DEVICE, args.n_iters_sheet))
        )

    model_scores = {}
    for model_name, loader_fn in models_to_run:
        print(f"\n{'='*60}\n  [{model_name}]")
        _, _, make_fn, sigmas = loader_fn()
        scores = score_all(make_fn, sigmas, all_imgs, DEVICE, model_name)
        model_scores[model_name] = scores
        order = np.argsort(scores)[::-1]
        for i, idx in enumerate(order):
            if is_shape[idx]:
                print(f"  Shape {shape_name_map[idx]:30s}  rank {i+1}/{N_total}  "
                      f"score={scores[idx]:.5f}")
        print(f"  Score range: {scores.min():.5f} … {scores.max():.5f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    N_BAND_ROWS    = 2
    N_COLS         = 10
    CELL           = 1.3
    BAND_PCT_EDGES = [0, 10, 40, 100]
    BAND_LABELS    = ["Top 10%  (simplest)", "10–40%", "40–100%  (most complex)"]
    BAND_COLORS    = ["#2ecc71", "#f39c12", "#e74c3c"]

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    panel_names = list(model_scores.keys())
    n_panels    = len(panel_names)
    total_rows  = (len(BAND_PCT_EDGES) - 1) * N_BAND_ROWS

    fig = plt.figure(figsize=(n_panels * N_COLS * CELL + 1.2, total_rows * CELL + 2.0))
    outer = GridSpec(1, n_panels, figure=fig, wspace=0.06,
                     left=0.03, right=0.97, top=0.90, bottom=0.06)

    for col_idx, model_name in enumerate(panel_names):
        scores = model_scores[model_name]
        bands, rank_of, band_ranges = build_band_sequences(
            scores, is_shape, N_COLS, N_BAND_ROWS, band_pct_edges=BAND_PCT_EDGES)

        inner = GridSpecFromSubplotSpec(
            total_rows, N_COLS, subplot_spec=outer[col_idx],
            hspace=0.18, wspace=0.04,
        )
        x_center = 0.5 / n_panels + col_idx / n_panels
        plot_band_panel(fig, inner, all_imgs, bands, rank_of,
                        is_shape, shape_name_map, N_total, band_ranges,
                        N_COLS, N_BAND_ROWS, panel_x_center=x_center)

        fig.text(x_center, 0.93, model_name,
                 ha="center", fontsize=13, fontweight="bold")

        if col_idx == 0:
            for b, ((lo, hi), bcolor, blabel) in enumerate(
                    zip(band_ranges, BAND_COLORS, BAND_LABELS)):
                row_h = (0.90 - 0.06) / total_rows
                mid_y = 0.90 - row_h * (b * N_BAND_ROWS + N_BAND_ROWS / 2)
                fig.text(0.005, mid_y, blabel, ha="left", va="center",
                         fontsize=8, color=bcolor, fontweight="bold", rotation=90)

    handles = [
        mpatches.Patch(color="#FF8C00", label="Shape stimulus (OOD)"),
        mpatches.Patch(color="#4A90D9", label="Training face"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Simplicity bias — MSE likelihood ranking (shapes never seen in training)",
                 fontsize=11, y=1.00)

    out_name = "simplicity_bias_unet.png" if args.unet_only else "simplicity_bias_combined.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved → {out_path}")
    print("All done.")


if __name__ == "__main__":
    main()
