"""
run_face_spreading.py

Same spreading/Jacobian visualization as run_contour_experiments.py, but:
  - Uses the face neural_sheet7 model (00026, RGB 64×64 CelebA)
  - Loads a face from the val loader instead of a stimulus PNG
  - Target neuron is a LOW-FF-NORM ("detached") neuron in the zoom window
    (not Gabor-fitted → no golden needle in the header panel)
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recurrent_diffusion_pkg.utils import load_lit_model, build_loaders_from_config
from recurrent_diffusion_pkg.needle_plot import (
    fit_all_levels_gabor_bank,
    drop_top_norm_filters,
    extract_full_image_interactions,
    compute_per_neuron_thresholds,
    render_gabor_overlay,
)
from generalization_plots import (
    fig_to_arr,
    SpreadContext,
    compute_spread_pair,
    make_combined_figure,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("pretrained_model/scaling-new-face/"
                "00026_simplify_layer1_neural_sheet7_intra_"
                "film_scalar_mul_ff_scale_large_large_noise_simple_control")
OUT_DIR  = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

VAL_IMAGE_IDX  = 2       # which image from the val batch to use
NOISE_LEVEL    = 0.3
N_ITERS        = 8
THR_FRACTION   = 0.15    # higher → fewer active needles (was 0.03)
TARGET_LEVELS  = [0, 1]
FF_THRESHOLD   = 0.1     # fit Gabors only for neurons above this FF norm
NW_MULT        = 0.8
INTENSITY_MULT = 1.0
DARK_MODE      = True
ZOOM_HALF      = 16      # 32×32 zoom window
RENDER_SCALE   = 4
TOP_K          = 9       # 3×3 spread grid
EDGE_MARGIN    = 2       # feature cells to exclude at image boundary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading face model...")
config, lit_model = load_lit_model(BASE_DIR, device=DEVICE)
net = lit_model.ema_model.eval()
for lvl in net.levels:
    lvl.reset_wnorm()

# ── Gabor bank: fitted neurons (high FF norm) ─────────────────────────────────
print("Fitting Gabor bank...")
fit_by_level = fit_all_levels_gabor_bank(net, ff_threshold=FF_THRESHOLD)
drop_top_norm_filters(fit_by_level, n_drop=2)

fitted_set = {
    (lvl, int(r["filter_idx"]))
    for lvl, fb in fit_by_level.items()
    for r in fb["results"]
    if r.get("successful_fit", False) and lvl in TARGET_LEVELS
}
print(f"Fitted neurons: { {lvl: sum(1 for l,c in fitted_set if l==lvl) for lvl in TARGET_LEVELS} }")

# Build set of ALL channels at each target level to find detached ones
all_channels = {
    lvl: set(range(net.levels[lvl].num_basis))
    for lvl in TARGET_LEVELS
}
fitted_channels = {
    lvl: {c for l, c in fitted_set if l == lvl}
    for lvl in TARGET_LEVELS
}
detached_channels = {
    lvl: all_channels[lvl] - fitted_channels[lvl]
    for lvl in TARGET_LEVELS
}
print(f"Detached neurons: { {lvl: len(chs) for lvl, chs in detached_channels.items()} }")

# ── Load a face image from val set ────────────────────────────────────────────
print("Loading face image from val set...")
_, _, val_loader, _ = build_loaders_from_config(
    config, batch_size=16, num_workers=2)
clean_batch, _ = next(iter(val_loader))
clean = clean_batch[[VAL_IMAGE_IDX]].to(DEVICE)   # (1, 3, 64, 64)
print(f"  Image shape: {clean.shape}  range: [{clean.min():.3f}, {clean.max():.3f}]")


# ── Zoom geometry ─────────────────────────────────────────────────────────────
def compute_zoom_geometry(denoised_final, a_final, target_level, target_y, target_x):
    IMG_H = denoised_final.shape[-2]
    IMG_W = denoised_final.shape[-1]
    H_feat = a_final[target_level].shape[2]
    W_feat = a_final[target_level].shape[3]
    tgt_pdx = ((target_x + 0.5) - W_feat / 2.0) * (2 ** (target_level + 1))
    tgt_pdy = ((target_y + 0.5) - H_feat / 2.0) * (2 ** (target_level + 1))
    cx = IMG_W / 2.0 + tgt_pdx
    cy = IMG_H / 2.0 + tgt_pdy
    x0_z = max(0, int(cx - ZOOM_HALF));  x1_z = min(IMG_W, int(cx + ZOOM_HALF))
    y0_z = max(0, int(cy - ZOOM_HALF));  y1_z = min(IMG_H, int(cy + ZOOM_HALF))
    dx_off = (x0_z + x1_z) / 2.0 - IMG_W / 2.0
    dy_off = (y0_z + y1_z) / 2.0 - IMG_H / 2.0
    return dict(IMG_H=IMG_H, IMG_W=IMG_W, tgt_pdx=tgt_pdx, tgt_pdy=tgt_pdy,
                x0_z=x0_z, x1_z=x1_z, y0_z=y0_z, y1_z=y1_z,
                dx_off=dx_off, dy_off=dy_off)


# ── Find detached (low-FF) neuron in zoom window ──────────────────────────────
def find_detached_neuron(a_final, zoom):
    """
    Find the most activated detached (non-Gabor) neuron whose receptive-field
    centre falls within the zoom window.
    """
    best_val = -1.0
    tgt_lvl = tgt_ch = tgt_y = tgt_x = None

    for lvl in TARGET_LEVELS:
        a_lvl  = a_final[lvl]
        H_lvl  = a_lvl.shape[2];  W_lvl = a_lvl.shape[3]
        IMG_H  = a_lvl.shape[2] * (2 ** (lvl + 1))
        IMG_W  = a_lvl.shape[3] * (2 ** (lvl + 1))

        # Convert zoom window bounds to feature coords, with edge margin
        x0_feat = zoom["x0_z"] / IMG_W * W_lvl
        x1_feat = zoom["x1_z"] / IMG_W * W_lvl
        y0_feat = zoom["y0_z"] / IMG_H * H_lvl
        y1_feat = zoom["y1_z"] / IMG_H * H_lvl
        x0f = max(EDGE_MARGIN, int(x0_feat))
        x1f = min(W_lvl - EDGE_MARGIN, int(x1_feat) + 1)
        y0f = max(EDGE_MARGIN, int(y0_feat))
        y1f = min(H_lvl - EDGE_MARGIN, int(y1_feat) + 1)

        for ch in detached_channels[lvl]:
            region = a_lvl[0, ch, y0f:y1f, x0f:x1f].abs()
            if region.numel() == 0:
                continue
            val = region.max().item()
            if val > best_val:
                best_val = val
                flat_idx = region.argmax()
                ry = (flat_idx // region.shape[1]).item()
                rx = (flat_idx %  region.shape[1]).item()
                tgt_lvl = lvl;  tgt_ch = ch
                tgt_y   = y0f + ry
                tgt_x   = x0f + rx

    return tgt_lvl, tgt_ch, tgt_y, tgt_x


# ── compute_top_k: all fitted Gabors anywhere in the zoom window ──────────────
def compute_top_k(a_final, per_neuron_thr, zoom, target_level, target_y, target_x,
                  target_channel, k=TOP_K):
    """
    Collect every fitted Gabor neuron whose receptive-field centre falls inside
    the zoom window, ranked by activation.  The detached target neuron is always
    prepended so it appears as the first spread slot.
    """
    tgt_pdx = zoom["tgt_pdx"];  tgt_pdy = zoom["tgt_pdy"]
    x0_z, x1_z = zoom["x0_z"], zoom["x1_z"]
    y0_z, y1_z = zoom["y0_z"], zoom["y1_z"]

    target_loc_specs = []
    for lvl in TARGET_LEVELS:
        H_lvl = a_final[lvl].shape[2];  W_lvl = a_final[lvl].shape[3]
        IMG_H = H_lvl * (2 ** (lvl + 1));  IMG_W = W_lvl * (2 ** (lvl + 1))

        # Convert zoom bounds → feature coords
        x0f = max(0,     int(x0_z / IMG_W * W_lvl))
        x1f = min(W_lvl, int(x1_z / IMG_W * W_lvl) + 1)
        y0f = max(0,     int(y0_z / IMG_H * H_lvl))
        y1f = min(H_lvl, int(y1_z / IMG_H * H_lvl) + 1)

        fit_map = {int(r["filter_idx"]): r
                   for r in fit_by_level[lvl]["results"]
                   if r.get("successful_fit", False)}

        for yy in range(y0f, y1f):
            for xx in range(x0f, x1f):
                for ch in fit_map:
                    w = a_final[lvl][0, ch, yy, xx].item()
                    if abs(w) < per_neuron_thr.get((lvl, ch), 0.0):
                        continue
                    # Pin needle to this neuron's own pixel position
                    px = ((xx + 0.5) - W_lvl / 2.0) * (2 ** (lvl + 1))
                    py = ((yy + 0.5) - H_lvl / 2.0) * (2 ** (lvl + 1))
                    target_loc_specs.append({
                        "level": lvl, "channel": ch, "weight": w,
                        "pixel_dx": px, "pixel_dy": py,
                        "y_feat": yy, "x_feat": xx,
                    })

    # Always include the detached target neuron (pinned to its own position)
    if not any(s["level"] == target_level and s["channel"] == target_channel
               and s["y_feat"] == target_y and s["x_feat"] == target_x
               for s in target_loc_specs):
        H_t = a_final[target_level].shape[2];  W_t = a_final[target_level].shape[3]
        px = ((target_x + 0.5) - W_t / 2.0) * (2 ** (target_level + 1))
        py = ((target_y + 0.5) - H_t / 2.0) * (2 ** (target_level + 1))
        w = a_final[target_level][0, target_channel, target_y, target_x].item()
        target_loc_specs.insert(0, {
            "level": target_level, "channel": target_channel, "weight": w,
            "pixel_dx": px, "pixel_dy": py,
            "y_feat": target_y, "x_feat": target_x,
        })

    ranked = sorted(target_loc_specs, key=lambda s: abs(s["weight"]), reverse=True)
    print(f"  Fitted Gabors in zoom window: {len(target_loc_specs)} → showing top {k}")
    return ranked[:k]


# ── Header panel helpers ──────────────────────────────────────────────────────
def _img_arr(t):
    """Tensor (1, C, H, W) or (C, H, W) → normalised uint8 H×W×3."""
    a = t.squeeze(0).cpu().float()   # (C, H, W)
    lo, hi = a.min(), a.max()
    a = (a - lo) / max((hi - lo).item(), 1e-8)
    a = a.clamp(0, 1)
    if a.shape[0] == 1:
        a = a.repeat(3, 1, 1)
    return (a.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_1a_arr(specs, denoised_final, zoom):
    fig = render_gabor_overlay(
        gabor_specs=specs, net=net, fit_by_level=fit_by_level,
        background_image=denoised_final[0],
        dark_mode=False, intensity_multiplier=INTENSITY_MULT,
        needle_width_multiplier=NW_MULT, return_fig=True,
    )
    import matplotlib.patches as mpatches
    x0_z, x1_z = zoom["x0_z"], zoom["x1_z"]
    y0_z, y1_z = zoom["y0_z"], zoom["y1_z"]
    fig.axes[0].add_patch(mpatches.Rectangle(
        (x0_z * RENDER_SCALE, y0_z * RENDER_SCALE),
        (x1_z - x0_z) * RENDER_SCALE, (y1_z - y0_z) * RENDER_SCALE,
        linewidth=3, edgecolor="red", facecolor="none", zorder=10,
    ))
    fig.axes[0].set_xlim(0, zoom["IMG_W"] * RENDER_SCALE)
    fig.axes[0].set_ylim(zoom["IMG_H"] * RENDER_SCALE, 0)
    arr = fig_to_arr(fig, dpi=150)
    plt.close(fig)
    return arr


def make_1b_arr(specs, denoised_final, zoom):
    x0_z, x1_z = zoom["x0_z"], zoom["x1_z"]
    y0_z, y1_z = zoom["y0_z"], zoom["y1_z"]
    dx_off, dy_off = zoom["dx_off"], zoom["dy_off"]
    IMG_W, IMG_H   = zoom["IMG_W"], zoom["IMG_H"]

    specs_zoom = [
        {**s, "pixel_dx": s["pixel_dx"] - dx_off,
              "pixel_dy": s["pixel_dy"] - dy_off}
        for s in specs
        if (x0_z <= IMG_W / 2.0 + s["pixel_dx"] < x1_z and
            y0_z <= IMG_H / 2.0 + s["pixel_dy"] < y1_z)
    ]
    bg_crop = denoised_final[0, :, y0_z:y1_z, x0_z:x1_z]

    fig = render_gabor_overlay(
        gabor_specs=specs_zoom, net=net, fit_by_level=fit_by_level,
        background_image=bg_crop,
        dark_mode=False, intensity_multiplier=INTENSITY_MULT * 2.0,
        needle_width_multiplier=NW_MULT * 2.0, return_fig=True,
    )
    fig.axes[0].set_xlim(0, bg_crop.shape[-1] * RENDER_SCALE)
    fig.axes[0].set_ylim(bg_crop.shape[-2] * RENDER_SCALE, 0)
    arr = fig_to_arr(fig, dpi=150)
    plt.close(fig)
    return arr, bg_crop


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Face spreading experiment")
print(f"{'='*60}")

# Denoise
sigma = torch.full((1,), NOISE_LEVEL, device=DEVICE, dtype=clean.dtype)
noisy = clean + torch.randn_like(clean) * NOISE_LEVEL

with torch.no_grad():
    history = net(noisy, noise_labels=sigma,
                  infer_mode=True, n_iters=N_ITERS, return_history=True)

final          = history[-1]
a_final        = final["a"]
denoised_final = final["denoised"]

# Thresholds & full-image needle specs
per_neuron_thr = compute_per_neuron_thresholds(
    history, batch_idx=0, fraction=THR_FRACTION, target_levels=TARGET_LEVELS)
specs = extract_full_image_interactions(
    a=a_final, batch_idx=0,
    act_threshold=per_neuron_thr, target_levels=TARGET_LEVELS)
print(f"  Active needles: {len(specs)}")

# First pass: pick a zoom centre using most-activated Gabor neuron (interior only)
# (just to define the window; the *target* will be the detached neuron inside)
best_gabor_val = -1.0
gabor_anchor_level = gabor_anchor_y = gabor_anchor_x = None
for lvl in TARGET_LEVELS:
    H_lvl = a_final[lvl].shape[2];  W_lvl = a_final[lvl].shape[3]
    for lvl2, ch in fitted_set:
        if lvl2 != lvl:
            continue
        # Restrict to interior to avoid boundary artifacts
        sp = a_final[lvl][0, ch,
                          EDGE_MARGIN:H_lvl - EDGE_MARGIN,
                          EDGE_MARGIN:W_lvl - EDGE_MARGIN].abs()
        if sp.numel() == 0:
            continue
        val = sp.max().item()
        if val > best_gabor_val:
            best_gabor_val = val
            yx = sp.argmax()
            gabor_anchor_level = lvl
            gabor_anchor_y = EDGE_MARGIN + (yx // sp.shape[1]).item()
            gabor_anchor_x = EDGE_MARGIN + (yx %  sp.shape[1]).item()

zoom = compute_zoom_geometry(denoised_final, a_final,
                             gabor_anchor_level, gabor_anchor_y, gabor_anchor_x)

# Now find the detached neuron inside that window
target_level, target_channel, target_y, target_x = find_detached_neuron(a_final, zoom)
print(f"  Detached target: L{target_level} ch{target_channel} @ (y={target_y}, x={target_x})")

# Recompute zoom centred on the detached neuron
zoom = compute_zoom_geometry(denoised_final, a_final,
                             target_level, target_y, target_x)

# Header panels
arr_1a          = make_1a_arr(specs, denoised_final, zoom)
arr_1b, bg_crop = make_1b_arr(specs, denoised_final, zoom)

# Top-k Gabor neurons in window (spreading visualisation)
top_k = compute_top_k(a_final, per_neuron_thr, zoom,
                      target_level, target_y, target_x, target_channel)
print(f"  Top-{len(top_k)} neurons: "
      + ", ".join(f"L{s['level']}ch{s['channel']}" for s in top_k))

# SpreadContext
decoded_final = final["decoded"]
with torch.no_grad():
    if net.constraint_energy == "SC" and decoded_final[0] is not None:
        actual_res_in = net.encoder_0(noisy - net.decoder_0(decoded_final))
    else:
        actual_res_in = net.encoder_0(noisy)
    noise_emb_ls = (
        [net.film_modulation(sigma, i) for i in range(net.n_levels)]
        if net.noise_embedding else
        [net.lambda_bias for _ in range(net.n_levels)]
    )

clean_crop = clean[0, :, zoom["y0_z"]:zoom["y1_z"], zoom["x0_z"]:zoom["x1_z"]]

ctx = SpreadContext(
    a_final=a_final, net=net, fit_by_level=fit_by_level,
    bg_crop=bg_crop, clean_crop=clean_crop,
    target_level=target_level, target_channel=target_channel,
    tgt_pdx=zoom["tgt_pdx"], tgt_pdy=zoom["tgt_pdy"],
    x0_z=zoom["x0_z"], x1_z=zoom["x1_z"],
    y0_z=zoom["y0_z"], y1_z=zoom["y1_z"],
    dx_off=zoom["dx_off"], dy_off=zoom["dy_off"],
    IMG_W=zoom["IMG_W"], IMG_H=zoom["IMG_H"],
    TARGET_LEVELS=TARGET_LEVELS,
    actual_res_in=actual_res_in, noise_emb_ls=noise_emb_ls,
    nw_mult=NW_MULT, dark_mode=DARK_MODE,
)

# Spread pairs
spread_pairs = []
for k, s in enumerate(top_k):
    print(f"  Spread #{k+1}/{len(top_k)} L{s['level']}ch{s['channel']} ...",
          end=" ", flush=True)
    pair = compute_spread_pair(
        s["level"], s["channel"], s["y_feat"], s["x_feat"], ctx)
    spread_pairs.append(pair)
    print("done")

# Header thumbnails (RGB-aware)
arr_clean_thumb    = _img_arr(clean)
arr_noisy_thumb    = _img_arr(noisy)
arr_denoised_thumb = _img_arr(denoised_final)

# 6th panel: plain zoomed clean image (no needle — detached neuron has no Gabor)
bg_clean_crop  = clean[0, :, zoom["y0_z"]:zoom["y1_z"], zoom["x0_z"]:zoom["x1_z"]]
arr_neuron_zoom = _img_arr(bg_clean_crop.unsqueeze(0))

# Combined figure
zoom_factor = zoom["IMG_W"] / max(zoom["x1_z"] - zoom["x0_z"], 1)
fig_combined = make_combined_figure(
    arr_1a=arr_1a, arr_1b=arr_1b,
    spread_pairs=spread_pairs,
    n_show=8, dark_mode=False,
    panel_w=3.0, panel_h=3.0,
    label_1a="Support set",
    label_1b=f"Zoom  ×{zoom_factor:.1f}",
    spread_specs=top_k,
    arr_clean=arr_clean_thumb,
    arr_noisy=arr_noisy_thumb,
    arr_denoised=arr_denoised_thumb,
    arr_neuron_zoom=arr_neuron_zoom,
)

out_path = OUT_DIR / "gen_combined_face.png"
fig_combined.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white")
plt.close(fig_combined)
print(f"  Saved → {out_path}")
print("\nAll done.")
