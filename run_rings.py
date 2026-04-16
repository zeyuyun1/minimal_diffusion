"""
run_contour_experiments.py

Batch-runs the generalization experiment (denoise → find target neuron →
compute spread pairs → make combined figure) for every stimulus listed in
STIMULI, using the same logic as generalization.ipynb.

Outputs: figures/gen_combined_{stem}.png  for each stimulus.

Usage:
    python run_contour_experiments.py
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from recurrent_diffusion_pkg.utils import load_lit_model
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
BASE_DIR    = Path("pretrained_model/scaling-VH-new-2/"
                   "00008_simple_sheet7_simple_control_small_noise_long_iter")
OUT_DIR     = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

STIMULI = [
    "shape_Circle_Ring_thin",
    "shape_Circle_Ring_medium",
]

NOISE_LEVEL    = 0.4
N_ITERS        = 8
THR_FRACTION   = 0.03
TARGET_LEVELS  = [0, 1]
NW_MULT        = 0.8
INTENSITY_MULT = 1.0
DARK_MODE      = True
ZOOM_HALF      = 14
RENDER_SCALE   = 4
TOP_K          = 6
WIN_HALF       = {1: 1, 0: 3}   # 1c coarse-cell window half-widths per level

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model (once) ─────────────────────────────────────────────────────────
print("Loading model...")
config, lit_model = load_lit_model(BASE_DIR, device=DEVICE)
net = lit_model.ema_model.eval()
for lvl in net.levels:
    lvl.reset_wnorm()

print("Fitting Gabor bank...")
fit_by_level = fit_all_levels_gabor_bank(net, ff_threshold=1.0)
drop_top_norm_filters(fit_by_level, n_drop=2)

fitted_set = {
    (lvl, int(r["filter_idx"]))
    for lvl, fb in fit_by_level.items()
    for r in fb["results"]
    if r.get("successful_fit", False) and lvl in TARGET_LEVELS
}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# ── Per-stimulus helpers ──────────────────────────────────────────────────────

def find_target_neuron(a_final):
    best_val = -1.0
    tgt_lvl = tgt_ch = tgt_y = tgt_x = None
    for lvl, a_lvl in enumerate(a_final):
        if lvl not in TARGET_LEVELS:
            continue
        for ch in range(a_lvl.shape[1]):
            if (lvl, ch) not in fitted_set:
                continue
            spatial = a_lvl[0, ch].abs()
            val = spatial.max().item()
            if val > best_val:
                best_val = val
                yx = spatial.argmax()
                tgt_lvl = lvl; tgt_ch = ch
                tgt_y = (yx // spatial.shape[1]).item()
                tgt_x = (yx %  spatial.shape[1]).item()
    return tgt_lvl, tgt_ch, tgt_y, tgt_x


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


def compute_top_k(a_final, per_neuron_thr, zoom, target_level, target_y, target_x,
                  target_channel, k=TOP_K):
    max_lvl  = max(TARGET_LEVELS)
    sc2max   = 2 ** (max_lvl - target_level)
    y_coarse = target_y // sc2max
    x_coarse = target_x // sc2max
    tgt_pdx  = zoom["tgt_pdx"];  tgt_pdy = zoom["tgt_pdy"]

    target_loc_specs = []
    for lvl in TARGET_LEVELS:
        H_lvl = a_final[lvl].shape[2];  W_lvl = a_final[lvl].shape[3]
        scale  = 2 ** (max_lvl - lvl)
        y_cen  = y_coarse * scale;  x_cen = x_coarse * scale
        half   = WIN_HALF[lvl]
        y0 = max(0, y_cen - half);  y1 = min(H_lvl, y_cen + half + 1)
        x0 = max(0, x_cen - half);  x1 = min(W_lvl, x_cen + half + 1)
        fit_map = {int(r["filter_idx"]): r
                   for r in fit_by_level[lvl]["results"]
                   if r.get("successful_fit", False)}
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                for ch, r in fit_map.items():
                    w = a_final[lvl][0, ch, yy, xx].item()
                    if abs(w) < per_neuron_thr.get((lvl, ch), 0.0):
                        continue
                    target_loc_specs.append({
                        "level": lvl, "channel": ch, "weight": w,
                        "pixel_dx": tgt_pdx, "pixel_dy": tgt_pdy,
                        "y_feat": yy, "x_feat": xx,
                    })

    # Always include the target neuron itself
    if not any(s["level"] == target_level and s["channel"] == target_channel
               and s["y_feat"] == target_y and s["x_feat"] == target_x
               for s in target_loc_specs):
        w = a_final[target_level][0, target_channel, target_y, target_x].item()
        target_loc_specs.insert(0, {
            "level": target_level, "channel": target_channel, "weight": w,
            "pixel_dx": tgt_pdx, "pixel_dy": tgt_pdy,
            "y_feat": target_y, "x_feat": target_x,
        })

    ranked = sorted(target_loc_specs, key=lambda s: abs(s["weight"]), reverse=True)
    return ranked[:k]


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


# ── Main loop ─────────────────────────────────────────────────────────────────

for stem in STIMULI:
    stim_path = Path("stimulus") / f"{stem}.png"
    if not stim_path.exists():
        print(f"  SKIP (not found): {stim_path}")
        continue
    print(f"\n{'='*60}")
    print(f"  Stimulus: {stem}")
    print(f"{'='*60}")

    # ── Load & denoise ────────────────────────────────────────────────────────
    clean = transform(Image.open(stim_path)).unsqueeze(0).to(DEVICE)
    sigma = torch.full((1,), NOISE_LEVEL, device=DEVICE, dtype=clean.dtype)
    torch.manual_seed(42)
    noisy = clean + torch.randn_like(clean) * NOISE_LEVEL

    with torch.no_grad():
        history = net(noisy, noise_labels=sigma,
                      infer_mode=True, n_iters=N_ITERS, return_history=True)

    final          = history[-1]
    a_final        = final["a"]
    denoised_final = final["denoised"]

    # ── Thresholds & specs ────────────────────────────────────────────────────
    per_neuron_thr = compute_per_neuron_thresholds(
        history, batch_idx=0, fraction=THR_FRACTION, target_levels=TARGET_LEVELS)
    specs = extract_full_image_interactions(
        a=a_final, batch_idx=0,
        act_threshold=per_neuron_thr, target_levels=TARGET_LEVELS)
    print(f"  Active needles: {len(specs)}")

    # ── Target neuron ─────────────────────────────────────────────────────────
    target_level, target_channel, target_y, target_x = find_target_neuron(a_final)
    print(f"  Target: L{target_level} ch{target_channel} @ (y={target_y}, x={target_x})")

    zoom = compute_zoom_geometry(denoised_final, a_final,
                                 target_level, target_y, target_x)

    # ── 1a and 1b ─────────────────────────────────────────────────────────────
    arr_1a             = make_1a_arr(specs, denoised_final, zoom)
    arr_1b, bg_crop    = make_1b_arr(specs, denoised_final, zoom)

    # ── Top-k neurons ─────────────────────────────────────────────────────────
    top_k = compute_top_k(a_final, per_neuron_thr, zoom,
                          target_level, target_y, target_x, target_channel)
    print(f"  Top-{len(top_k)} neurons: "
          + ", ".join(f"L{s['level']}ch{s['channel']}" for s in top_k))

    # ── Reconstruct inputs for SpreadContext ──────────────────────────────────
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

    ctx = SpreadContext(
        a_final=a_final, net=net, fit_by_level=fit_by_level,
        bg_crop=bg_crop,
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

    # ── Spread pairs ──────────────────────────────────────────────────────────
    spread_pairs = []
    for k, s in enumerate(top_k):
        print(f"  Spread #{k+1}/{len(top_k)} L{s['level']}ch{s['channel']} ...",
              end=" ", flush=True)
        pair = compute_spread_pair(
            s["level"], s["channel"], s["y_feat"], s["x_feat"], ctx)
        spread_pairs.append(pair)
        print("done")

    # ── Combined figure ───────────────────────────────────────────────────────
    zoom_factor = zoom["IMG_W"] / max(zoom["x1_z"] - zoom["x0_z"], 1)
    fig_combined = make_combined_figure(
        arr_1a=arr_1a, arr_1b=arr_1b,
        spread_pairs=spread_pairs,
        n_show=5, dark_mode=False,
        panel_w=3.0, panel_h=3.0,
        label_1a="Support set",
        label_1b=f"Zoom  ×{zoom_factor:.1f}",
        spread_specs=top_k,
    )

    out_path = OUT_DIR / f"gen_combined_{stem}.png"
    fig_combined.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig_combined)
    print(f"  Saved → {out_path}")

print("\nAll done.")
