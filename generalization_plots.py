"""
generalization_plots.py
Reusable plotting helpers for generalization experiments.
Import this in generalization.ipynb (or any script) instead of defining
everything inline.

Public API
----------
fig_to_arr(fig, dpi)
mark_source_on_canvas(ax, fit_r, canvas_cx, canvas_cy, render_scale)
mark_source_on_heatmap(ax, fit_r, hmap_cx, hmap_cy, stride)
SpreadContext          – dataclass holding per-experiment shared state
compute_spread_pair(src_level, src_channel, src_y, src_x, ctx)
make_combined_figure(arr_1a, arr_1b, spread_pairs, ...)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
from torch.func import jvp


# ── Low-level helpers ─────────────────────────────────────────────────────────

def fig_to_arr(fig: plt.Figure, dpi: int = 150) -> np.ndarray:
    """Render a matplotlib Figure to an H×W×3 uint8 numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def mark_source_on_canvas(
    ax,
    fit_r: dict,
    canvas_cx: float,
    canvas_cy: float,
    render_scale: int = 4,
) -> None:
    """Draw a gold glowing needle on a render_gabor_overlay canvas axes."""
    theta   = fit_r["theta"]
    sigma_x = fit_r.get("sigma_x", 2.0)
    half_len = sigma_x * render_scale * 3.5
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = cos_t * half_len, sin_t * half_len

    ax.plot([canvas_cx - dx, canvas_cx + dx],
            [canvas_cy - dy, canvas_cy + dy],
            color="gold", linewidth=7, alpha=0.30,
            solid_capstyle="round", zorder=10)
    ax.plot([canvas_cx - dx, canvas_cx + dx],
            [canvas_cy - dy, canvas_cy + dy],
            color="gold", linewidth=2.5, alpha=1.0,
            solid_capstyle="round", zorder=11)
    ax.plot(canvas_cx, canvas_cy, "o", markersize=6, zorder=12,
            markerfacecolor="gold", markeredgecolor="white",
            markeredgewidth=0.8)


def mark_source_on_heatmap(
    ax,
    fit_r: dict,
    hmap_cx: float,
    hmap_cy: float,
    stride: int,
) -> None:
    """Draw the gold source needle on a pixel-heatmap axes (image-pixel coords)."""
    theta   = fit_r["theta"]
    sigma_x = fit_r.get("sigma_x", 2.0)
    half_len = sigma_x * stride * 0.9
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = cos_t * half_len, sin_t * half_len

    ax.plot([hmap_cx - dx, hmap_cx + dx],
            [hmap_cy - dy, hmap_cy + dy],
            color="gold", linewidth=4, alpha=0.30,
            solid_capstyle="round", zorder=10)
    ax.plot([hmap_cx - dx, hmap_cx + dx],
            [hmap_cy - dy, hmap_cy + dy],
            color="gold", linewidth=1.5, alpha=1.0,
            solid_capstyle="round", zorder=11)
    ax.plot(hmap_cx, hmap_cy, "o", markersize=4, zorder=12,
            markerfacecolor="gold", markeredgecolor="white",
            markeredgewidth=0.6)


# ── Spread context ────────────────────────────────────────────────────────────

@dataclass
class SpreadContext:
    """
    All the shared state needed to compute and render JVP spread panels.
    Build once per experiment run and pass to compute_spread_pair().
    """
    # Model outputs
    a_final:       list          # list of [1, C, H, W] tensors per level
    net:           object
    fit_by_level:  dict

    # Cropped background images (1, H_crop, W_crop) float tensor
    bg_crop:       torch.Tensor          # denoised crop (dark rendering BG)

    # Source / target neuron info
    target_level:   int
    target_channel: int
    tgt_pdx:        float        # pixel_dx of target in full-image coords
    tgt_pdy:        float

    # Zoom crop geometry (full-image pixel coords)
    x0_z: int;  x1_z: int
    y0_z: int;  y1_z: int
    dx_off: float                # crop-centre shift (x)
    dy_off: float                # crop-centre shift (y)
    IMG_W: int;  IMG_H: int

    # Misc
    TARGET_LEVELS:  list
    actual_res_in:  torch.Tensor
    noise_emb_ls:   list

    # Render settings
    nw_mult:    float = 0.8
    thr_spread: float = 0.025
    dark_mode:  bool  = True
    render_scale: int = 4

    # Optional clean input crop for the perturbed-image panel
    clean_crop: object = None   # torch.Tensor (1, H_crop, W_crop) or None

    # Lazy import to avoid circular deps
    _render_gabor_overlay: object = field(default=None, repr=False)

    def _one_inter_step(self, a_temp, upstream_temp, decoded_temp):
        self.net.forward_dynamics(
            a_temp, self.actual_res_in, upstream_temp, decoded_temp,
            self.noise_emb_ls, self.net.T, None, "spatial_mean", reverse=False,
        )
        self.net.forward_dynamics(
            a_temp, self.actual_res_in, upstream_temp, decoded_temp,
            self.noise_emb_ls, self.net.T, None, "spatial_mean", reverse=True,
        )

    def _spread_combined(self, *a_current):
        a_temp        = list(a_current)
        upstream_temp = [None] * self.net.n_levels
        decoded_temp  = [None] * self.net.n_levels
        self._one_inter_step(a_temp, upstream_temp, decoded_temp)
        pixel   = self.net.decoder_0(decoded_temp)
        neurons = tuple(
            a_t if a_t is not None else torch.zeros(1, device=a_current[0].device)
            for a_t in a_temp
        )
        return (pixel,) + neurons


# ── compute_spread_pair ───────────────────────────────────────────────────────

def compute_spread_pair(
    src_level:   int,
    src_channel: int,
    src_y:       int,
    src_x:       int,
    ctx:         SpreadContext,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JVP from one impulse neuron → (gabor_arr, heatmap_arr), both
    cropped to the zoom window.

    Parameters
    ----------
    src_level, src_channel, src_y, src_x : source neuron indices
    ctx : SpreadContext

    Returns
    -------
    gabor_arr    : H×W×3 uint8 — needle overlay on bg_crop
    heatmap_arr  : H×W×3 uint8 — pixel-spread heatmap, zoom crop
    perturbed_arr: H×W×3 uint8 — clean_crop + scaled pixel spread
    """
    from recurrent_diffusion_pkg.needle_plot import render_gabor_overlay

    e_ls = [torch.zeros_like(a) for a in ctx.a_final]
    e_ls[src_level][0, src_channel, src_y, src_x] = 1.0

    _, spread_tuple = jvp(ctx._spread_combined, tuple(ctx.a_final), tuple(e_ls))
    pixel_spread  = spread_tuple[0]
    neuron_spread = list(spread_tuple[1:])

    # ── Pixel heatmap ─────────────────────────────────────────────────────────
    ps      = pixel_spread[0, 0].detach().cpu().float().numpy()
    ps_norm = ps / (np.abs(ps).max() + 1e-8)
    ps_crop = ps_norm[ctx.y0_z:ctx.y1_z, ctx.x0_z:ctx.x1_z]

    fig_h, ax_h = plt.subplots(1, 1, figsize=(3, 3), facecolor="white")
    ax_h.imshow(ps_crop, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_h.axis("off")

    # Source needle on heatmap
    hmap_cx = ctx.IMG_W / 2.0 + ctx.tgt_pdx - ctx.x0_z
    hmap_cy = ctx.IMG_H / 2.0 + ctx.tgt_pdy - ctx.y0_z
    fit_r = next(
        (r for r in ctx.fit_by_level[src_level]["results"]
         if int(r["filter_idx"]) == src_channel and r.get("successful_fit")),
        None,
    )
    if fit_r is not None:
        mark_source_on_heatmap(ax_h, fit_r, hmap_cx, hmap_cy,
                               stride=2 ** (src_level + 1))
    heatmap_arr = fig_to_arr(fig_h, dpi=100)
    plt.close(fig_h)

    # ── Gabor needle overlay ──────────────────────────────────────────────────
    all_abs    = torch.cat([ns.abs().flatten()
                            for ns in neuron_spread if ns.numel() > 1])
    spread_thr = ctx.thr_spread * all_abs.max().item()

    spread_specs_full = []
    for lvl, ns in enumerate(neuron_spread):
        if ns.numel() <= 1 or lvl not in ctx.TARGET_LEVELS:
            continue
        fitted = {int(r["filter_idx"]) for r in ctx.fit_by_level[lvl]["results"]
                  if r.get("successful_fit", False)}
        ns_vol = ns[0]
        _, h, w = ns_vol.shape
        stride  = 2 ** (lvl + 1)
        for idx in torch.nonzero(ns_vol.abs() > spread_thr, as_tuple=False):
            ch, y, x = idx[0].item(), idx[1].item(), idx[2].item()
            if ch not in fitted:
                continue
            spread_specs_full.append({
                "level": lvl, "channel": ch,
                "weight": ns_vol[ch, y, x].item(),
                "pixel_dx": ((x + 0.5) - w / 2.0) * stride,
                "pixel_dy": ((y + 0.5) - h / 2.0) * stride,
            })
    spread_specs_full.sort(key=lambda s: abs(s["weight"]))

    specs_zoom = [
        {**s,
         "pixel_dx": s["pixel_dx"] - ctx.dx_off,
         "pixel_dy": s["pixel_dy"] - ctx.dy_off}
        for s in spread_specs_full
        if (ctx.x0_z <= ctx.IMG_W / 2.0 + s["pixel_dx"] < ctx.x1_z and
            ctx.y0_z <= ctx.IMG_H / 2.0 + s["pixel_dy"] < ctx.y1_z)
    ]

    fig_g = render_gabor_overlay(
        gabor_specs=specs_zoom,
        net=ctx.net,
        fit_by_level=ctx.fit_by_level,
        background_image=ctx.bg_crop,
        dark_mode=ctx.dark_mode,
        intensity_multiplier=3.0,
        needle_width_multiplier=ctx.nw_mult * 3.0,
        return_fig=True,
    )
    # Clip to canvas so bbox doesn't grow with out-of-frame needles
    ax_g = fig_g.axes[0]
    ax_g.set_xlim(0, ctx.bg_crop.shape[-1] * ctx.render_scale)
    ax_g.set_ylim(ctx.bg_crop.shape[-2] * ctx.render_scale, 0)

    # Source needle in gold
    pdx_crop = ctx.tgt_pdx - ctx.dx_off
    pdy_crop = ctx.tgt_pdy - ctx.dy_off
    canvas_cx = ctx.bg_crop.shape[-1] * ctx.render_scale / 2.0 + pdx_crop * ctx.render_scale
    canvas_cy = ctx.bg_crop.shape[-2] * ctx.render_scale / 2.0 + pdy_crop * ctx.render_scale
    if fit_r is not None:
        mark_source_on_canvas(ax_g, fit_r, canvas_cx, canvas_cy, ctx.render_scale)

    gabor_arr = fig_to_arr(fig_g, dpi=100)
    plt.close(fig_g)

    # ── Perturbed image: clean_crop + scaled pixel spread ─────────────────────
    if ctx.clean_crop is not None:
        clean_np = ctx.clean_crop[0].detach().cpu().float().numpy()   # [H, W]
        # Normalise clean to [0, 1]
        lo, hi = clean_np.min(), clean_np.max()
        clean_01 = (clean_np - lo) / max(hi - lo, 1e-8)
        # Add spread with a fixed alpha so perturbation is clearly visible
        ALPHA = 1.0
        perturbed_01 = np.clip(clean_01 + ALPHA * ps_crop, 0.0, 1.0)
        perturbed_u8 = (perturbed_01 * 255).astype(np.uint8)
        perturbed_rgb = np.stack([perturbed_u8] * 3, axis=-1)
    else:
        # Fallback: just show the spread on its own
        ps_u8 = ((ps_crop * 0.5 + 0.5) * 255).astype(np.uint8)
        perturbed_rgb = np.stack([ps_u8] * 3, axis=-1)

    return gabor_arr, heatmap_arr, perturbed_rgb


# ── make_combined_figure ──────────────────────────────────────────────────────

def make_combined_figure(
    arr_1a:        np.ndarray,
    arr_1b:        np.ndarray,
    spread_pairs:  List[Tuple[np.ndarray, np.ndarray]],
    n_show:        int   = 5,
    dark_mode:     bool  = False,
    panel_w:       float = 3.0,
    panel_h:       float = 3.0,
    label_1a:      str   = "Support set",
    label_1b:      str   = "Zoom",
    spread_specs:  Optional[list] = None,
    arr_clean:      Optional[np.ndarray] = None,
    arr_noisy:      Optional[np.ndarray] = None,
    arr_denoised:   Optional[np.ndarray] = None,
    arr_neuron_zoom: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Layout (when arr_clean/noisy/denoised are supplied):

        ┌───────┬───────┬─────────┬──────────┬────────┐
        │ clean │ noisy │ denoised│   1a     │  1b    │  header row
        ├───────┴───────┴─────────┴──────────┴────────┤
        │  g1 h1  │  g2 h2  │  g3 h3  │               spread rows
        │  g4 h4  │  g5 h5  │   ...   │
        └─────────┴─────────┴─────────┘

    Without arr_clean/noisy/denoised the header row is omitted and the old
    two-row layout (1a/1b stacked in left col) is used instead.
    """
    bg = "black" if dark_mode else "white"
    fg = "white" if dark_mode else "black"

    n_spread = 9   # always 3×3 grid
    n_pairs  = 3   # spread triples per row

    has_header = (arr_clean is not None)

    # Each spread slot now has 3 sub-panels: gabor / heatmap / perturbed
    n_sub = 3

    if has_header:
        # ── New layout ────────────────────────────────────────────────────────
        # Spread: n_pairs triples × n_sub cols = 9 cols
        # Header: 6 panels × 1 col each (cols 0-5), cols 6-8 blank
        n_spread_rows = n_spread // n_pairs   # 2 or 3 depending on n_spread
        n_spread_cols = n_pairs * n_sub
        n_rows = 1 + n_spread_rows
        fig_w  = panel_w * n_spread_cols
        fig_h  = panel_h * n_rows
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg)
        fig.patch.set_facecolor(bg)

        gs = GridSpec(n_rows, n_spread_cols, figure=fig, wspace=0.03, hspace=0.08)

        header = [
            (arr_clean,       "clean"),
            (arr_noisy,       "noisy"),
            (arr_denoised,    "denoised"),
            (arr_1a,          label_1a),
            (arr_1b,          label_1b),
            (arr_neuron_zoom, "neuron of interest"),
        ]
        for col, (arr, label) in enumerate(header):
            ax = fig.add_subplot(gs[0, col])
            if arr is not None:
                ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
                ax.set_title(label, fontsize=9, color=fg, pad=3)
            ax.axis("off")
            ax.set_facecolor(bg)
        for col in range(len(header), n_spread_cols):   # blank remainder
            ax = fig.add_subplot(gs[0, col])
            ax.axis("off"); ax.set_facecolor(bg)

        spread_row_offset = 1
        n_header_axes = n_spread_cols   # one ax per header col

    else:
        # ── Legacy layout: 1a/1b stacked in left col ─────────────────────────
        n_spread_rows = n_spread // n_pairs
        n_spread_cols = n_pairs * n_sub + 1   # +1 for left col
        n_rows = n_spread_rows
        fig_w  = panel_w * (2 + n_pairs * n_sub)
        fig_h  = panel_h * n_spread_rows
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg)
        fig.patch.set_facecolor(bg)

        gs = GridSpec(n_rows, n_spread_cols, figure=fig,
                      width_ratios=[2] + [1] * (n_pairs * n_sub),
                      wspace=0.03, hspace=0.08)

        for row, (arr, label) in enumerate([(arr_1a, label_1a), (arr_1b, label_1b)]):
            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(arr)
            ax.set_title(label, fontsize=9, color=fg, pad=3)
            ax.axis("off")
            ax.set_facecolor(bg)

        spread_row_offset = 0
        n_header_axes = 2

    # ── Spread panels ─────────────────────────────────────────────────────────
    col_offset = 0 if has_header else 1

    for k in range(n_spread):
        row       = k // n_pairs + spread_row_offset
        triple    = k %  n_pairs
        col_g     = triple * n_sub     + col_offset
        col_h     = triple * n_sub + 1 + col_offset
        col_p     = triple * n_sub + 2 + col_offset

        ax_g = fig.add_subplot(gs[row, col_g])
        ax_h = fig.add_subplot(gs[row, col_h])
        ax_p = fig.add_subplot(gs[row, col_p])

        if k < min(n_show, len(spread_pairs)):
            pair = spread_pairs[k]
            gabor_arr, heatmap_arr = pair[0], pair[1]
            perturbed_arr = pair[2] if len(pair) > 2 else None

            ax_g.imshow(gabor_arr);   ax_g.axis("off"); ax_g.set_facecolor(bg)
            ax_h.imshow(heatmap_arr); ax_h.axis("off"); ax_h.set_facecolor(bg)
            if perturbed_arr is not None:
                ax_p.imshow(perturbed_arr, cmap="gray" if perturbed_arr.ndim == 2 else None)
            ax_p.axis("off"); ax_p.set_facecolor(bg)

            if spread_specs is not None and k < len(spread_specs):
                s = spread_specs[k]
                ax_g.set_title(
                    f"#{k+1} L{s['level']} ch{s['channel']}  {s['weight']:+.3f}",
                    color=fg, fontsize=7, pad=2,
                )
        else:
            for ax in (ax_g, ax_h, ax_p):
                ax.axis("off"); ax.set_facecolor(bg)
            ax_g.text(0.5, 0.5, "...", ha="center", va="center",
                      fontsize=22, color=fg, fontweight="bold",
                      transform=ax_g.transAxes)

    # ── Sub-borders around each spread triple ─────────────────────────────────
    fig.canvas.draw()
    for k in range(min(n_show, len(spread_pairs))):
        ax_g = fig.axes[n_header_axes + k * n_sub]
        ax_p = fig.axes[n_header_axes + k * n_sub + n_sub - 1]
        bb_g = ax_g.get_position()
        bb_p = ax_p.get_position()
        pad  = 0.004
        rect = mpatches.FancyBboxPatch(
            (bb_g.x0 - pad, bb_g.y0 - pad),
            (bb_p.x1 - bb_g.x0) + 2 * pad,
            (bb_g.y1 - bb_g.y0) + 2 * pad,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor="#888888", facecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=20,
        )
        fig.add_artist(rect)

    return fig
