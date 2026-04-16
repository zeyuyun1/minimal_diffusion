"""
needle_plot.py — Gabor fitting and needle-plot visualisation for sparse codes.

Main API
--------
fit_all_levels_gabor_bank(net, ...)
    Fit Gabor envelopes for every decoder level of a neural_sheet7 network.

plot_gabor_diagnostic_grid(net, fit_by_level, ...)
    Three-panel grid: original kernels | Gabor reconstructions | needles.

extract_full_image_interactions(a, ...)
    Scan a full latent `a` and return every active unit as a positioned spec.

render_gabor_overlay(gabor_specs, net, fit_by_level, ...)
    Draw oriented needles (and optionally reconstructed Gabor patches) over
    a background image or blank canvas.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from recurrent_diffusion_pkg.utils import (
    fit_decoder_level_gabor_bank,
    render_gabor_from_fit,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_3ch(x):
    """Convert a [C, H, W] tensor to exactly 3 channels."""
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    if x.shape[0] >= 3:
        return x[:3]
    return torch.cat([x, torch.zeros(1, x.shape[1], x.shape[2], dtype=x.dtype)], dim=0)


def _norm01_per_patch(x):
    """Normalise each patch in a [N, C, H, W] tensor to [0, 1]."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn = x.amin(dim=(-1, -2, -3), keepdim=True)
    mx = x.amax(dim=(-1, -2, -3), keepdim=True)
    denom = mx - mn
    denom[denom < 1e-8] = 1.0
    return (x - mn) / denom


def _clip_line_segment(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Liang-Barsky line clipping. Returns clipped (x0,y0,x1,y1) or None."""
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
# Public API
# ---------------------------------------------------------------------------

def fit_all_levels_gabor_bank(
    net,
    ff_threshold=1.0,
    unit_norm=True,
    use_color=True,
):
    """
    Fit Gabor envelopes for every decoder level of *net*.

    Parameters
    ----------
    net : neural_sheet7
    ff_threshold : float
        Keep only filters whose feedforward norm exceeds this value.
    unit_norm : bool
        Normalise each filter to unit L2 norm before fitting.
    use_color : bool
        Use rank-1 SVD colour decomposition for the spatial fit.

    Returns
    -------
    fit_by_level : dict[int -> fit_bank]
        fit_bank has keys ``"results"``, ``"kept_idx"``, ``"ff_norms"``.
    """
    fit_by_level = {}
    print(f"Starting multi-level Gabor fit (ff_threshold={ff_threshold})...")

    for level_idx in range(net.n_levels):
        fit_bank = fit_decoder_level_gabor_bank(
            model_net=net,
            level_idx=level_idx,
            ff_threshold=ff_threshold,
            unit_norm=unit_norm,
            use_color=use_color,
        )
        fit_by_level[level_idx] = fit_bank

        n_total = len(fit_bank["ff_norms"])
        n_kept = len(fit_bank["kept_idx"])
        successful = sum(int(r.get("successful_fit", False)) for r in fit_bank["results"])
        contrast = [r.get("color_contrast_norm", float("nan")) for r in fit_bank["results"]]
        contrast_mean = float(np.nanmean(contrast)) if contrast else float("nan")

        print(
            f"  L{level_idx}: kept {n_kept}/{n_total} (threshold={ff_threshold}); "
            f"successful_fit={successful}/{n_kept}; "
            f"mean_color_contrast={contrast_mean:.4f}"
        )

    print("Fitting complete.")
    return fit_by_level


def drop_top_norm_filters(fit_by_level, n_drop=2):
    """
    Remove the n_drop highest ff_norm filters per level from fit_by_level.

    These are typically isotropic (Gaussian) DC filters that fire everywhere
    and clutter the needle plot with spurious same-orientation needles.
    Modifies fit_by_level in-place and returns it.
    """
    for lvl, fit_bank in fit_by_level.items():
        results = fit_bank["results"]
        if not results:
            continue
        # Sort by ff_norm descending, get the filter_idx of the top n_drop
        sorted_by_norm = sorted(results, key=lambda r: r.get("ff_norm", 0.0), reverse=True)
        drop_idx = {int(r["filter_idx"]) for r in sorted_by_norm[:n_drop]}
        kept = [r for r in results if int(r["filter_idx"]) not in drop_idx]
        fit_bank["results"] = kept
        norms_str = [f"{r.get('ff_norm', 0):.3f}" for r in sorted_by_norm[:n_drop]]
        print(f"  L{lvl}: dropped top-{n_drop} norm filters (idx={sorted(drop_idx)}, ff_norms={norms_str})")
    return fit_by_level


def plot_gabor_diagnostic_grid(
    net,
    fit_by_level,
    n_show_per_level=32,
    render_scale=4,
    ncol=16,
    grid_padding=2,
    needle_length_multiplier=1.8,
    needle_width_multiplier=0.6,
    use_color=True,
    save_path=None,
):
    """
    Three-panel grid: original kernels | Gabor reconstructions | needles.

    Parameters
    ----------
    net : neural_sheet7
    fit_by_level : dict  (output of fit_all_levels_gabor_bank)
    n_show_per_level : int  filters to show per level
    render_scale : int  upsample factor for display
    ncol : int  columns in the grid
    grid_padding : int  pixels between patches
    needle_length_multiplier : float  needle half-length = sigma_y * this
    needle_width_multiplier : float  linewidth = sigma_x * scale * this
    use_color : bool
    save_path : str or Path or None  if given, save instead of showing
    """
    base_kernel_size = net.levels[0].decoder.conv.weight.shape[-1]
    target_size = base_kernel_size * (2 ** (net.n_levels - 1))
    final_target_size = target_size * render_scale

    orig_list, rec_list, needle_data_list = [], [], []

    for level_idx in range(net.n_levels):
        conv = net.levels[level_idx].decoder.conv
        dictionary = (
            conv.weight_v.detach().cpu().float()
            if hasattr(conv, "weight_v")
            else conv.weight.detach().cpu().float()
        )
        _, _, kh, kw = dictionary.shape

        fit_bank = fit_by_level[level_idx]
        fit_map = {
            int(r["filter_idx"]): r
            for r in fit_bank["results"]
            if r.get("successful_fit", False)
        }
        show_idx = [
            int(i) for i in fit_bank["kept_idx"].tolist() if int(i) in fit_map
        ][:n_show_per_level]

        scale = (2 ** level_idx) * render_scale

        for fi in show_idx:
            k0 = dictionary[fi]
            fit = fit_map[fi]

            # Build and clip native needle
            x0, y0 = fit["mean"]
            theta = fit["theta"]
            sigma_x = fit.get("sigma_x", 2.0)
            sigma_y = fit.get("sigma_y", 2.0)
            native_length = sigma_y * needle_length_multiplier
            dx_val = native_length * np.cos(theta)
            dy_val = native_length * np.sin(theta)
            clip_res = _clip_line_segment(
                x0 - dx_val, y0 - dy_val,
                x0 + dx_val, y0 + dy_val,
                xmin=-0.5, xmax=kw - 0.5,
                ymin=-0.5, ymax=kh - 0.5,
            )
            if clip_res is not None:
                c_x0, c_y0, c_x1, c_y1 = clip_res
                needle_data_list.append({
                    "x_start": c_x0, "y_start": c_y0,
                    "x_end": c_x1,   "y_end": c_y1,
                    "scale": scale, "sigma_x": sigma_x,
                    "kw": kw, "kh": kh,
                })
            else:
                needle_data_list.append(None)

            # Build image patches
            rec_np = render_gabor_from_fit(fit, out_shape=(kh, kw), use_color=use_color)
            kr = torch.from_numpy(rec_np).float()
            if (kr * k0).sum() < 0:
                kr = -kr

            if scale > 1:
                k0 = F.interpolate(k0.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)
                kr = F.interpolate(kr.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)

            current_size = k0.shape[-1]
            if current_size < final_target_size:
                pad = final_target_size - current_size
                p0, p1 = pad // 2, pad - pad // 2
                k0 = F.pad(k0, (p0, p1, p0, p1))
                kr = F.pad(kr, (p0, p1, p0, p1))

            orig_list.append(_to_3ch(k0))
            rec_list.append(_to_3ch(kr))

    orig_tensor = _norm01_per_patch(torch.stack(orig_list))
    rec_tensor  = _norm01_per_patch(torch.stack(rec_list))
    g1 = make_grid(orig_tensor, nrow=ncol, padding=grid_padding, pad_value=1.0)
    g2 = make_grid(rec_tensor,  nrow=ncol, padding=grid_padding, pad_value=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    axes[0].imshow(g1.permute(1, 2, 0).numpy(), interpolation="nearest")
    axes[0].set_title("Original Kernels", fontsize=18)
    axes[0].axis("off")

    axes[1].imshow(g2.permute(1, 2, 0).numpy(), interpolation="nearest")
    axes[1].set_title("Faithful Gabor Fits", fontsize=18)
    axes[1].axis("off")

    axes[2].imshow(g2.permute(1, 2, 0).numpy(), interpolation="nearest")
    axes[2].set_title("Gabor Fits + Scaled/Clipped Needles", fontsize=18)
    axes[2].axis("off")

    for i, ndata in enumerate(needle_data_list):
        if ndata is None:
            continue
        r, c = divmod(i, ncol)
        patch_left = grid_padding + c * (final_target_size + grid_padding)
        patch_top  = grid_padding + r * (final_target_size + grid_padding)
        scale = ndata["scale"]
        pad_x = (final_target_size - ndata["kw"] * scale) // 2
        pad_y = (final_target_size - ndata["kh"] * scale) // 2

        def _map(coord, pad, offset):
            return offset + pad + (coord + 0.5) * scale - 0.5

        axes[2].plot(
            [_map(ndata["x_start"], pad_x, patch_left),
             _map(ndata["x_end"],   pad_x, patch_left)],
            [_map(ndata["y_start"], pad_y, patch_top),
             _map(ndata["y_end"],   pad_y, patch_top)],
            color="red",
            linewidth=max(1.5, ndata["sigma_x"] * scale * needle_width_multiplier),
            solid_capstyle="round",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def extract_full_image_interactions(a, batch_idx=0, act_threshold=0.1, target_levels=None):
    """
    Scan a full latent `a` and return every active unit as a positioned spec.

    Each spec is a dict with keys:
        level, channel, weight, pixel_dx, pixel_dy

    where (pixel_dx, pixel_dy) is the displacement from the image centre in
    *image* pixels (so the same coordinate system as the raw image).

    Parameters
    ----------
    a : list of Tensor  [B, C, H, W] per level, as returned by net forward
    batch_idx : int
    act_threshold : float or dict[(level, channel) -> float]
        Scalar applies the same threshold to every neuron.
        Dict (from compute_per_neuron_thresholds) applies a per-neuron threshold,
        which lets the visualization show the full sparsification trajectory —
        many neurons active early (noise-driven) converging to few late (contour).
    target_levels : list[int] or None  restrict to these levels

    Returns
    -------
    list of spec dicts, sorted by ascending |weight| (weakest drawn first)
    """
    interactions = []
    per_neuron = isinstance(act_threshold, dict)

    for lvl, a_level in enumerate(a):
        if a_level is None:
            continue
        if target_levels is not None and lvl not in target_levels:
            continue

        latent = a_level[batch_idx]  # [C, H, W]
        _, h, w = latent.shape
        center_y, center_x = h / 2.0, w / 2.0
        stride = 2 ** (lvl + 1)

        if per_neuron:
            # Build a per-channel threshold tensor, then threshold all at once
            thr_vec = latent.new_tensor([
                act_threshold.get((lvl, c), 0.0) for c in range(latent.shape[0])
            ])                                           # [C]
            mask = latent.abs() > thr_vec[:, None, None]
        else:
            mask = latent.abs() > act_threshold

        active = torch.nonzero(mask, as_tuple=False)
        for idx in active:
            c, y, x = idx[0].item(), idx[1].item(), idx[2].item()
            val = latent[c, y, x].item()
            interactions.append({
                "level":    lvl,
                "channel":  c,
                "weight":   val,
                "pixel_dx": ((x + 0.5) - center_x) * stride,
                "pixel_dy": ((y + 0.5) - center_y) * stride,
            })

    interactions.sort(key=lambda s: abs(s["weight"]))
    return interactions


def compute_per_neuron_thresholds(history, batch_idx=0, fraction=0.3, target_levels=None):
    """
    Compute a per-neuron activation threshold from a denoising history.

    For each neuron (level, channel), the threshold is set to::

        fraction × max(|activation|)  across ALL history steps and spatial positions

    This makes the threshold adapt to each neuron's dynamic range so that:
    - Early noisy steps show many active neurons (noise drives broad activation)
    - Later steps sparsify onto the contour (only strong, selective activations remain)

    Parameters
    ----------
    history : list of snap dicts  (from net(..., return_history=True))
    batch_idx : int  which image in the batch
    fraction : float  threshold = fraction × per-neuron max  (0.2–0.4 works well)
    target_levels : list[int] or None

    Returns
    -------
    dict[(level, channel) -> float]  — pass directly to extract_full_image_interactions
    """
    max_act = {}   # (lvl, c) -> float

    for snap in history:
        for lvl, a_level in enumerate(snap["a"]):
            if a_level is None:
                continue
            if target_levels is not None and lvl not in target_levels:
                continue
            # a_level: [B, C, H, W]  — collapse spatial and batch dims
            per_channel_max = a_level[batch_idx].abs().flatten(1).max(dim=1).values  # [C]
            for c, val in enumerate(per_channel_max.tolist()):
                key = (lvl, c)
                if val > max_act.get(key, 0.0):
                    max_act[key] = val

    return {k: v * fraction for k, v in max_act.items()}


def render_gabor_overlay(
    gabor_specs,
    net,
    fit_by_level,
    render_scale=4,
    needle_length_multiplier=1.8,
    needle_width_multiplier=0.6,
    use_color=True,
    dark_mode=False,
    intensity_multiplier=1.0,
    background_image=None,
    save_path=None,
    return_fig=False,
    normalize_per_level=True,
    level_max_act_override=None,
):
    """
    Overlay oriented needle glyphs (and optionally Gabor patches) on a canvas.

    Parameters
    ----------
    gabor_specs : list of spec dicts  (output of extract_full_image_interactions,
                  or hand-crafted with keys level/channel/weight/pixel_dx/pixel_dy)
    net : neural_sheet7
    fit_by_level : dict  (output of fit_all_levels_gabor_bank)
    render_scale : int  upsample factor
    needle_length_multiplier : float
    needle_width_multiplier : float
    use_color : bool
    dark_mode : bool  black background, neon needles, intensity-modulated alpha
    intensity_multiplier : float  scales needle alpha in dark_mode
    background_image : Tensor [C, H, W] or None
        If given, use this as the canvas instead of compositing Gabor patches.
    save_path : str or Path or None
    normalize_per_level : bool  (default True)
        If True, needle alpha is normalised by the per-level max |activation|
        instead of the per-filter g_pp.  Equalises visibility across levels
        whose absolute activation magnitudes differ systematically.
        If False, the original g_pp normalisation is used.
    level_max_act_override : dict or None
        If provided, used as the per-level max instead of computing from
        gabor_specs.  Useful when rendering single specs in a loop and you
        want a shared normalisation reference (e.g. top-K subplots).
    """
    # Pre-compute per-level max |activation| for optional level normalisation.
    level_max_act: dict = {}
    if level_max_act_override is not None:
        level_max_act = level_max_act_override
    elif normalize_per_level:
        for _spec in gabor_specs:
            _lvl = _spec["level"]
            _w   = abs(_spec.get("weight", 1.0))
            if _w > level_max_act.get(_lvl, 0.0):
                level_max_act[_lvl] = _w
    if background_image is not None:
        bg = background_image.detach().cpu().float()
        if bg.shape[0] == 1:
            bg = bg.repeat(3, 1, 1)
        canvas = F.interpolate(bg.unsqueeze(0), scale_factor=render_scale, mode="nearest").squeeze(0)
        canvas = _norm01_per_patch(canvas.unsqueeze(0)).squeeze(0)
        if dark_mode:
            canvas = canvas * 0.3
        _, h_c, w_c = canvas.shape
        center_y, center_x = h_c // 2, w_c // 2
        draw_patches = False
    else:
        base_kernel_size = net.levels[0].decoder.conv.weight.shape[-1]
        canvas_size = int(base_kernel_size * (2 ** (net.n_levels - 1)) * render_scale * 4)
        canvas = torch.zeros(3, canvas_size, canvas_size)
        center_y, center_x = canvas_size // 2, canvas_size // 2
        draw_patches = True

    needle_draw_data = []

    for spec in gabor_specs:
        level_idx = spec["level"]
        fi        = spec["channel"]
        weight    = spec.get("weight", 1.0)
        shift_x   = int(spec["pixel_dx"] * render_scale)
        shift_y   = int(spec["pixel_dy"] * render_scale)

        conv = net.levels[level_idx].decoder.conv
        dictionary = (
            conv.weight_v.detach().cpu().float()
            if hasattr(conv, "weight_v")
            else conv.weight.detach().cpu().float()
        )
        _, _, kh, kw = dictionary.shape

        fit = next(
            (r for r in fit_by_level[level_idx]["results"]
             if int(r["filter_idx"]) == fi and r.get("successful_fit", False)),
            None,
        )
        if fit is None:
            continue

        k0      = dictionary[fi]
        ff_norm = float(fit.get("ff_norm", 1.0))   # weight_g scale (1.0 if no weight norm)
        g_pp    = (k0.max() - k0.min()).item() * ff_norm   # true peak-to-peak of the effective filter
        scale   = (2 ** level_idx) * render_scale

        # Build and clip needle
        x0, y0   = fit["mean"]
        theta    = fit["theta"]
        sigma_x  = fit.get("sigma_x", 2.0)
        sigma_y  = fit.get("sigma_y", 2.0)
        length   = sigma_y * needle_length_multiplier
        clip_res = _clip_line_segment(
            x0 - length * np.cos(theta), y0 - length * np.sin(theta),
            x0 + length * np.cos(theta), y0 + length * np.sin(theta),
            xmin=-0.5, xmax=kw - 0.5,
            ymin=-0.5, ymax=kh - 0.5,
        )

        kh_s, kw_s = int(kh * scale), int(kw * scale)
        patch_top  = center_y - kh_s // 2 + shift_y
        patch_left = center_x - kw_s // 2 + shift_x

        if draw_patches:
            rec_np = render_gabor_from_fit(fit, out_shape=(kh, kw), use_color=use_color)
            kr = torch.from_numpy(rec_np).float() * abs(weight)
            if weight > 0 and (kr * k0).sum() < 0:
                kr = -kr
            if weight < 0 and (kr * k0).sum() > 0:
                kr = -kr
            if scale > 1:
                kr = F.interpolate(kr.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)
            kr = _to_3ch(kr)

            ct = max(0, patch_top);  cb = min(canvas.shape[1], patch_top  + kh_s)
            cl = max(0, patch_left); cr = min(canvas.shape[2], patch_left + kw_s)
            kt = ct - patch_top;     kb = kh_s - ((patch_top  + kh_s) - cb)
            kl = cl - patch_left;    kr_r = kw_s - ((patch_left + kw_s) - cr)
            if ct < cb and cl < cr:
                canvas[:, ct:cb, cl:cr] += kr[:, kt:kb, kl:kr_r]

        if clip_res is not None:
            c_x0, c_y0, c_x1, c_y1 = clip_res
            needle_draw_data.append({
                "x_start": c_x0, "y_start": c_y0,
                "x_end":   c_x1, "y_end":   c_y1,
                "scale": scale, "sigma_x": sigma_x,
                "patch_top": patch_top, "patch_left": patch_left,
                "weight": weight, "g_pp": g_pp,
                "level_idx": level_idx,
            })

    if draw_patches:
        canvas = _norm01_per_patch(canvas.unsqueeze(0)).squeeze(0)

    facecolor = "black" if dark_mode else "white"
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=facecolor)
    ax.imshow(canvas.permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.axis("off")

    for ndata in needle_draw_data:
        s = ndata["scale"]

        def _map(coord, offset, _s=s):   # capture s now, not at call time
            return offset + (coord + 0.5) * _s - 0.5

        lw      = max(1.5, ndata["sigma_x"] * s * needle_width_multiplier)
        g_pp    = ndata["g_pp"]
        # Needle alpha: either per-level normalisation (equalises brightness
        # across levels with different absolute activation scales) or the
        # original per-filter g_pp normalisation.
        if normalize_per_level:
            denom = level_max_act.get(ndata["level_idx"], 1.0) + 1e-8
        else:
            denom = g_pp + 1e-8
        norm_act = abs(ndata["weight"]) / denom
        if dark_mode:
            alpha = min(1.0, norm_act * intensity_multiplier)
            color = "#ff3333" if ndata["weight"] >= 0 else "#33ccff"
        else:
            alpha = min(1.0, norm_act * intensity_multiplier)
            color = "red" if ndata["weight"] >= 0 else "blue"

        ax.plot(
            [_map(ndata["x_start"], ndata["patch_left"]),
             _map(ndata["x_end"],   ndata["patch_left"])],
            [_map(ndata["y_start"], ndata["patch_top"]),
             _map(ndata["y_end"],   ndata["patch_top"])],
            color=color, alpha=alpha, linewidth=lw, solid_capstyle="round",
        )

    title_color = "white" if dark_mode else "black"
    ax.set_title("Gabor Interaction Overlay", fontsize=16, color=title_color)

    if draw_patches:
        non_zero = torch.nonzero(canvas.sum(dim=0))
        if len(non_zero) > 0:
            min_y, min_x = non_zero[:, 0].min().item(), non_zero[:, 1].min().item()
            max_y, max_x = non_zero[:, 0].max().item(), non_zero[:, 1].max().item()
            pad = 8 * render_scale
            ax.set_ylim(max_y + pad, min_y - pad)
            ax.set_xlim(min_x - pad, max_x + pad)

    plt.tight_layout()
    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Prior (M_inter) spread visualisation
# ---------------------------------------------------------------------------

def compute_prior_spread(net, level_idx, channel_idx, spatial_size=(33, 33)):
    """
    Inject a unit pulse at (level_idx, channel_idx, center) and propagate
    one step through M_inter.

    Parameters
    ----------
    net : neural_sheet7
    level_idx : int
    channel_idx : int
    spatial_size : (H, W)
        Size of the dummy spatial map.  Odd numbers keep the center at an
        integer index.

    Returns
    -------
    a_spread : Tensor [1, C, H, W]  or None if no M_inter at this level.
    """
    level = net.levels[level_idx]
    M_inter = level.prior_node.M_inter
    if M_inter is None:
        return None

    C = level.prior_node.num_basis
    H, W = spatial_size
    device = next(net.parameters()).device

    a_pulse = torch.zeros(1, C, H, W, device=device)
    a_pulse[0, channel_idx, H // 2, W // 2] = 1.0

    with torch.no_grad():
        a_spread = M_inter(a_pulse)

    return a_spread  # [1, C, H, W]


def render_prior_spread_panel(
    net,
    fit_by_level,
    level_idx,
    channel_idx,
    spatial_size=(33, 33),
    cell_size=90,
    n_top_per_cell=5,
    act_threshold_fraction=0.05,
    dark_mode=True,
    return_fig=False,
    save_path=None,
):
    """
    3×3 grid panel showing the one-step M_inter spread of one source neuron.

    Layout
    ------
    Each of the 9 cells corresponds to a spatial offset (dy, dx) ∈ {-1,0,1}².
    The center cell (0,0) shows the source neuron needle (yellow/green).
    All other cells show target needles colored by sign:
      - warm (red)  : excitatory connection
      - cool (blue) : inhibitory connection
    Needle alpha and thickness scale with |M_inter weight|.

    Parameters
    ----------
    net : neural_sheet7
    fit_by_level : dict  (output of fit_all_levels_gabor_bank)
    level_idx, channel_idx : int
    spatial_size : (H, W)  dummy map size for pulse injection
    cell_size : int  pixels per grid cell
    n_top_per_cell : int  max target needles drawn per cell
    act_threshold_fraction : float  skip targets below this × max(|spread|)
    dark_mode : bool
    return_fig : bool  if True, return the Figure instead of showing/saving
    save_path : str or Path or None
    """
    a_spread = compute_prior_spread(net, level_idx, channel_idx, spatial_size)
    if a_spread is None:
        return None

    H, W = spatial_size
    cy, cx = H // 2, W // 2
    max_val = a_spread.abs().max().item()
    if max_val < 1e-8:
        return None
    act_thr = act_threshold_fraction * max_val

    grid_px = 3 * cell_size
    bg_color = "black" if dark_mode else "white"
    fg_color = "white" if dark_mode else "black"

    dpi = 100
    fig_size = grid_px / dpi
    fig, ax = plt.subplots(
        figsize=(fig_size, fig_size + 0.25), dpi=dpi, facecolor=bg_color
    )
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, grid_px)
    ax.set_ylim(grid_px, 0)   # y-axis: down is positive (image convention)
    ax.axis("off")

    offsets = [
        (-1, -1), (-1,  0), (-1,  1),
        ( 0, -1), ( 0,  0), ( 0,  1),
        ( 1, -1), ( 1,  0), ( 1,  1),
    ]

    fit_map = {
        int(r["filter_idx"]): r
        for r in fit_by_level[level_idx]["results"]
        if r.get("successful_fit", False)
    }

    needle_half = cell_size * 0.38   # half-length of needle in canvas pixels

    for gi, (dy, dx) in enumerate(offsets):
        row, col = gi // 3, gi % 3
        ccx = col * cell_size + cell_size / 2   # cell centre x
        ccy = row * cell_size + cell_size / 2   # cell centre y
        is_source = (dy == 0 and dx == 0)

        if is_source:
            fit = fit_map.get(channel_idx)
            if fit:
                theta = fit["theta"]
                ax.plot(
                    [ccx - needle_half * np.cos(theta),
                     ccx + needle_half * np.cos(theta)],
                    [ccy - needle_half * np.sin(theta),
                     ccy + needle_half * np.sin(theta)],
                    color="yellow" if dark_mode else "green",
                    lw=3.5, alpha=1.0, solid_capstyle="round", zorder=5,
                )
        else:
            fy, fx = cy + dy, cx + dx
            if not (0 <= fy < H and 0 <= fx < W):
                continue
            vals = a_spread[0, :, fy, fx]   # [C]
            abs_vals = vals.abs()
            above = (abs_vals > act_thr).nonzero(as_tuple=False).squeeze(1)
            if len(above) == 0:
                continue
            # pick top-N by abs value
            top_abs, order = abs_vals[above].sort(descending=True)
            top_idx = above[order[:n_top_per_cell]]
            for i, c in enumerate(top_idx.tolist()):
                w = float(vals[c])
                fit = fit_map.get(c)
                if fit is None:
                    continue
                theta = fit["theta"]
                alpha = float(np.clip(abs(w) / max_val * 4.0, 0.15, 1.0))
                lw    = 1.5 + abs(w) / max_val * 2.5
                color = ("#ff4444" if dark_mode else "red") if w > 0 else ("#55aaff" if dark_mode else "blue")
                ax.plot(
                    [ccx - needle_half * np.cos(theta),
                     ccx + needle_half * np.cos(theta)],
                    [ccy - needle_half * np.sin(theta),
                     ccy + needle_half * np.sin(theta)],
                    color=color, lw=lw, alpha=alpha, solid_capstyle="round",
                )

    # Grid lines
    grid_lc = "#404040" if dark_mode else "#cccccc"
    for i in range(4):
        ax.axhline(i * cell_size, color=grid_lc, lw=0.8, zorder=1)
        ax.axvline(i * cell_size, color=grid_lc, lw=0.8, zorder=1)

    # Highlight source cell
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (cell_size, cell_size), cell_size, cell_size,
        fill=False,
        edgecolor="yellow" if dark_mode else "green",
        lw=1.8, zorder=4,
    ))

    ax.set_title(f"L{level_idx} ch{channel_idx}", color=fg_color, fontsize=8, pad=3)

    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=bg_color)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def render_prior_spread_grid(
    net,
    fit_by_level,
    n_top_per_level=12,
    spatial_size=(33, 33),
    cell_size=70,
    n_top_per_cell=5,
    act_threshold_fraction=0.05,
    dark_mode=True,
    save_path=None,
    return_fig=False,
):
    """
    Grid of prior-spread panels: rows = levels, cols = top-N neurons (by FF norm).

    Each panel is the 3×3 M_inter spread for one source neuron.  Neurons are
    ranked by their feedforward decoder norm (most selective first).

    Parameters
    ----------
    net : neural_sheet7
    fit_by_level : dict  (output of fit_all_levels_gabor_bank)
    n_top_per_level : int  neurons per row
    spatial_size : (H, W)  dummy map for pulse injection
    cell_size : int  pixels per 3×3 cell
    n_top_per_cell : int  max target needles per offset cell
    act_threshold_fraction : float
    dark_mode : bool
    save_path : str or Path or None
    return_fig : bool
    """
    import io
    from PIL import Image as _PILImage

    bg_color = "black" if dark_mode else "white"
    fg_color = "white" if dark_mode else "black"
    panel_px  = 3 * cell_size   # each panel is (panel_px × panel_px) canvas pixels
    n_levels  = net.n_levels

    # Collect top-N fitted channels per level
    level_channels = {}
    for lvl in range(n_levels):
        if net.levels[lvl].prior_node.M_inter is None:
            level_channels[lvl] = []
            continue
        fit_bank = fit_by_level[lvl]
        fitted_set = {int(r["filter_idx"]) for r in fit_bank["results"] if r.get("successful_fit", False)}
        kept = [int(i) for i in fit_bank["kept_idx"].tolist() if int(i) in fitted_set]
        level_channels[lvl] = kept[:n_top_per_level]

    n_cols = max((len(v) for v in level_channels.values()), default=0)
    if n_cols == 0:
        print("No M_inter found at any level.")
        return None

    dpi       = 100
    fig_w_in  = n_cols * panel_px / dpi
    fig_h_in  = n_levels * (panel_px / dpi + 0.25)

    fig, axes = plt.subplots(
        n_levels, n_cols,
        figsize=(fig_w_in, fig_h_in),
        facecolor=bg_color,
        squeeze=False,
    )

    for lvl in range(n_levels):
        channels = level_channels[lvl]
        for col in range(n_cols):
            ax = axes[lvl][col]
            ax.set_facecolor(bg_color)
            ax.axis("off")
            if col >= len(channels):
                continue
            ch = channels[col]
            panel_fig = render_prior_spread_panel(
                net, fit_by_level, lvl, ch,
                spatial_size=spatial_size,
                cell_size=cell_size,
                n_top_per_cell=n_top_per_cell,
                act_threshold_fraction=act_threshold_fraction,
                dark_mode=dark_mode,
                return_fig=True,
            )
            if panel_fig is None:
                continue
            buf = io.BytesIO()
            panel_fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                              facecolor=bg_color)
            buf.seek(0)
            img = np.array(_PILImage.open(buf).convert("RGB"))
            plt.close(panel_fig)
            ax.imshow(img, interpolation="nearest")

        # Row label
        axes[lvl][0].set_ylabel(f"L{lvl}", color=fg_color, fontsize=9,
                                 rotation=0, labelpad=18, va="center")

    plt.suptitle(
        "M_inter prior spread — one step per source neuron\n"
        "yellow = source | red = excitatory | blue = inhibitory",
        color=fg_color, fontsize=10, y=1.01,
    )
    plt.tight_layout(pad=0.4)

    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=bg_color)
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()
