import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import math
import inspect
import json
from pathlib import Path
from tqdm.auto import tqdm
import os

def warmup_fn(step):
    warmup_steps = 500
    return min((step + 1) / warmup_steps, 1.0)


def compute_width_threshold(needle_properties, channel_indices, quantile=0.85):
    vals = []
    for ci in channel_indices:
        ci = int(ci)
        if ci not in needle_properties:
            continue
        fit = needle_properties[ci]
        sx = max(float(abs(fit.get("sigma_x", 1.0))), 1e-4)
        sy = max(float(abs(fit.get("sigma_y", 1.0))), 1e-4)
        vals.append(min(sx, sy))
    if len(vals) == 0:
        return 1.0
    return float(np.quantile(np.asarray(vals, dtype=np.float32), float(quantile)))


def draw_needle_glyph(
    ax,
    fit,
    *,
    x_line=None,
    y_line=None,
    center_xy=None,
    color="red",
    alpha=1.0,
    linewidth=1.0,
    width_threshold=1.0,
    isotropic_mode="ball",
    isotropic_alpha_scale=0.35,
    isotropic_size_scale=0.95,
    size_override=None,
):
    """Draw one needle as line or isotropic glyph on a matplotlib axis."""
    if x_line is None or y_line is None:
        x_line = np.asarray(fit["x"], dtype=np.float32)
        y_line = np.asarray(fit["y"], dtype=np.float32)
    else:
        x_line = np.asarray(x_line, dtype=np.float32)
        y_line = np.asarray(y_line, dtype=np.float32)

    sx = max(float(abs(fit.get("sigma_x", 1.0))), 1e-4)
    sy = max(float(abs(fit.get("sigma_y", 1.0))), 1e-4)
    is_iso = min(sx, sy) >= float(width_threshold)

    if is_iso and isotropic_mode in ("ball", "block"):
        if center_xy is None:
            cx = float(np.mean(x_line))
            cy = float(np.mean(y_line))
        else:
            cx, cy = float(center_xy[0]), float(center_xy[1])

        if size_override is None:
            size = float(max(0.18, float(isotropic_size_scale) * min(sx, sy)))
        else:
            size = float(max(0.08, size_override))
        alpha_iso = float(np.clip(float(alpha) * float(isotropic_alpha_scale), 0.0, 1.0))

        if isotropic_mode == "ball":
            patch = plt.Circle((cx, cy), radius=0.5 * size, color=color, alpha=alpha_iso, linewidth=0)
        else:
            patch = plt.Rectangle((cx - 0.5 * size, cy - 0.5 * size), size, size, color=color, alpha=alpha_iso, linewidth=0)
        ax.add_patch(patch)
        return True

    ax.plot(x_line, y_line, color=color, linewidth=float(linewidth), alpha=float(alpha))
    return False


def plot_feature_map_needles_matplotlib(
    a_feat,
    *,
    needle_properties,
    channel_order=None,
    allowed_channels=None,
    topk_total=1800,
    abs_threshold=None,
    glyph_scale=0.82,
    spatial_pitch=0.72,
    spatial_stride=None,
    glyph_size_stride_mix=0.35,
    width_thr_quantile=0.85,
    isotropic_mode="ball",
    isotropic_alpha_scale=0.35,
    isotropic_size_scale=0.95,
    flip_y=True,
    color="red",
    alpha_min=0.10,
    alpha_max=0.95,
    linewidth=1.0,
    figsize=(8.0, 8.0),
    title="",
):
    """Render a generic feature map onto needle glyphs using matplotlib."""
    if a_feat.dim() == 4:
        if a_feat.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {a_feat.shape[0]}")
        a = a_feat[0].detach().cpu().float()
    elif a_feat.dim() == 3:
        a = a_feat.detach().cpu().float()
    else:
        raise ValueError(f"a_feat must be [C,H,W] or [1,C,H,W], got {tuple(a_feat.shape)}")

    C, H, W = a.shape
    if channel_order is None:
        channel_order = list(range(C))

    allowed_set = None if allowed_channels is None else {int(c) for c in allowed_channels}
    valid_ch = []
    for c in channel_order:
        ci = int(c)
        if not (0 <= ci < C):
            continue
        if ci not in needle_properties:
            continue
        if allowed_set is not None and ci not in allowed_set:
            continue
        valid_ch.append(ci)
    if len(valid_ch) == 0:
        raise RuntimeError("No valid channels overlap between a_feat and needle_properties (after filtering).")

    width_thr = compute_width_threshold(needle_properties, valid_ch, quantile=width_thr_quantile)

    candidates = []
    for ch in valid_ch:
        v = a[ch]
        if abs_threshold is not None:
            ys, xs = torch.where(v.abs() >= float(abs_threshold))
        else:
            ys, xs = torch.where(torch.ones_like(v, dtype=torch.bool))
        vals = v[ys, xs]
        for yy, xx, vv in zip(ys.tolist(), xs.tolist(), vals.tolist()):
            candidates.append((int(ch), int(yy), int(xx), float(vv)))
    if len(candidates) == 0:
        raise RuntimeError("No active entries found under current threshold/settings.")

    if topk_total is not None and len(candidates) > int(topk_total):
        candidates.sort(key=lambda t: abs(t[3]), reverse=True)
        candidates = candidates[: int(topk_total)]

    vals_abs = np.asarray([abs(vv) for (_, _, _, vv) in candidates], dtype=np.float32)
    vmin, vmax = float(vals_abs.min()), float(vals_abs.max())

    pitch_base = spatial_pitch if spatial_stride is None else spatial_stride
    pitch = float(max(pitch_base, 1e-3))
    Wc = float(W) * pitch
    Hc = float(H) * pitch

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim([0.0, Wc])
    ax.set_ylim([0.0, Hc])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor([0.5, 0.5, 0.5])

    n_drawn, n_iso = 0, 0
    for ch, yy, xx, vv in candidates:
        res = needle_properties[ch]
        x_line = np.asarray(res["x"], dtype=np.float32)
        y_line = np.asarray(res["y"], dtype=np.float32)
        mx, my = [float(v) for v in res.get("mean", [x_line.mean(), y_line.mean()])]
        denom = float(max(np.ptp(x_line), np.ptp(y_line), 1e-4))

        x_center = (float(xx) + 0.5) * pitch
        y_center = (float(H) - (float(yy) + 0.5)) * pitch if flip_y else (float(yy) + 0.5) * pitch

        mix = float(np.clip(glyph_size_stride_mix, 0.0, 1.0))
        glyph_scale_eff = float(glyph_scale) * ((1.0 - mix) + mix * pitch)
        x_local = ((x_line - mx) / denom) * glyph_scale_eff + x_center
        y_local = ((y_line - my) / denom) * glyph_scale_eff + y_center

        wn = (abs(vv) - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0
        alpha_i = float(alpha_min + (alpha_max - alpha_min) * wn)
        lw_i = float(linewidth * (0.75 + 0.75 * wn))

        is_iso = draw_needle_glyph(
            ax,
            res,
            x_line=x_local,
            y_line=y_local,
            center_xy=(x_center, y_center),
            color=color,
            alpha=alpha_i,
            linewidth=lw_i,
            width_threshold=width_thr,
            isotropic_mode=isotropic_mode,
            isotropic_alpha_scale=isotropic_alpha_scale,
            isotropic_size_scale=isotropic_size_scale,
            size_override=float(max(0.12, glyph_scale_eff * float(isotropic_size_scale))),
        )
        n_iso += int(is_iso)
        n_drawn += 1

    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    print(
        f"feature-map needles: drawn={n_drawn}, iso={n_iso}, CxHxW={tuple(a.shape)}, topk_total={topk_total}, "
        f"spatial_step={pitch:.2f}, glyph_mix={glyph_size_stride_mix:.2f}, width_thr={width_thr:.3f}, "
        f"iso_mode={isotropic_mode}, flip_y={flip_y}"
    )


def vis_patches(patches, title="", figsize=None, colorbar=False,
                      ncol=None, pad_value="min", show=True,
                      return_tensor=False, vmin=None, vmax=None, fontsize=20,
                      dpi=None, normalize = False,cmap_value =2,padding=1,name = "plot"):
    """
    Given patches of images in the dataset, create a grid and display it.

    Parameters
    ----------
    patches : Tensor of (batch_size, pixels_per_patch) or 
        (batch_size, channels, pixels_per_patch)

    title : String; title of figure. Optional.
    """
    if patches.dim() == 4:
        patches = patches.flatten(2)
    
    if normalize:
        # print(patches.norm(dim=[-1,-2],keepdim=True).shape)
        patches=patches-patches.mean(dim=(1,2),keepdim=True)
        # patches=patches/patches.norm(dim=[-1,-2],keepdim=True)
        p2p = patches.amax(dim=(1,2),keepdim=True)-patches.amin(dim=(1,2),keepdim=True).clamp(min=1e-8)
        patches=(patches/p2p).clamp(min=-cmap_value,max=cmap_value)
    
    if patches.dim() == 2:
        channels = 1
        patches.unsqueeze_(1)
    else:
        channels = patches.size(1)
    batch_size = patches.size(0)
    size = int(np.sqrt(patches.size(-1)))

    img_grid = []
    for i in range(batch_size):
        img = torch.reshape(patches[i], (channels, size, size))
        img_grid.append(img)

    if pad_value != 0:
        if pad_value == "min":
            pad_value = torch.min(patches)
        elif pad_value == "max":
            pad_value = torch.max(patches)

    if not ncol:
        ncol = int(np.sqrt(batch_size))
    out = make_grid(img_grid, padding=padding, nrow=ncol, pad_value=pad_value)
    
    # normalize between 0 and 1 for rgb
    if channels == 3:
        out = ((out - torch.min(out))/(torch.max(out) - torch.min(out))).permute(1, 2, 0)
    else:
        out = out[0]
        
    # 1. Prepare the data for Matplotlib
    out_show = out.detach().cpu() if torch.is_tensor(out) else out
    
    # 2. Setup the plot and draw the image FIRST
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=fontsize)
    ax.set_frame_on(False)
    
    if vmin is not None and vmax is not None:
        if channels != 3:
            plt.imshow(out_show, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            plt.imshow(out_show, vmin=vmin, vmax=vmax)
    else:
        if channels != 3:
            plt.imshow(out_show, cmap="gray")
        else:
            plt.imshow(out_show)
            
    if colorbar:
        plt.colorbar(fraction=0.046, pad=0.04)

    # 3. Save the figure AFTER drawing, but BEFORE showing
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/{}.png'.format(name), bbox_inches='tight') # bbox_inches='tight' removes extra white borders
    
    # 4. Finally, show or close
    if show:
        plt.show()
    else:
        plt.close(fig) # Frees up memory if you are only saving the images in a loop
        
    if return_tensor:
        return out

def load_lit_model(base_dir, device="cpu", ckpt_name=None):
    """
    Load LitDenoiser + checkpoint from an experiment folder.

    Args:
        base_dir: directory containing config.json and checkpoint files.
        device: torch device string, e.g. "cpu" or "cuda".
        ckpt_name: optional checkpoint filename. If None, uses
            config["best_ckpt"] or "ema_model.ckpt".

    Returns:
        (config, model) where model is LitDenoiser on `device` and in eval mode.
    """
    # Local import prevents circular dependency because main.py imports utils.py.
    from main import LitDenoiser

    base = Path(base_dir).expanduser()
    with (base / "config.json").open("r") as f:
        config = json.load(f)

    sig = inspect.signature(LitDenoiser.__init__)
    valid_keys = {k for k in sig.parameters.keys() if k != "self"}
    model_kwargs = {k: config[k] for k in valid_keys if k in config}

    # Map legacy "no_*" config flags to current constructor args.
    if "no_learning_horizontal" in config and "learning_horizontal" in valid_keys:
        model_kwargs["learning_horizontal"] = not config["no_learning_horizontal"]
    if "no_noise_embedding" in config and "noise_embedding" in valid_keys:
        model_kwargs["noise_embedding"] = not config["no_noise_embedding"]
    if "no_bias" in config and "bias" in valid_keys:
        model_kwargs["bias"] = not config["no_bias"]
    if "no_tied_transpose" in config and "tied_transpose" in valid_keys:
        model_kwargs["tied_transpose"] = not config["no_tied_transpose"]
    # model_kwargs["model_arch"] = "neural_sheet2"

    model = LitDenoiser(**model_kwargs).to(device).eval()

    ckpt_file = ckpt_name or config.get("best_ckpt", "ema_model.ckpt")
    ckpt_path = base / ckpt_file
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    def _patch_state_dict(sd, ref_sd):
        """Expand scalar parameters to match the model's expected shape."""
        patched = {}
        for k, v in sd.items():
            if k in ref_sd and v.shape != ref_sd[k].shape:
                target = ref_sd[k]
                # Scalar [1,1,1,1] → per-channel [1,C,1,1]: broadcast-expand
                if v.numel() == 1 or (v.dim() == target.dim() and
                                       all(sv == 1 or sv == tv
                                           for sv, tv in zip(v.shape, target.shape))):
                    patched[k] = v.expand_as(target).clone()
                else:
                    patched[k] = v  # leave mismatches for strict=False to skip
            else:
                patched[k] = v
        return patched

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = _patch_state_dict(checkpoint["state_dict"],
                               dict(model.named_parameters()))
        model.load_state_dict({**checkpoint["state_dict"], **sd}, strict=False)
    elif isinstance(checkpoint, dict) and any(
        k.startswith("model.") or k.startswith("ema_model.")
        for k in checkpoint.keys()
    ):
        model.load_state_dict(checkpoint, strict=False)
    else:
        # Raw EMA/full-model state_dict.
        ema_ref = dict(model.ema_model.named_parameters())
        patched = _patch_state_dict(checkpoint, ema_ref)
        model.ema_model.load_state_dict({**checkpoint, **patched}, strict=False)
        model.model.load_state_dict({**checkpoint, **patched}, strict=False)

    return config, model


def load_toy_scdeq_checkpoint(exp_name, device=None, save_dir="checkpoints"):
    """
    Load a ToySCDEQ checkpoint saved by notebook/train utilities.

    Args:
        exp_name: experiment folder name under `save_dir`.
        device: torch device string. Defaults to cuda if available.
        save_dir: root checkpoint directory.

    Returns:
        (model, optimizer, hparams, checkpoint)
    """
    from recurrent_diffusion_pkg.model import ToySCDEQ

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    load_path = Path(save_dir).expanduser() / exp_name / "latest_checkpoint.pt"
    if not load_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {load_path}")

    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hparams", {})

    model = ToySCDEQ(
        data_dim=hparams["data_dim"],
        hidden_dim=hparams["hidden_dim"],
        eta=hparams["eta"],
        lam=hparams["lam"],
        jfb_no_grad_iters=hparams["jfb_no_grad_iters"],
        jfb_with_grad_iters=hparams["jfb_with_grad_iters"],
        learning_horizontal=hparams.get("learning_horizontal", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.get("lr", 3e-4))
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, hparams, checkpoint


def cat_then_mean(features_ls):
    return torch.cat(features_ls, dim=-1).mean(-1)


def cat(features_ls):
    return torch.cat(features_ls, dim=-1)


def compute_energy_from_history(
    features_T,
    energy_fun,
    level_agg_func=cat_then_mean,
    normalizer=None,
    mean_dims=(0, 2, 3),
    level_ls=None,
):
    """
    Compute an aggregated statistic from history returned by:
      net(..., return_feature=True, return_history=True)

    features_T format:
      list over time, each entry is a list of per-level feature dicts.
    """
    if not features_T:
        raise ValueError("features_T is empty.")
    n_levels = len(features_T[0])
    if level_ls is None:
        level_ls = list(range(n_levels))

    features_levels = []
    for i in level_ls:
        if normalizer is not None:
            features_i = torch.stack([
                energy_fun(features[i]).mean(dim=mean_dims).flatten()
                / normalizer(features[i]).mean(dim=mean_dims).flatten().clamp_min(1e-12)
                for features in features_T
            ])
        else:
            features_i = torch.stack([
                energy_fun(features[i]).mean(dim=mean_dims).flatten()
                for features in features_T
            ])
        features_levels.append(features_i)
    return level_agg_func(features_levels)


@torch.no_grad()
def compute_activation_stats_over_noise(
    model_net,
    data_loader,
    noise_levels,
    device="cpu",
    max_batches=10,
    n_iters=20,
    level_ls=None,
    dead_threshold=0.0,
    show_progress=True,
    collect_distribution=False,
    dist_bins=60,
    dist_range=(-2.0, 2.0),
):
    """
    Compute activation rates and dead-unit stats by noise level.

    Returns:
      dict[noise] = {
        "per_level_rate": [Tensor(C_l), ...],   # P(a_next>0) per unit
        "per_level_max": [Tensor(C_l), ...],    # max a_next per unit (over B,H,W,time)
        "overall_rate": float,                  # mean over all units
        "dead_count": int,                      # rate <= dead_threshold
        "dead_fraction": float,
        "n_units": int,
        # Optional when collect_distribution=True:
        "per_level_hist": [Tensor(C_l, bins), ...],   # hist over batch+spatial+time
        "hist_edges": Tensor(bins+1),
        "per_level_mean": [Tensor(C_l), ...],
        "per_level_std": [Tensor(C_l), ...],
      }
    """
    model_net = model_net.to(device).eval()
    results = {}

    noise_iter = tqdm(noise_levels, desc="noise levels", leave=True) if show_progress else noise_levels
    for noise in noise_iter:
        per_level_sum = None
        per_level_count = None
        per_level_max = None
        per_level_hist = None
        per_level_val_sum = None
        per_level_sq_sum = None
        per_level_npix = None
        n_batches = 0

        batch_iter = tqdm(
            data_loader,
            desc=f"sigma={float(noise):.3f}",
            leave=False,
            total=max_batches,
        ) if show_progress else data_loader

        for batch in batch_iter:
            clean = batch[0] if isinstance(batch, (tuple, list)) else batch
            clean = clean.to(device)
            bsz = clean.shape[0]
            sigma = torch.full((bsz,), float(noise), device=device, dtype=clean.dtype)
            noisy = clean + torch.randn_like(clean) * sigma.view(bsz, 1, 1, 1)

            features_T = model_net(
                noisy,
                noise_labels=sigma,
                infer_mode=True,
                n_iters=n_iters,
                return_feature=True,
                return_history=True,
            )
            if not features_T:
                continue

            use_levels = list(range(len(features_T[0]))) if level_ls is None else list(level_ls)
            if per_level_sum is None:
                per_level_sum = [None for _ in use_levels]
                per_level_count = [0 for _ in use_levels]
                per_level_max = [None for _ in use_levels]
                if collect_distribution:
                    per_level_hist = [None for _ in use_levels]
                    per_level_val_sum = [None for _ in use_levels]
                    per_level_sq_sum = [None for _ in use_levels]
                    per_level_npix = [0 for _ in use_levels]

            for t_features in features_T:
                for li, level_idx in enumerate(use_levels):
                    a_next = t_features[level_idx]["a_next"]
                    rate = (a_next > 0).float().mean(dim=(0, 2, 3)).cpu()  # [C]
                    if per_level_sum[li] is None:
                        per_level_sum[li] = torch.zeros_like(rate)
                    per_level_sum[li] += rate
                    per_level_count[li] += 1

                    # Channel-wise maxima over (B,H,W), then accumulate across time.
                    mx = a_next.detach().amax(dim=(0, 2, 3)).float().cpu()  # [C]
                    if per_level_max[li] is None:
                        per_level_max[li] = mx
                    else:
                        per_level_max[li] = torch.maximum(per_level_max[li], mx)

                    if collect_distribution:
                        # Flatten over batch+spatial, keep per-channel axis.
                        # vals_ch: [C, B*H*W]
                        vals_ch = a_next.detach().float().permute(1, 0, 2, 3).reshape(a_next.shape[1], -1).cpu()
                        c = vals_ch.shape[0]

                        if per_level_hist[li] is None:
                            per_level_hist[li] = torch.zeros((c, int(dist_bins)), dtype=torch.float32)
                            per_level_val_sum[li] = torch.zeros((c,), dtype=torch.float64)
                            per_level_sq_sum[li] = torch.zeros((c,), dtype=torch.float64)

                        # Running moments.
                        per_level_val_sum[li] += vals_ch.sum(dim=1, dtype=torch.float64)
                        per_level_sq_sum[li] += (vals_ch * vals_ch).sum(dim=1, dtype=torch.float64)
                        per_level_npix[li] += int(vals_ch.shape[1])

                        # Running histogram per neuron.
                        dmin, dmax = float(dist_range[0]), float(dist_range[1])
                        for ci in range(c):
                            h = torch.histc(vals_ch[ci], bins=int(dist_bins), min=dmin, max=dmax)
                            per_level_hist[li][ci] += h

            n_batches += 1
            if n_batches >= max_batches:
                break

        if per_level_sum is None:
            results[float(noise)] = {
                "per_level_rate": [],
                "per_level_max": [],
                "overall_rate": float("nan"),
                "dead_count": 0,
                "dead_fraction": float("nan"),
                "n_units": 0,
                "per_level_hist": [],
                "hist_edges": torch.linspace(float(dist_range[0]), float(dist_range[1]), int(dist_bins) + 1),
                "per_level_mean": [],
                "per_level_std": [],
            }
            continue

        per_level_rate = [s / max(c, 1) for s, c in zip(per_level_sum, per_level_count)]
        all_rates = torch.cat(per_level_rate) if per_level_rate else torch.tensor([])
        n_units = int(all_rates.numel())
        dead_count = int((all_rates <= dead_threshold).sum().item()) if n_units else 0

        out = {
            "per_level_rate": per_level_rate,
            "per_level_max": per_level_max if per_level_max is not None else [],
            "overall_rate": float(all_rates.mean().item()) if n_units else float("nan"),
            "dead_count": dead_count,
            "dead_fraction": float(dead_count / max(n_units, 1)),
            "n_units": n_units,
        }
        if collect_distribution:
            per_level_mean = []
            per_level_std = []
            for li in range(len(per_level_rate)):
                if per_level_val_sum[li] is None or per_level_npix[li] == 0:
                    per_level_mean.append(torch.tensor([]))
                    per_level_std.append(torch.tensor([]))
                    continue
                n = float(per_level_npix[li])
                mu = (per_level_val_sum[li] / n).float()
                var = (per_level_sq_sum[li] / n).float() - mu * mu
                var = torch.clamp(var, min=0.0)
                per_level_mean.append(mu)
                per_level_std.append(torch.sqrt(var))

            out["per_level_hist"] = per_level_hist
            out["hist_edges"] = torch.linspace(float(dist_range[0]), float(dist_range[1]), int(dist_bins) + 1)
            out["per_level_mean"] = per_level_mean
            out["per_level_std"] = per_level_std
        results[float(noise)] = out

    return results


def plot_neuron_distribution_grid(
    stats_by_noise,
    noise_level,
    level_idx=0,
    neuron_indices=None,
    top_k=16,
    sort_by="rate",
    ncols=4,
    normalize_hist=True,
    positive_only=True,
):
    """
    Plot per-neuron activation distributions (histograms) for one noise level.
    Requires stats computed with collect_distribution=True.
    """
    key = float(noise_level)
    if key not in stats_by_noise:
        # nearest fallback
        keys = sorted([float(k) for k in stats_by_noise.keys()])
        key = min(keys, key=lambda k: abs(k - float(noise_level)))

    st = stats_by_noise[key]
    if "per_level_hist" not in st or len(st["per_level_hist"]) == 0:
        raise ValueError("No distributions found. Recompute with collect_distribution=True.")

    hist = st["per_level_hist"][int(level_idx)]  # [C, bins]
    if hist is None or hist.numel() == 0:
        raise ValueError(f"Level {level_idx} has empty histogram.")
    edges = st["hist_edges"]  # [bins+1]
    centers = 0.5 * (edges[:-1] + edges[1:])

    rates = st["per_level_rate"][int(level_idx)]
    means = st.get("per_level_mean", [None])[int(level_idx)] if "per_level_mean" in st else None

    C = hist.shape[0]
    if neuron_indices is None:
        if sort_by == "mean" and means is not None and means.numel() == C:
            order = torch.argsort(means, descending=True)
        else:
            order = torch.argsort(rates, descending=True)
        neuron_indices = order[: min(int(top_k), C)].tolist()
    else:
        neuron_indices = [int(i) for i in neuron_indices if 0 <= int(i) < C][: int(top_k)]

    n = len(neuron_indices)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(max(n, 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.4 * nrows), squeeze=False)

    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        ni = neuron_indices[i]
        h = hist[ni].float()
        x = centers
        if positive_only:
            pos_mask = x > 0
            x = x[pos_mask]
            h = h[pos_mask]
        if normalize_hist:
            denom = h.sum().clamp_min(1e-12)
            h = h / denom
        ax.plot(x.numpy(), h.numpy(), linewidth=1.5)
        title = f"n{ni} rate={float(rates[ni]):.3f}"
        if means is not None and torch.is_tensor(means) and means.numel() == C:
            title += f" mu={float(means[ni]):.3f}"
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Neuron distributions @ noise={key} level={level_idx}", fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig, neuron_indices


def build_loaders_from_config(
    config,
    batch_size=64,
    num_workers=4,
    val_split=0.2,
    test_split=0.0,
    seed=42,
):
    """
    Build Lightning DataModule and loaders from experiment config.

    Returns:
      (dm, train_loader, val_loader, test_loader)
    """
    from recurrent_diffusion_pkg.data import MNISTDataModule, STL10DataModule, ImageDataModule

    dataset = config.get("dataset", "celeba")
    if dataset == "mnist":
        dm = MNISTDataModule(data_dir="./data", batch_size=batch_size)
    elif dataset == "stl10":
        dm = STL10DataModule(data_dir="~/data", batch_size=batch_size, num_workers=num_workers)
    else:
        if dataset not in ("celeba", "berkeley", "vh"):
            raise ValueError(f"Unknown dataset for ImageDataModule: {dataset}")
        data_dir_cfg = config.get("data_dir", None)
        if data_dir_cfg is not None:
            data_dir = str(Path(data_dir_cfg).expanduser())
        else:
            default_data_dirs = {
                "celeba": "~/celeba",
                "berkeley": "/home/zeyu/data/BSDS500",
                "vh": "/home/zeyu/vanhateren_all/vh_patches256_train",
            }
            data_dir = str(Path(default_data_dirs[dataset]).expanduser())
        dm = ImageDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=config.get("img_size", 64),
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            grayscale=config.get("grayscale", False),
            no_resize=config.get("no_resize", False),
            random_crop=config.get("random_crop", False),
        )

    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    return dm, train_loader, val_loader, test_loader


def split_neurons_by_ff_threshold(model_net, threshold=0.35, use_injection_levels_only=True):
    """
    Split neuron indices into high/low FF-norm groups for each level.

    Returns:
      high_noi: dict[level_idx] -> LongTensor(indices)
      low_noi: dict[level_idx] -> LongTensor(indices)
      ff_norms: dict[level_idx] -> Tensor(norms per channel)
    """
    if use_injection_levels_only and hasattr(model_net, "injection_dic"):
        level_indices = list(model_net.injection_dic.keys())
    else:
        level_indices = list(range(len(model_net.levels)))

    high_noi, low_noi, ff_norms = {}, {}, {}
    for level_idx in level_indices:
        conv = model_net.levels[level_idx].decoder.conv
        if hasattr(conv, "weight_g"):
            norms = conv.weight_g.detach().flatten().cpu()
        else:
            # weight shape for conv_transpose: [C_in, C_out, k, k]
            norms = conv.weight.detach().flatten(1).norm(dim=1).cpu()

        ff_norms[level_idx] = norms
        high_noi[level_idx] = (norms >= threshold).nonzero(as_tuple=False).flatten().long()
        low_noi[level_idx] = (norms < threshold).nonzero(as_tuple=False).flatten().long()

    return high_noi, low_noi, ff_norms


@torch.no_grad()
def reconstruct_from_noi(a, model_net, noi_by_level, return_complement=True):
    """
    Reconstruct image using selected neurons (NOI) at each level.

    Args:
      a: list of latent tensors per level (from model(..., return_feature=True)["a"])
      model_net: neural_sheet2 model (e.g., lit_model.ema_model)
      noi_by_level: dict[level_idx] -> iterable of channel indices to keep
      return_complement: if True, also reconstruct from complementary channels

    Returns:
      recon_keep, recon_other (if return_complement=True) else recon_keep
    """
    if hasattr(model_net, "injection_dic"):
        level_indices = list(model_net.injection_dic.keys())
    else:
        level_indices = list(range(len(model_net.levels)))

    decoded_keep = []
    decoded_other = []

    for level_idx in level_indices:
        a_i = a[level_idx]
        c = a_i.size(1)
        keep_idx = noi_by_level.get(level_idx, [])
        keep_idx = torch.as_tensor(keep_idx, device=a_i.device, dtype=torch.long)
        if keep_idx.numel() > 0:
            keep_idx = keep_idx[(keep_idx >= 0) & (keep_idx < c)]
            keep_idx = torch.unique(keep_idx)

        mask = torch.zeros(c, dtype=torch.bool, device=a_i.device)
        if keep_idx.numel() > 0:
            mask[keep_idx] = True

        a_keep = a_i.clone()
        a_keep[:, ~mask] = 0
        decoded_keep.append(model_net.levels[level_idx].decoder(a_keep))

        if return_complement:
            a_other = a_i.clone()
            a_other[:, mask] = 0
            decoded_other.append(model_net.levels[level_idx].decoder(a_other))

    recon_keep = model_net.decoder(decoded_keep)
    if not return_complement:
        return recon_keep

    recon_other = model_net.decoder(decoded_other)
    return recon_keep, recon_other


def visualize_pyramid(x_py, show=False, return_feature=True):
    """
    Visualize a pyramid on a shared canvas:
      - left: highest resolution
      - right: remaining levels stacked vertically

    Each level is independently normalized to [0,1] for visibility.
    """
    assert isinstance(x_py, (list, tuple)) and len(x_py) > 0
    assert x_py[0].dim() == 4 and x_py[0].shape[0] == 1

    _, c, h0, w0 = x_py[0].shape

    norm_levels = []
    for x in x_py:
        x0 = x[0]
        mn, mx = x0.min(), x0.max()
        if (mx - mn) > 0:
            x_norm = (x0 - mn) / (mx - mn)
        else:
            x_norm = torch.zeros_like(x0)
        norm_levels.append(x_norm)

    canvas = torch.ones((c, h0, int(1.5 * w0)), dtype=norm_levels[0].dtype, device=norm_levels[0].device)
    canvas[:, :h0, :w0] = norm_levels[0]

    y_offset = 0
    sep_value = 1.0
    for level in norm_levels[1:]:
        c_i, h_i, w_i = level.shape
        if y_offset >= h0:
            break
        h_fit = min(h_i, h0 - y_offset)
        w_fit = min(w_i, w0)
        canvas[:, y_offset:y_offset + h_fit, w0:w0 + w_fit] = level[:, :h_fit, :w_fit]
        if y_offset + h_fit < h0:
            canvas[:, y_offset + h_fit:y_offset + h_fit + 1, w0:int(1.5 * w0)] = sep_value
        y_offset += h_fit + 1

    img = canvas.detach().cpu().clamp(0, 1)
    if c == 1:
        to_show = img[0]
        if show:
            plt.imshow(to_show, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    else:
        to_show = img.permute(1, 2, 0)
        if show:
            plt.imshow(to_show, interpolation="nearest", vmin=0.0, vmax=1.0)
    if show:
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if return_feature:
        return to_show
    return None


def convert_img_to_vis(img):
    c, _, _ = img.shape
    if c == 1:
        return img[0].cpu()
    return img.permute(1, 2, 0).cpu()


def load_one_img_pyramid(x_py, idx=0):
    return visualize_pyramid([x[idx:idx + 1] for x in x_py], show=False, return_feature=True)


def show_images_with_titles(
    viz_ls_per_noise,
    titles=None,
    return_image=False,
    dpi=100,
    normalize_each=True,
    save_path=None,
):
    """
    Show a grid where rows are noise levels and columns are views/variants.
    """
    assert len(viz_ls_per_noise) > 0, "Need at least one row"
    num_rows = len(viz_ls_per_noise)
    num_cols = len(viz_ls_per_noise[0])
    for row in viz_ls_per_noise:
        assert len(row) == num_cols, "All rows must have same number of columns"

    imgs = [[np.asarray(im) for im in row] for row in viz_ls_per_noise]
    h = imgs[0][0].shape[0]
    for r in range(num_rows):
        for c in range(num_cols):
            assert imgs[r][c].shape[0] == h, "All images must share same height"

    widths = []
    for c in range(num_cols):
        w_c = imgs[0][c].shape[1]
        for r in range(num_rows):
            assert imgs[r][c].shape[1] == w_c, f"Column {c}: all rows must share same width"
        widths.append(w_c)

    if titles is None:
        titles = ["" for _ in range(num_cols)]
    assert len(titles) == num_cols

    proc_imgs = []
    for r in range(num_rows):
        row_proc = []
        for c in range(num_cols):
            im = imgs[r][c]
            if normalize_each:
                im = im.astype(np.float32)
                vmin, vmax = im.min(), im.max()
                if vmax > vmin:
                    im = (im - vmin) / (vmax - vmin)
                else:
                    im = np.zeros_like(im, dtype=np.float32)
            else:
                if np.issubdtype(im.dtype, np.floating):
                    im = np.clip(im, 0.0, 1.0)
                else:
                    im = np.clip(im, 0, 255)
            row_proc.append(im)
        proc_imgs.append(row_proc)

    total_width = sum(widths)
    aspect = total_width / h
    fig_width = 4 * aspect
    fig_height = 4 * num_rows
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=num_rows,
        ncols=num_cols,
        width_ratios=widths,
        wspace=0.05,
        hspace=0.05,
        left=0.0,
        right=1.0,
    )

    for r in range(num_rows):
        for c in range(num_cols):
            im = proc_imgs[r][c]
            ax = fig.add_subplot(gs[r, c])
            if im.ndim == 2:
                ax.imshow(im, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(im, interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and titles[c]:
                ax.set_title(titles[c], fontsize=16)
            ax.set_aspect("equal")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if not return_image:
        if save_path is None:
            plt.show()
        plt.close(fig)
        return None

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img_fig = buf[..., :3].copy()
    plt.close(fig)
    return img_fig


@torch.no_grad()
def visualize_denoising_before_after_ablation(
    clean_batch,
    model_net,
    sigma_viz_ls=(0.1, 0.5),
    device="cpu",
    n_iters=20,
    ablate_noi=None,
    ablate_mode="spatial_mean",
    ablate_global_mean_by_level=None,
    ablate_ff_noise_by_level=None,
    b_idx_viz=0,
    save_path=None,
    return_image=False,
):
    """
    Compare baseline vs ablated denoising per noise level for one exemplar.
    """
    model_net = model_net.to(device).eval()
    clean_batch = clean_batch.to(device)
    bsz = clean_batch.shape[0]

    viz_ls_ls = []
    out_features = []
    for sigma_viz in sigma_viz_ls:
        sigma = torch.full((bsz,), float(sigma_viz), device=device, dtype=clean_batch.dtype)
        noisy = clean_batch + torch.randn_like(clean_batch) * sigma.view(bsz, 1, 1, 1)
        use_global_mean_mode = isinstance(ablate_mode, str) and ablate_mode.endswith("global_spatial_mean")
        use_global_noise_mode = isinstance(ablate_mode, str) and ablate_mode.endswith("global_spatial_noise")
        if use_global_mean_mode and callable(ablate_global_mean_by_level):
            ablate_global_mean_i = ablate_global_mean_by_level(float(sigma_viz))
        elif use_global_mean_mode:
            ablate_global_mean_i = ablate_global_mean_by_level
        else:
            ablate_global_mean_i = None
        if use_global_noise_mode and callable(ablate_ff_noise_by_level):
            ablate_ff_noise_i = ablate_ff_noise_by_level(float(sigma_viz), bsz, device, clean_batch.dtype)
        elif use_global_noise_mode:
            ablate_ff_noise_i = ablate_ff_noise_by_level
        else:
            ablate_ff_noise_i = None

        feat_base = model_net(
            noisy,
            noise_labels=sigma,
            infer_mode=True,
            n_iters=n_iters,
            return_feature=True,
        )
        if ablate_noi is not None:
            call_kwargs = dict(
                noise_labels=sigma,
                infer_mode=True,
                n_iters=n_iters,
                return_feature=True,
                ablate_noi=ablate_noi,
                ablate_mode=ablate_mode,
            )
            if use_global_mean_mode:
                call_kwargs["ablate_global_mean_by_level"] = ablate_global_mean_i
            if use_global_noise_mode:
                call_kwargs["ablate_ff_noise_by_level"] = ablate_ff_noise_i
            feat_ablt = model_net(noisy, **call_kwargs)
        else:
            feat_ablt = None

        with torch.no_grad():
            noisy_py = model_net.encoder(noisy)

        row = [
            convert_img_to_vis(clean_batch[b_idx_viz]),
            convert_img_to_vis(noisy[b_idx_viz]),
            load_one_img_pyramid(noisy_py, b_idx_viz),
            load_one_img_pyramid(feat_base["decoded"], b_idx_viz),
            convert_img_to_vis(feat_base["denoised"][b_idx_viz]),
        ]
        if feat_ablt is not None:
            row += [
                load_one_img_pyramid(feat_ablt["decoded"], b_idx_viz),
                convert_img_to_vis(feat_ablt["denoised"][b_idx_viz]),
            ]
        viz_ls_ls.append(row)
        out_features.append((feat_base, feat_ablt))

    titles = [
        "Clean",
        "Noisy",
        "Noisy (pyr)",
        "Base decoded (pyr)",
        "Base denoised",
    ]
    if ablate_noi is not None:
        titles += ["Ablated decoded (pyr)", "Ablated denoised"]

    img = show_images_with_titles(
        viz_ls_ls,
        titles=titles,
        return_image=return_image,
        save_path=save_path,
        dpi=180,
        normalize_each=True,
    )
    return out_features, img


def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def gaussian_2d_full(coords, a, x0, y0, sigma_x, sigma_y, rho, c):
    x, y = coords
    cov = np.array([
        [sigma_x ** 2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y ** 2],
    ])
    inv_cov = np.linalg.inv(cov)
    dx = x - x0
    dy = y - y0
    exponent = -(
        dx * (inv_cov[0, 0] * dx + inv_cov[0, 1] * dy)
        + dy * (inv_cov[1, 0] * dx + inv_cov[1, 1] * dy)
    ) / 2
    return np.exp(a) * np.exp(exponent) + c


def unit_norm_filters(filters, eps=1e-8):
    """
    Normalize filter bank to unit norm per filter.

    Args:
      filters: Tensor [N, C, H, W]
    """
    flat = filters.flatten(1)
    norms = flat.norm(dim=1, keepdim=True).clamp_min(eps)
    return (flat / norms).view_as(filters), norms.squeeze(1)


def get_decoder_filters_by_level(model_net, level_idx, unit_norm=False):
    """
    Returns:
      filters: Tensor [N, C, H, W] (CPU)
      ff_norms: Tensor [N] (CPU)
    """
    conv = model_net.levels[level_idx].decoder.conv
    if hasattr(conv, "weight_v") and hasattr(conv, "weight_g"):
        filters = conv.weight_v.detach().cpu()
        ff_norms = conv.weight_g.detach().flatten().cpu()
    else:
        filters = conv.weight.detach().cpu()
        ff_norms = filters.flatten(1).norm(dim=1)

    if unit_norm:
        filters, _ = unit_norm_filters(filters)
    return filters, ff_norms


def _dominant_color_spatial_decomposition(kernel_chw):
    """
    Color-aware decomposition:
      kernel[C,H,W] ~= color_vec[C] outer spatial[H,W]
    via rank-1 SVD on C x (H*W).
    """
    c, h, w = kernel_chw.shape
    k2 = kernel_chw.reshape(c, h * w)
    u, s, vt = np.linalg.svd(k2, full_matrices=False)
    color_vec = u[:, 0]
    spatial = (s[0] * vt[0]).reshape(h, w)

    # Sign convention for reproducibility.
    max_idx = np.unravel_index(np.argmax(np.abs(spatial)), spatial.shape)
    if spatial[max_idx] < 0:
        spatial = -spatial
        color_vec = -color_vec
    return spatial, color_vec


def get_basis_needle_plot(image_2d):
    """
    Fit a needle-like summary (center, orientation, length) from a 2D basis map.

    Improvements over original:
    - Tight sigma bounds [0.4, 0.65*ksize] to require envelope localization
    - 7 random restarts, keep best-MSE fit
    - R² quality gate (>= 0.30) to reject diffuse/non-Gabor filters
    - Correct length = sigma_major (was 0.3 * sigma_major², quadratically wrong)
    - Moment-based FFT orientation estimate (no slow 180-rotation loop)
    """
    from scipy.optimize import curve_fit
    from scipy.signal import hilbert

    kh, kw = image_2d.shape
    ksize = max(kh, kw)

    # --- Orientation via power-spectrum second moments (no rotation loop) ---
    padded = np.pad(image_2d, ((50, 50), (50, 50)), mode="constant", constant_values=0)
    power_spec = np.abs(np.fft.fftshift(np.fft.fft2(padded))) ** 2
    H, W = power_spec.shape
    cy, cx = H // 2, W // 2
    xx_f = np.arange(W, dtype=float) - cx
    yy_f = np.arange(H, dtype=float) - cy
    xx_f2d, yy_f2d = np.meshgrid(xx_f, yy_f)
    # Exclude DC blob to avoid it dominating
    dc_mask = (np.abs(yy_f2d) > 2) | (np.abs(xx_f2d) > 2)
    ps = power_spec * dc_mask
    total = ps.sum() + 1e-12
    Ixx = float((ps * xx_f2d ** 2).sum() / total)
    Iyy = float((ps * yy_f2d ** 2).sum() / total)
    Ixy = float((ps * xx_f2d * yy_f2d).sum() / total)
    M_ps = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    _, eigvecs_ps = np.linalg.eigh(M_ps)
    gdir = eigvecs_ps[:, 1]  # largest eigenvalue → grating direction [x, y]
    # Edge orientation = perpendicular to grating direction
    theta = float(np.arctan2(gdir[1], gdir[0]) + np.pi / 2)

    # --- Envelope via 1-D Hilbert on each axis ---
    envelope = np.abs(hilbert(hilbert(image_2d, axis=0).real, axis=1))

    xx, yy = np.meshgrid(np.arange(kw), np.arange(kh))
    max_index = int(np.argmax(envelope))
    row_index = max_index // kw
    col_index = max_index % kw

    # --- Gaussian envelope fit with tight localization bounds ---
    sigma_min = 0.4
    sigma_max = ksize * 0.65
    center_slack = 1.5  # allow center up to 1.5 px outside kernel

    signal_var = float(np.var(envelope)) + 1e-12
    best_params = None
    best_mse = float("inf")

    for restart in range(7):
        if restart == 0:
            x0_init = float(col_index)
            y0_init = float(row_index)
            sx_init = min(2.0, sigma_max)
            sy_init = min(2.0, sigma_max)
        else:
            x0_init = float(np.random.uniform(kw * 0.15, kw * 0.85))
            y0_init = float(np.random.uniform(kh * 0.15, kh * 0.85))
            sx_init = float(np.random.uniform(sigma_min, sigma_max * 0.7))
            sy_init = float(np.random.uniform(sigma_min, sigma_max * 0.7))

        p0 = (0.1, x0_init, y0_init, sx_init, sy_init, 0.0, 0.0)
        bounds_lo = (-np.inf, -center_slack, -center_slack,
                     sigma_min, sigma_min, -0.99, -np.inf)
        bounds_hi = (np.inf, kw - 1 + center_slack, kh - 1 + center_slack,
                     sigma_max, sigma_max, 0.99, np.inf)
        try:
            params, _ = curve_fit(
                gaussian_2d_full,
                (xx.ravel(), yy.ravel()),
                envelope.ravel(),
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=5000,
            )
        except RuntimeError:
            continue

        pred = gaussian_2d_full((xx.ravel(), yy.ravel()), *params)
        mse = float(np.mean((pred - envelope.ravel()) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_params = params

    # --- Quality gate ---
    successful_fit = False
    sigma_x, sigma_y, rho = 2.0, 2.0, 0.0
    x0, y0 = float(col_index), float(row_index)
    r2 = 0.0

    if best_params is not None:
        _, _x0, _y0, _sx, _sy, _rho, _ = best_params
        _sx, _sy = abs(float(_sx)), abs(float(_sy))
        r2 = float(1.0 - best_mse / signal_var)
        if r2 >= 0.30 and max(_sx, _sy) < sigma_max:
            successful_fit = True
            x0, y0 = float(_x0), float(_y0)
            sigma_x, sigma_y, rho = _sx, _sy, float(_rho)

    # --- Length = sigma_major (not 0.3 * sigma_major²) ---
    if successful_fit:
        cov = np.array([
            [sigma_x ** 2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y ** 2],
        ])
        length = float(np.sqrt(max(np.linalg.eigvalsh(cov))))
    else:
        length = 0.0

    mean = (x0, y0)
    one_extreme = polar_to_cartesian(length, theta)
    other_extreme = polar_to_cartesian(length, theta + np.pi)

    return {
        "center_surround": False,
        "successful_fit": successful_fit,
        "theta": theta,
        "length": length,
        "mean": mean,
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "rho": float(rho),
        "r2": r2,
        "x": np.asarray([one_extreme[0], other_extreme[0]]) + x0,
        "y": np.asarray([one_extreme[1], other_extreme[1]]) + y0,
    }


def _estimate_edge_side_colors(kernel_chw, theta, mean_xy):
    """
    Estimate two side colors across the fitted orientation line.
    """
    c, h, w = kernel_chw.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0 = mean_xy

    # Normal to line direction(theta): n = (-sin, cos)
    nx, ny = -np.sin(theta), np.cos(theta)
    signed_dist = nx * (xx - x0) + ny * (yy - y0)
    pos_mask = signed_dist >= 0
    neg_mask = ~pos_mask

    # Weight with activity magnitude to suppress empty background.
    weight = np.linalg.norm(kernel_chw, axis=0)
    w_pos = weight[pos_mask].sum()
    w_neg = weight[neg_mask].sum()

    if w_pos < 1e-12 or w_neg < 1e-12:
        return np.zeros(c, dtype=np.float32), np.zeros(c, dtype=np.float32)

    pos_color = (kernel_chw[:, pos_mask] * weight[pos_mask][None, :]).sum(axis=1) / w_pos
    neg_color = (kernel_chw[:, neg_mask] * weight[neg_mask][None, :]).sum(axis=1) / w_neg
    return pos_color.astype(np.float32), neg_color.astype(np.float32)


def _build_gaussian_envelope(h, w, mean_xy, sigma_x, sigma_y, rho, eps=1e-6):
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0 = mean_xy
    dx = xx - x0
    dy = yy - y0

    sigma_x = max(float(sigma_x), eps)
    sigma_y = max(float(sigma_y), eps)
    rho = float(np.clip(rho, -0.95, 0.95))

    cov = np.array([
        [sigma_x ** 2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y ** 2],
    ], dtype=np.float64)
    inv_cov = np.linalg.inv(cov + eps * np.eye(2))
    quad = (
        dx * (inv_cov[0, 0] * dx + inv_cov[0, 1] * dy)
        + dy * (inv_cov[1, 0] * dx + inv_cov[1, 1] * dy)
    )
    return np.exp(-0.5 * quad).astype(np.float32)


def _estimate_gabor_carrier(spatial_map, theta, mean_xy, sigma_x, sigma_y, rho):
    """
    Estimate frequency/phase/amplitude for:
      spatial ~ amp * envelope * cos(2*pi*f*u + phase),
    where u is along normal-to-edge direction.
    """
    h, w = spatial_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0 = mean_xy

    nx, ny = -np.sin(theta), np.cos(theta)
    u = nx * (xx - x0) + ny * (yy - y0)
    env = _build_gaussian_envelope(h, w, mean_xy, sigma_x, sigma_y, rho)

    s = spatial_map.astype(np.float32).reshape(-1)
    e = env.reshape(-1)
    u_flat = u.reshape(-1).astype(np.float32)

    f_grid = np.linspace(0.02, 0.45, 90, dtype=np.float32)
    best = {"mse": np.inf, "f": 0.08, "b1": 0.0, "b2": 0.0}

    wt = e + 1e-6
    for f in f_grid:
        omega_u = 2.0 * np.pi * f * u_flat
        c = e * np.cos(omega_u)
        si = e * np.sin(omega_u)
        xmat = np.stack([c, si], axis=1)

        xw = xmat * wt[:, None]
        yw = s * wt
        try:
            beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = xmat @ beta
        mse = float(np.mean((s - pred) ** 2))
        if mse < best["mse"]:
            best = {"mse": mse, "f": float(f), "b1": float(beta[0]), "b2": float(beta[1])}

    amp = float(np.hypot(best["b1"], best["b2"]))
    phase = float(np.arctan2(best["b2"], best["b1"]))
    return {
        "frequency": float(best["f"]),
        "phase": phase,
        "amplitude": amp,
        "fit_mse": float(best["mse"]),
    }


def render_gabor_from_fit(fit, out_shape=None, use_color=True):
    """
    Render a reconstructed kernel from fitted gabor params.
    Returns np.ndarray [C,H,W].
    """
    spatial_ref = fit["spatial_map"]
    h, w = spatial_ref.shape if out_shape is None else out_shape
    x0, y0 = fit["mean"]
    theta = fit["theta"]

    sigma_x = fit.get("sigma_x", 2.0)
    sigma_y = fit.get("sigma_y", 2.0)
    rho = fit.get("rho", 0.0)
    freq = fit.get("frequency", 0.08)
    phase = fit.get("phase", 0.0)
    amp = fit.get("amplitude", 1.0)

    yy, xx = np.mgrid[0:h, 0:w]
    nx, ny = -np.sin(theta), np.cos(theta)
    u = nx * (xx - x0) + ny * (yy - y0)
    env = _build_gaussian_envelope(h, w, (x0, y0), sigma_x, sigma_y, rho)
    spatial = amp * env * np.cos(2.0 * np.pi * freq * u + phase)

    if use_color and "color_vec" in fit:
        color_vec = np.asarray(fit["color_vec"], dtype=np.float32)
    else:
        color_vec = np.ones((1,), dtype=np.float32)
    color_vec = color_vec / max(np.linalg.norm(color_vec), 1e-12)
    kernel = color_vec[:, None, None] * spatial[None, :, :]
    return kernel.astype(np.float32)


def plot_original_vs_reconstructed_gabor(
    model_net,
    level_idx,
    fit_bank,
    n_show=32,
    ncol=16,
    unit_norm=True,
):
    """
    Plot original decoder kernels and reconstructed gabor kernels side-by-side.
    """
    filters_t, _ = get_decoder_filters_by_level(model_net, level_idx, unit_norm=unit_norm)
    kept_idx = fit_bank["kept_idx"]
    if len(kept_idx) == 0:
        print(f"Level {level_idx}: no filters above threshold.")
        return

    n_show = int(min(n_show, len(kept_idx)))
    show_idx = kept_idx[:n_show]

    orig = filters_t[show_idx]
    fit_map = {int(r["filter_idx"]): r for r in fit_bank["results"]}
    rec_list = []
    for idx in show_idx:
        fit = fit_map[int(idx)]
        rec = render_gabor_from_fit(fit, out_shape=orig.shape[-2:], use_color=True)
        rec_list.append(torch.from_numpy(rec))
    rec = torch.stack(rec_list, dim=0).to(orig.dtype)

    p_orig = orig.flatten(2)
    p_rec = rec.flatten(2)
    g_orig = vis_patches(p_orig, ncol=ncol, normalize=False, return_tensor=True, show=False)
    g_rec = vis_patches(p_rec, ncol=ncol, normalize=False, return_tensor=True, show=False)

    fig, axes = plt.subplots(2, 1, figsize=(8, 3.2), squeeze=False)
    axes[0, 0].imshow(g_orig.numpy())
    axes[0, 0].set_title(f"Level {level_idx} original kernels (N={n_show})", fontsize=10)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(g_rec.numpy())
    axes[1, 0].set_title(f"Level {level_idx} reconstructed gabor kernels", fontsize=10)
    axes[1, 0].axis("off")
    plt.tight_layout()
    plt.show()


def fit_single_filter_gabor(
    kernel_chw,
    unit_norm=True,
    use_color=True,
):
    """
    Fit one filter to a needle summary with color-aware stats.

    Args:
      kernel_chw: np.ndarray [C,H,W]
      unit_norm: normalize whole filter to unit norm before fitting
      use_color: use rank-1 color decomposition for spatial fit
    """
    k = np.asarray(kernel_chw, dtype=np.float32)
    if unit_norm:
        nrm = np.linalg.norm(k.reshape(-1))
        if nrm > 1e-12:
            k = k / nrm

    if use_color and k.shape[0] > 1:
        spatial_map, color_vec = _dominant_color_spatial_decomposition(k)
    else:
        spatial_map = k.mean(axis=0)
        color_vec = np.ones((k.shape[0],), dtype=np.float32) / np.sqrt(max(k.shape[0], 1))

    fit = get_basis_needle_plot(spatial_map)
    if fit.get("successful_fit", False):
        pos_color, neg_color = _estimate_edge_side_colors(k, fit["theta"], fit["mean"])
    else:
        pos_color = np.zeros((k.shape[0],), dtype=np.float32)
        neg_color = np.zeros((k.shape[0],), dtype=np.float32)

    fit["color_vec"] = color_vec.astype(np.float32)
    fit["pos_color"] = pos_color
    fit["neg_color"] = neg_color
    fit["color_contrast_norm"] = float(np.linalg.norm(pos_color - neg_color))
    fit["spatial_map"] = spatial_map.astype(np.float32)
    fit["sigma_x"] = float(abs(fit.get("sigma_x", 2.0)))
    fit["sigma_y"] = float(abs(fit.get("sigma_y", 2.0)))
    fit["rho"] = float(fit.get("rho", 0.0))
    if fit.get("successful_fit", False):
        carrier = _estimate_gabor_carrier(
            spatial_map=spatial_map,
            theta=fit["theta"],
            mean_xy=fit["mean"],
            sigma_x=fit["sigma_x"],
            sigma_y=fit["sigma_y"],
            rho=fit["rho"],
        )
        fit.update(carrier)

        # Resolve global sign ambiguity (SVD/color sign + phase periodicity).
        # Keep the reconstructed kernel aligned with original kernel polarity.
        rec = render_gabor_from_fit(fit, out_shape=k.shape[-2:], use_color=use_color)
        corr = float(np.sum(k * rec))
        if corr < 0:
            fit["phase"] = float(fit["phase"] + np.pi)
            # Wrap to [-pi, pi] for readability/stability.
            fit["phase"] = float((fit["phase"] + np.pi) % (2.0 * np.pi) - np.pi)
    else:
        fit["frequency"] = 0.08
        fit["phase"] = 0.0
        fit["amplitude"] = 0.0
        fit["fit_mse"] = float("nan")
    return fit


def fit_decoder_level_gabor_bank(
    model_net,
    level_idx,
    ff_threshold=0.31,
    unit_norm=True,
    use_color=True,
):
    """
    Fit all decoder filters above ff_threshold for one level.

    Returns:
      dict with:
        "results": list of per-filter fit dict
        "kept_idx": np.ndarray kept filter indices
        "ff_norms": np.ndarray all ff norms
    """
    filters_t, ff_norms_t = get_decoder_filters_by_level(model_net, level_idx, unit_norm=unit_norm)
    filters = filters_t.numpy()
    ff_norms = ff_norms_t.numpy()

    kept_idx = np.where(ff_norms >= float(ff_threshold))[0]
    results = []
    for idx in kept_idx:
        fit = fit_single_filter_gabor(filters[idx], unit_norm=unit_norm, use_color=use_color)
        fit["filter_idx"] = int(idx)
        fit["ff_norm"] = float(ff_norms[idx])
        results.append(fit)

    return {
        "results": results,
        "kept_idx": kept_idx.astype(np.int64),
        "ff_norms": ff_norms.astype(np.float32),
    }


def plot_gabor_needles(
    fit_bank,
    image_size=16,
    figsize=(8, 8),
    color_mode="contrast",
    alpha=0.85,
):
    """
    Plot fitted needles on a canonical image plane.

    color_mode:
      - "contrast": color by ||pos_color - neg_color||
      - "ff_norm": color by feedforward norm
    """
    results = fit_bank["results"]
    if len(results) == 0:
        plt.figure(figsize=figsize)
        plt.title("No filters above threshold")
        plt.xlim(0, image_size - 1)
        plt.ylim(image_size - 1, 0)
        plt.gca().set_aspect("equal")
        plt.show()
        return

    if color_mode == "ff_norm":
        vals = np.asarray([r.get("ff_norm", 0.0) for r in results], dtype=np.float32)
    else:
        vals = np.asarray([r.get("color_contrast_norm", 0.0) for r in results], dtype=np.float32)
    vmin, vmax = float(vals.min()), float(vals.max())
    denom = max(vmax - vmin, 1e-12)

    plt.figure(figsize=figsize)
    for r, v in zip(results, vals):
        x = r["x"]
        y = r["y"]
        score = (float(v) - vmin) / denom
        color = plt.cm.viridis(score)
        plt.plot(x, y, color=color, alpha=alpha, linewidth=1.5)

    plt.xlim(0, image_size - 1)
    plt.ylim(image_size - 1, 0)
    plt.gca().set_aspect("equal")
    plt.title(f"Needle summary ({color_mode})")
    plt.grid(alpha=0.15)
    plt.show()
