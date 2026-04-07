import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from recurrent_diffusion_pkg.utils import compute_width_threshold, draw_needle_glyph


def _shift_coords(x, y, ox, oy, kw, kh, stride_x=1.0, stride_y=1.0):
    """Shift a needle's coordinates to a spatial offset in a kh x kw neighborhood."""
    dx = (float(ox) - float(kw // 2)) * float(stride_x)
    dy = (float(oy) - float(kh // 2)) * float(stride_y)
    return np.asarray(x, dtype=np.float32) + dx, np.asarray(y, dtype=np.float32) + dy


def to_vis01(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo + 1e-8:
        lo, hi = float(np.min(x)), float(np.max(x))
    y = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return y


def to_vis_img(kernel_chw):
    arr = kernel_chw.detach().cpu().float().numpy() if torch.is_tensor(kernel_chw) else np.asarray(kernel_chw)
    if arr.ndim == 2:
        return arr, "gray"
    if arr.shape[0] == 1:
        return arr[0], "gray"
    rgb = np.transpose(arr[:3], (1, 2, 0))
    return to_vis01(rgb), None


def _normalize_color(fit, out_ch, abs_color=True):
    if "color_vec" in fit:
        color = np.asarray(fit["color_vec"], dtype=np.float32)
    else:
        color = np.ones((out_ch,), dtype=np.float32)
    if color.shape[0] < out_ch:
        color = np.pad(color, (0, out_ch - color.shape[0]), mode="constant")
    color = color[:out_ch]
    if abs_color:
        color = np.abs(color)
    n = float(np.linalg.norm(color))
    if n < 1e-8:
        color = np.ones((out_ch,), dtype=np.float32) / np.sqrt(max(out_ch, 1))
    else:
        color = color / n
    return color


def _render_glyph_from_fit_custom(
    fit,
    out_ch,
    h,
    w,
    width_threshold,
    line_len_scale=2.6,
    line_w_scale=0.4,
    min_half_len=1.0,
    ball_sigma_scale=0.95,
    render_scale=1,
    aa_scale=2,
):
    """Width-threshold glyph renderer: wide -> ball, narrow -> line.

    Renders on an internal supersampled grid (render_scale * aa_scale), then
    downsamples to (h*render_scale, w*render_scale) for anti-aliased sharp lines.
    """
    s = int(max(1, render_scale))
    aa = int(max(1, aa_scale))
    sr = s * aa
    hh, ww = h * sr, w * sr

    cx, cy = [float(v) for v in fit.get("mean", ((w - 1) * 0.5, (h - 1) * 0.5))]
    cx = float(np.clip(cx, 0.0, w - 1.0)) * sr + 0.5 * (sr - 1)
    cy = float(np.clip(cy, 0.0, h - 1.0)) * sr + 0.5 * (sr - 1)
    th = float(fit.get("theta", 0.0))

    sx = max(float(abs(fit.get("sigma_x", 1.0))), 1e-4)
    sy = max(float(abs(fit.get("sigma_y", 1.0))), 1e-4)
    major, minor = max(sx, sy), min(sx, sy)

    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)

    # Decide ball vs line in native units to keep logic consistent.
    if minor >= float(width_threshold):
        sig = float(np.clip(ball_sigma_scale * minor * sr, 0.45 * sr, 0.48 * min(hh, ww)))
        q = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(sig ** 2, 1e-6)
        spatial = np.exp(-0.5 * q)
    else:
        if "x" in fit and "y" in fit:
            x0, x1 = [float(v) for v in fit["x"]]
            y0, y1 = [float(v) for v in fit["y"]]
            x0 = x0 * sr + 0.5 * (sr - 1)
            x1 = x1 * sr + 0.5 * (sr - 1)
            y0 = y0 * sr + 0.5 * (sr - 1)
            y1 = y1 * sr + 0.5 * (sr - 1)

            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            dx, dy = x1 - x0, y1 - y0
            nrm = np.sqrt(dx * dx + dy * dy)
            if nrm < 1e-6:
                ux, uy = np.cos(th), np.sin(th)
                base_half_len = float(min_half_len * sr)
            else:
                ux, uy = dx / nrm, dy / nrm
                base_half_len = 0.5 * nrm
            desired_half = float(max(base_half_len * line_len_scale, min_half_len * sr))
        else:
            ux, uy = np.cos(th), np.sin(th)
            mx, my = cx, cy
            desired_half = float(max(line_len_scale * major * sr, min_half_len * sr))

        # Cap by available border distance along the line direction.
        ex = max(abs(ux), 1e-8)
        ey = max(abs(uy), 1e-8)
        max_half = min(mx / ex, (ww - 1 - mx) / ex, my / ey, (hh - 1 - my) / ey)
        half_len = float(np.clip(desired_half, 1.0 * sr, 0.95 * max(max_half, 1.0 * sr)))

        x0, y0 = mx - ux * half_len, my - uy * half_len
        x1, y1 = mx + ux * half_len, my + uy * half_len

        x0 = float(np.clip(x0, 0.0, ww - 1.0))
        x1 = float(np.clip(x1, 0.0, ww - 1.0))
        y0 = float(np.clip(y0, 0.0, hh - 1.0))
        y1 = float(np.clip(y1, 0.0, hh - 1.0))

        vx, vy = x1 - x0, y1 - y0
        vv = vx * vx + vy * vy + 1e-8
        t = ((xx - x0) * vx + (yy - y0) * vy) / vv
        t = np.clip(t, 0.0, 1.0)
        px = x0 + t * vx
        py = y0 + t * vy

        line_w = float(np.clip(line_w_scale * minor * sr, 0.6 * aa, 2.8 * aa))
        dist2 = (xx - px) ** 2 + (yy - py) ** 2
        core = np.exp(-0.5 * dist2 / max(line_w ** 2, 1e-6))
        taper = 0.85 + 0.15 * np.sqrt(np.clip(np.sin(np.pi * t), 0.0, None))
        spatial = core * taper

    m = float(spatial.max())
    if m < 1e-6:
        d = np.abs((xx - cx) * (-np.sin(th)) + (yy - cy) * np.cos(th))
        spatial = np.exp(-0.5 * (d / max(0.9 * aa, 1e-6)) ** 2)
        m = float(spatial.max())

    spatial = spatial / max(m, 1e-8)

    # Anti-aliased downsample to target render_scale.
    if aa > 1:
        spatial_t = torch.from_numpy(spatial.astype(np.float32))[None, None]
        spatial_t = F.avg_pool2d(spatial_t, kernel_size=aa, stride=aa)
        spatial = spatial_t[0, 0].numpy()

    color = _normalize_color(fit, out_ch, abs_color=True)
    kernel = color[:, None, None] * spatial[None, :, :]
    return torch.from_numpy(kernel.astype(np.float32))


def plot_feature_on_needles(
    a_feat,
    *,
    net,
    fit_by_level,
    level_idx=0,
    render_scale_hr=4,
    width_thr_quantile=0.85,
    line_len_scale=2.6,
    line_w_scale=0.2,
    min_half_len=1.0,
    ball_sigma_scale=0.95,
    title=None,
    figsize=(5.2, 5.2),
    show=True,
    verbose=True,
):
    """Render a feature tensor onto width-threshold glyph decoder at high resolution."""
    conv_t = net.levels[level_idx].decoder.conv
    w_native = conv_t.weight.detach()
    cin, cout, kh, kw = w_native.shape

    if a_feat.dim() == 3:
        a0 = a_feat.unsqueeze(0)
    elif a_feat.dim() == 4:
        a0 = a_feat
    else:
        raise ValueError(f"a_feat must be 3D or 4D, got shape={tuple(a_feat.shape)}")

    if a0.shape[0] != 1:
        raise ValueError(f"Batch size must be 1 for plotting, got B={a0.shape[0]}")
    if a0.shape[1] != cin:
        raise ValueError(f"Feature channels={a0.shape[1]} != decoder in_channels={cin}")

    a0 = a0.to(device=w_native.device, dtype=w_native.dtype)

    fit_bank = fit_by_level[level_idx]
    fit_map = {int(r["filter_idx"]): r for r in fit_bank["results"] if r.get("successful_fit", False)}
    kept_idx = [int(i) for i in fit_bank["kept_idx"].tolist() if int(i) in fit_map]
    if len(kept_idx) == 0:
        raise RuntimeError(f"No fitted filters available for level {level_idx}")

    width_vals = []
    for fi in kept_idx:
        fit = fit_map[fi]
        sx = max(float(abs(fit.get("sigma_x", 1.0))), 1e-4)
        sy = max(float(abs(fit.get("sigma_y", 1.0))), 1e-4)
        width_vals.append(min(sx, sy))
    width_thr = float(np.quantile(np.asarray(width_vals, dtype=np.float32), width_thr_quantile))

    s = int(max(1, render_scale_hr))
    w_glyph_hr = torch.zeros(cin, cout, kh * s, kw * s, device=w_native.device, dtype=w_native.dtype)

    n_ball, n_line = 0, 0
    for ci in range(cin):
        if ci in fit_map:
            fit = fit_map[ci]
            sx = max(float(abs(fit.get("sigma_x", 1.0))), 1e-4)
            sy = max(float(abs(fit.get("sigma_y", 1.0))), 1e-4)
            if min(sx, sy) >= width_thr:
                n_ball += 1
            else:
                n_line += 1

            w_glyph_hr[ci] = _render_glyph_from_fit_custom(
                fit,
                out_ch=cout,
                h=kh,
                w=kw,
                width_threshold=width_thr,
                line_len_scale=line_len_scale,
                line_w_scale=line_w_scale,
                min_half_len=min_half_len,
                ball_sigma_scale=ball_sigma_scale,
                render_scale=s,
            ).to(w_native.dtype)
        else:
            w_glyph_hr[ci] = F.interpolate(
                w_native[ci].unsqueeze(0),
                scale_factor=s,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    with torch.no_grad():
        recon_hr = F.conv_transpose2d(
            a0,
            w_glyph_hr,
            bias=None,
            stride=tuple(int(v * s) for v in conv_t.stride),
            padding=tuple(int(v * s) for v in conv_t.padding),
            output_padding=tuple(int(v * s) for v in conv_t.output_padding),
            dilation=conv_t.dilation,
        )[0].detach().cpu()

    img_hr = to_vis01(recon_hr.numpy())
    if img_hr.ndim == 3 and img_hr.shape[0] in (1, 3):
        if img_hr.shape[0] == 1:
            img_hr = np.repeat(img_hr, 3, axis=0)
        img_hr = np.transpose(img_hr, (1, 2, 0))
    elif img_hr.ndim != 2:
        raise ValueError(f"Unexpected image shape for visualization: {tuple(img_hr.shape)}")
    if show:
        plt.figure(figsize=figsize)
        plt.imshow(img_hr, interpolation="nearest")
        if title is None:
            title = f"L{level_idx} glyph render (HR x{s})"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if verbose:
        print(
            f"L{level_idx} | width_thr(q={width_thr_quantile:.2f})={width_thr:.3f} | "
            f"line={n_line}, ball={n_ball} | HR={tuple(recon_hr.shape)}"
        )

    return {
        "recon_hr": recon_hr,
        "img_hr": img_hr,
        "w_glyph_hr": w_glyph_hr.detach().cpu(),
        "width_thr": width_thr,
        "n_line": n_line,
        "n_ball": n_ball,
    }


def estimate_effective_connectivity(
    net,
    loader,
    *,
    level_conn,
    weight_name,
    src_level=None,
    tgt_level=None,
    sigma_noise=0.2,
    n_iters=10,
    max_batches=8,
    gate_threshold=0.0,
    device="cpu",
):
    """Generic effective connectivity estimate for conv-like recurrent weights.

    Supports `weight_name` in {"M_inter", "M_intra", "M_intra_T"}.
    Returns dict with `raw_W`, `gate_mean`, `effective_W`, `n_batches`.
    """
    level = net.levels[int(level_conn)]
    if not hasattr(level, weight_name):
        raise AttributeError(f"Level {level_conn} has no module '{weight_name}'.")
    module = getattr(level, weight_name)
    if not hasattr(module, "weight"):
        raise AttributeError(f"Module '{weight_name}' has no 'weight' tensor.")

    W = module.weight.detach().to(device)  # [n_tgt, n_src, kh, kw]
    n_tgt, n_src, kh, kw = W.shape
    cy, cx = kh // 2, kw // 2

    if tgt_level is None:
        tgt_level = int(level_conn)
    if src_level is None:
        if weight_name == "M_intra_T":
            src_level = max(int(level_conn) - 1, 0)
        else:
            src_level = int(level_conn)

    gate_acc = torch.zeros_like(W)
    n_used = 0

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        x_clean = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_clean = x_clean.to(device)
        x_noisy = x_clean + float(sigma_noise) * torch.randn_like(x_clean)
        noise_labels = torch.full((x_noisy.shape[0],), float(sigma_noise), device=device, dtype=x_noisy.dtype)

        with torch.no_grad():
            feat = net(
                x_noisy,
                noise_labels=noise_labels,
                return_feature=True,
                infer_mode=True,
                n_iters=int(n_iters),
            )

        a_src = feat["a"][int(src_level)].detach()
        a_tgt = feat["a"][int(tgt_level)].detach()
        m_src = (a_src > float(gate_threshold)).float()
        m_tgt = (a_tgt > float(gate_threshold)).float()

        # Align source spatial size to target for local overlap products.
        if m_src.shape[-2:] != m_tgt.shape[-2:]:
            m_src = F.adaptive_max_pool2d(m_src, output_size=m_tgt.shape[-2:])

        if m_tgt.shape[1] != n_tgt:
            raise ValueError(
                f"Target channels mismatch: gates={m_tgt.shape[1]}, weight_n_tgt={n_tgt}. "
                "Pass explicit tgt_level."
            )
        if m_src.shape[1] != n_src:
            raise ValueError(
                f"Source channels mismatch: gates={m_src.shape[1]}, weight_n_src={n_src}. "
                "Pass explicit src_level."
            )

        _, _, h, w = m_tgt.shape
        for ky in range(kh):
            for kx in range(kw):
                dy = ky - cy
                dx = kx - cx

                if dy >= 0:
                    ys_src = slice(dy, h)
                    ys_tgt = slice(0, h - dy)
                else:
                    ys_src = slice(0, h + dy)
                    ys_tgt = slice(-dy, h)

                if dx >= 0:
                    xs_src = slice(dx, w)
                    xs_tgt = slice(0, w - dx)
                else:
                    xs_src = slice(0, w + dx)
                    xs_tgt = slice(-dx, w)

                src = m_src[:, :, ys_src, xs_src]  # [B, n_src, hs, ws]
                tgt = m_tgt[:, :, ys_tgt, xs_tgt]  # [B, n_tgt, hs, ws]

                denom = max(src.shape[0] * src.shape[2] * src.shape[3], 1)
                corr = torch.einsum("bcij,bdij->dc", src, tgt) / float(denom)  # [n_tgt, n_src]
                gate_acc[:, :, ky, kx] += corr

        n_used += 1

    if n_used == 0:
        raise RuntimeError("No batches processed for effective connectivity estimate.")

    gate_mean = gate_acc / float(n_used)
    effective_W = W * gate_mean
    return {
        "raw_W": W.detach().cpu(),
        "gate_mean": gate_mean.detach().cpu(),
        "effective_W": effective_W.detach().cpu(),
        "n_batches": n_used,
    }


def estimate_effective_m_intra_t_l1_to_l0(
    net,
    loader,
    *,
    level_conn=1,
    sigma_noise=0.2,
    n_iters=10,
    max_batches=8,
    gate_threshold=0.0,
    device="cpu",
):
    """Backward-compatible wrapper around the generic estimator for M_intra_T."""
    return estimate_effective_connectivity(
        net,
        loader,
        level_conn=level_conn,
        weight_name="M_intra_T",
        src_level=max(int(level_conn) - 1, 0),
        tgt_level=int(level_conn),
        sigma_noise=sigma_noise,
        n_iters=n_iters,
        max_batches=max_batches,
        gate_threshold=gate_threshold,
        device=device,
    )


def plot_grouped_filter_raw_eff(
    connectivity_raw,
    connectivity_eff,
    *,
    dictionary,
    needle_properties,
    needle_properties_anchor=None,
    dictionary_anchor=None,
    k=10,
    n_groups=2,
    color="red",
    linewidth=1.2,
    alpha_min=0.20,
    alpha_max=1.00,
    use_strength_intensity=True,
    group_titles=None,
    neuron_order=None,
    spatial_stride=1.0,
    width_thr_quantile=0.85,
    isotropic_mode="ball",
    isotropic_alpha_scale=0.35,
    isotropic_size_scale=0.95,
    source_n=None,
    kh=None,
    kw=None,
):
    if needle_properties_anchor is None:
        needle_properties_anchor = needle_properties
    if dictionary_anchor is None:
        dictionary_anchor = dictionary

    n_anchor, total = connectivity_raw.shape
    assert connectivity_eff.shape == connectivity_raw.shape, "raw/eff shapes must match"

    dict_n = int(dictionary.shape[0])

    if source_n is None:
        if kh is not None and kw is not None and int(kh) > 0 and int(kw) > 0 and (total % (int(kh) * int(kw)) == 0):
            source_n = int(total // (int(kh) * int(kw)))
        elif total % dict_n == 0:
            source_n = dict_n
        else:
            # Fallback: pick a source channel count compatible with odd square kernels
            # and closest to dictionary channels.
            candidates = []
            for k_try in range(1, 34, 2):
                area = k_try * k_try
                if total % area == 0:
                    n_try = total // area
                    candidates.append((abs(n_try - dict_n), n_try, k_try, k_try))
            if not candidates:
                source_n = dict_n
            else:
                _, source_n, kh_c, kw_c = min(candidates, key=lambda x: x[0])
                if kh is None:
                    kh = kh_c
                if kw is None:
                    kw = kw_c

    source_n = int(source_n)
    if kh is None or kw is None:
        if source_n <= 0 or (total % source_n != 0):
            raise ValueError(
                f"Cannot decode flattened connectivity shape={tuple(connectivity_raw.shape)}. "
                f"Pass explicit source_n/kh/kw."
            )
        hw = total // source_n
        kh = int(round(np.sqrt(hw)))
        kw = hw // max(kh, 1)
        if kh * kw != hw:
            kh = 1
            kw = hw

    kh = int(kh)
    kw = int(kw)
    if source_n * kh * kw != total:
        raise ValueError(
            f"Flatten decode mismatch: source_n*kh*kw={source_n*kh*kw}, total={total}. "
            f"Pass correct source_n/kh/kw."
        )
    assert n_anchor <= dictionary_anchor.shape[0], "Anchor dictionary must cover all anchor rows."

    if neuron_order is None:
        order = np.arange(n_anchor)
    else:
        order = np.asarray(neuron_order, dtype=np.int64)
        assert order.shape[0] == n_anchor, "neuron_order must include all anchor neurons exactly once"

    groups = [g.tolist() for g in np.array_split(order, int(n_groups)) if len(g) > 0]
    if group_titles is None:
        group_titles = [f"neurons group {gi + 1}" for gi in range(len(groups))]

    sx = float(spatial_stride)
    sy = float(spatial_stride)

    width_thr_conn = compute_width_threshold(
        needle_properties,
        range(min(source_n, dict_n)),
        quantile=width_thr_quantile,
    )
    width_thr_anchor = compute_width_threshold(
        needle_properties_anchor,
        range(dictionary_anchor.shape[0]),
        quantile=width_thr_quantile,
    )
    same_level = (dictionary_anchor.shape[0] == dictionary.shape[0]) and (needle_properties_anchor is needle_properties)

    def _draw_conn_row(ax, row_flat, anchor_neuron):
        x_pad = 1.0 + 0.5 * (kw - 1) * sx
        y_pad = 1.0 + 0.5 * (kh - 1) * sy
        ax.set_xlim([-x_pad, (dictionary.shape[-1] - 1) + x_pad])
        ax.set_ylim([-y_pad, (dictionary.shape[-2] - 1) + y_pad])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor([0.5, 0.5, 0.5])

        topk = row_flat.topk(k + 8, largest=True)
        topk_idx = topk.indices
        topk_val = topk.values
        conn_j, yy, xx = torch.unravel_index(topk_idx, (source_n, kh, kw))

        if use_strength_intensity:
            w_abs = topk_val.abs().detach().cpu().numpy().astype(np.float32)
            wmn, wmx = float(w_abs.min()), float(w_abs.max())
            w_norm = (w_abs - wmn) / (wmx - wmn) if (wmx - wmn) > 1e-8 else np.ones_like(w_abs)
        else:
            w_norm = np.ones((len(topk_idx),), dtype=np.float32)

        plotted = 0
        for jj, oy, ox, wi in zip(conn_j.tolist(), yy.tolist(), xx.tolist(), w_norm.tolist()):
            if same_level and jj == anchor_neuron and oy == (kh // 2) and ox == (kw // 2):
                continue
            if jj >= dict_n or (jj not in needle_properties):
                continue

            res = needle_properties[jj]
            x0 = np.asarray(res["x"], dtype=np.float32)
            y0 = np.asarray(res["y"], dtype=np.float32)
            res_x = x0 + (float(ox) - float(kw // 2)) * sx
            res_y = y0 + (float(oy) - float(kh // 2)) * sy
            alpha_i = float(alpha_min + (alpha_max - alpha_min) * wi)

            draw_needle_glyph(
                ax,
                res,
                x_line=res_x,
                y_line=res_y,
                color=color,
                alpha=alpha_i,
                linewidth=linewidth,
                width_threshold=width_thr_conn,
                isotropic_mode=isotropic_mode,
                isotropic_alpha_scale=isotropic_alpha_scale,
                isotropic_size_scale=isotropic_size_scale,
            )

            plotted += 1
            if plotted >= k:
                break

        if anchor_neuron in needle_properties_anchor:
            r0 = needle_properties_anchor[anchor_neuron]
            draw_needle_glyph(
                ax,
                r0,
                color="black",
                alpha=float(np.clip(0.75 * float(isotropic_alpha_scale) + 0.25, 0.0, 1.0)),
                linewidth=1.7,
                width_threshold=width_thr_anchor,
                isotropic_mode=isotropic_mode,
                isotropic_alpha_scale=1.0,
                isotropic_size_scale=isotropic_size_scale,
                size_override=0.22,
            )

    for gi, neuron_group in enumerate(groups):
        n_cols = len(neuron_group)
        fig, axes = plt.subplots(3, n_cols, figsize=(1.35 * n_cols, 4.2), squeeze=False)

        for ci, n in enumerate(neuron_group):
            ax0 = axes[0, ci]
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_aspect("equal")
            img, cmap = to_vis_img(dictionary_anchor[n])
            if cmap is None:
                ax0.imshow(img, origin="lower")
            else:
                ax0.imshow(img, cmap=cmap, origin="lower", vmin=float(dictionary_anchor.min()), vmax=float(dictionary_anchor.max()))
            ax0.set_title(f"n{n}", fontsize=7)

            _draw_conn_row(axes[1, ci], connectivity_raw[n], n)
            _draw_conn_row(axes[2, ci], connectivity_eff[n], n)

        axes[0, 0].set_ylabel("anchor", fontsize=8)
        axes[1, 0].set_ylabel("raw", fontsize=8)
        axes[2, 0].set_ylabel("eff", fontsize=8)

        fig.suptitle(
            f"Grouped connectivity | {group_titles[gi]} | top-{k} | width_thr_conn={width_thr_conn:.3f}, width_thr_anchor={width_thr_anchor:.3f}",
            fontsize=11,
        )
        plt.tight_layout()
        plt.show()
