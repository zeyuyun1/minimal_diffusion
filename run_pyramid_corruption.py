"""
run_pyramid_corruption.py
=========================
Corrupt shape stimuli selectively in the Laplacian pyramid, then observe how
the model repairs the contour as the noise label increases.

Pipeline per stimulus
---------------------
1. Encode clean image -> Laplacian pyramid  [pyr[0]=fine, ..., pyr[-1]=coarse]
2. Detect boundary pixels via morphological gradient.
3. Zero out selected pyramid levels inside the boundary mask -> decode back to
   pixel space.  This removes edge information at the chosen scale(s) while
   leaving global structure intact at coarser scales.
4. Feed the corrupted image (no noise added) into the network, sweeping the
   noise label from low to high.

Output
------
figures/pyramid_corruption_<tag>.png
  rows = stimuli
  cols = [original | corrupted(pyr0) | boundary mask | sigma=... | ...]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from recurrent_diffusion_pkg.utils import load_lit_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str,
                   default="pretrained_model/scaling-VH-new-2/"
                           "00008_simple_sheet7_simple_control_small_noise_long_iter")
    p.add_argument("--stim_dir",   type=str, default="stimulus",
                   help="Directory with shape_*.png stimuli")
    p.add_argument("--out_dir",    type=str, default="figures")
    p.add_argument("--tag",        type=str, default="00008")
    p.add_argument("--n_iters",    type=int, default=8)
    p.add_argument("--noise_labels", type=float, nargs="+",
                   default=[0.1, 0.3, 0.6, 1.0, 1.5, 2.0])
    p.add_argument("--corrupt_levels", type=int, nargs="+", default=[0],
                   help="Which pyramid levels to zero out at boundary (0=finest)")
    p.add_argument("--boundary_dilation", type=int, default=3,
                   help="Dilation radius (px) for the boundary mask")
    p.add_argument("--panel_size", type=float, default=1.6)
    return p.parse_args()


def build_boundary_mask(img_np, dilation=3):
    """
    Binary boundary mask from a grayscale [0,1] image.
    Returns (H, W) float32 array, 1 where boundary is.
    """
    gray = (img_np * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=dilation)
    eroded  = cv2.erode(gray,  kernel, iterations=dilation)
    boundary = np.clip(dilated.astype(np.int16) - eroded.astype(np.int16), 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(boundary, 10, 255, cv2.THRESH_BINARY)
    return (mask / 255.0).astype(np.float32)


def corrupt_pyramid(net, img_t, corrupt_levels, boundary_dilation, device):
    """
    Encode img_t, zero out boundary in selected pyramid levels, decode back.
    Returns (corrupted tensor (1,C,H,W), mask_hw ndarray).
    """
    img_t = img_t.unsqueeze(0).to(device)
    H, W = img_t.shape[-2:]

    img_np = img_t[0, 0].cpu().float().numpy()
    mask_hw = build_boundary_mask(img_np, dilation=boundary_dilation)

    with torch.no_grad():
        pyr = net.encoder_0(img_t)

    pyr_corrupted = list(pyr)
    for lvl_idx in corrupt_levels:
        if lvl_idx >= len(pyr_corrupted):
            continue
        feat = pyr_corrupted[lvl_idx]
        _, _, H_l, W_l = feat.shape
        mask_t = torch.from_numpy(mask_hw).to(device).unsqueeze(0).unsqueeze(0)
        if (H_l, W_l) != (H, W):
            mask_t = F.interpolate(mask_t, size=(H_l, W_l), mode="nearest")
        pyr_corrupted[lvl_idx] = feat * (1.0 - mask_t)

    with torch.no_grad():
        corrupted = net.decoder_0(pyr_corrupted)

    return corrupted.clamp(0, 1), mask_hw


def to_display(t, n_channels):
    t = t.cpu().float()
    t = (t - t.min()) / max((t.max() - t.min()).item(), 1e-8)
    t = t.clamp(0, 1)
    if n_channels == 1:
        return t[0].numpy(), "gray"
    return t.permute(1, 2, 0).numpy(), None


def main():
    args = parse_args()
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {Path(args.model_dir).name} ...")
    config, lit = load_lit_model(Path(args.model_dir), device=DEVICE, ckpt_name=None)
    net = lit.ema_model.eval()
    for lvl in net.levels:
        lvl.reset_wnorm()

    import torch.nn as _nn
    _first_conv = next(m for m in net.levels[0].modules() if isinstance(m, _nn.Conv2d))
    n_channels = _first_conv.in_channels
    print(f"  n_channels={n_channels}, pyramid levels={net.n_levels}")

    stim_paths = sorted(Path(args.stim_dir).glob("shape_*.png"))
    if n_channels == 1:
        tf = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
    else:
        tf = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")),
                                  transforms.ToTensor()])

    stimuli = [(p.stem, tf(Image.open(p))) for p in stim_paths]
    n_stim  = len(stimuli)
    print(f"  {n_stim} stimuli")

    print(f"Corrupting pyramid levels {args.corrupt_levels} "
          f"(dilation={args.boundary_dilation}) ...")
    corrupted_imgs = []
    masks = []
    for name, t in stimuli:
        c_t, mask = corrupt_pyramid(net, t, args.corrupt_levels,
                                    args.boundary_dilation, DEVICE)
        corrupted_imgs.append(c_t.squeeze(0).cpu())
        masks.append(mask)

    noise_labels = args.noise_labels
    n_labels = len(noise_labels)

    batch = torch.stack(corrupted_imgs).to(DEVICE)
    results = []
    for sigma_val in noise_labels:
        sigma = torch.full((n_stim,), sigma_val, device=DEVICE, dtype=batch.dtype)
        print(f"  sigma_label={sigma_val:.2f} ...", end=" ", flush=True)
        with torch.no_grad():
            history = net(batch, noise_labels=sigma,
                          infer_mode=True, n_iters=args.n_iters, return_history=True)
        results.append(history[-1]["denoised"].cpu())
        print("done")

    n_extra = 3   # original, corrupted, mask
    n_cols  = n_extra + n_labels
    cell    = args.panel_size

    fig, axes = plt.subplots(n_stim, n_cols,
                              figsize=(n_cols * cell, n_stim * cell),
                              facecolor="white")
    if n_stim == 1:
        axes = axes[np.newaxis, :]

    lvl_str = "+".join(str(l) for l in args.corrupt_levels)
    col_titles = (["original", f"corrupted\n(pyr lvl {lvl_str})", "boundary\nmask"]
                  + [f"sigma={v:.2f}" for v in noise_labels])

    for row, (stim_name, stim_t) in enumerate(stimuli):
        arr, cmap = to_display(stim_t, n_channels)
        axes[row, 0].imshow(arr, cmap=cmap, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(stim_name.replace("shape_", ""),
                                fontsize=6, rotation=0, labelpad=60,
                                va="center", color="#333333")

        arr_c, _ = to_display(corrupted_imgs[row], n_channels)
        axes[row, 1].imshow(arr_c, cmap=cmap, vmin=0, vmax=1)

        axes[row, 2].imshow(masks[row], cmap="gray", vmin=0, vmax=1)

        for col, denoised_batch in enumerate(results):
            arr_d, _ = to_display(denoised_batch[row], n_channels)
            axes[row, col + n_extra].imshow(arr_d, cmap=cmap, vmin=0, vmax=1)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=7, pad=3, color="#333333")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#dddddd")
            sp.set_linewidth(0.5)

    fig.suptitle(
        f"Pyramid corruption (levels {args.corrupt_levels}, dilation={args.boundary_dilation})"
        f"  ->  noise label sweep  |  {Path(args.model_dir).name}",
        fontsize=8, y=1.002, color="#222222",
    )
    plt.tight_layout(pad=0.3)

    out_path = OUT_DIR / f"pyramid_corruption_{args.tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
