"""
run_simplicity_bias.py
======================
Sort images from the training set by estimated log likelihood under the UNet
denoiser.

Estimator
---------
The denoiser Jacobian J_D = d D(x_σ, σ)/d x_σ is related to the score:
    s(x, σ) = (D(x, σ) - x) / σ²
    ∇·s      = (tr(J_D) - n) / σ²

Integrating the instantaneous log-det (probability flow ODE / change of
variables) over noise levels gives:

    log p(x) ≈ ∫ (tr(J_D(x_σ, σ)) - n) / σ²  ·  w(σ) dσ  +  const

For ranking we drop the constant and use a uniform average over K log-spaced
σ levels. We estimate tr(J_D) per noise level via the Hutchinson estimator:

    tr(J) ≈ (1/m) Σ_i  vᵢᵀ Jᵀ vᵢ  ,  vᵢ ~ N(0,I)

with Jᵀ v computed as a VJP (one backward pass per v).

Outputs
-------
figures/simplicity_bias_grid.png   – top-25 / mid-25 / bottom-25 images
figures/simplicity_bias_scores.png – histogram of scores
"""

from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recurrent_diffusion_pkg.utils import load_lit_model, build_loaders_from_config

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("pretrained_model/scaling-VH-new-2/"
                  "00033_layer1_unet_small_edm_noatt")
OUT_DIR    = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

SIGMA_MIN   = 0.05
SIGMA_MAX   = 2.0
N_SIGMA     = 8       # noise levels to average over
N_HUTCH     = 4       # Hutchinson vectors per noise level
N_IMAGES    = 500     # how many training images to score
BATCH_SIZE  = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Hutchinson trace estimator ────────────────────────────────────────────────

def hutchinson_trace(model_fn, x_noisy, n_vecs=4):
    """
    Estimate tr(d D(x)/d x) at x=x_noisy via Hutchinson.

    For each probe vector v ~ N(0,I):
        tr(J) ≈ vᵀ Jᵀ v   (VJP gives Jᵀ v; equals J v for symmetric J)
    Returns scalar estimate.
    """
    estimates = []
    for _ in range(n_vecs):
        v = torch.randn_like(x_noisy)
        with torch.enable_grad():
            x_req = x_noisy.detach().clone().requires_grad_(True)
            out   = model_fn(x_req)            # (1, C, H, W)
            vjp   = torch.autograd.grad(out, x_req, grad_outputs=v,
                                        retain_graph=False)[0]
        estimates.append((v * vjp).sum().item())
    return float(np.mean(estimates))


# ── Score one image ────────────────────────────────────────────────────────────

def score_image(model_fn, x_clean, sigmas, n_hutch):
    """
    Compute Σ_σ  (tr(J_D(x_σ,σ)) - n) / σ²  averaged over σ levels.
    x_clean: (1, C, H, W)
    Returns scalar score (higher = simpler / higher likelihood).
    """
    n = x_clean.numel()
    total = 0.0
    for sigma in sigmas:
        noise = torch.randn_like(x_clean) * sigma
        x_noisy = (x_clean + noise).detach()
        tr_J = hutchinson_trace(model_fn, x_noisy, n_vecs=n_hutch)
        total += (tr_J - n) / (sigma ** 2)
    return total / len(sigmas)


# ── Plotting ──────────────────────────────────────────────────────────────────

def show_img(ax, t, title="", fontsize=8):
    """Display a (1,C,H,W) or (C,H,W) tensor."""
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
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


def plot_grid(images, scores, title, out_path, n_cols=25):
    """Plot a row of images with their scores."""
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(n * 1.6, 2.2))
    for i, (img, sc) in enumerate(zip(images, scores)):
        show_img(axs[i], img, f"{sc:.0f}", fontsize=7)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_summary(all_images, all_scores, out_path, n_show=25):
    """
    3-row grid: top-n_show (simplest), middle-n_show, bottom-n_show.
    """
    order  = np.argsort(all_scores)[::-1]   # descending: high score = simple
    n_total = len(order)
    mid_start = (n_total - n_show) // 2

    top_idx  = order[:n_show]
    mid_idx  = order[mid_start: mid_start + n_show]
    bot_idx  = order[-n_show:]

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 1.6, 5.5))
    row_labels = [
        f"Simplest (top {n_show})",
        f"Middle",
        f"Most complex (bottom {n_show})",
    ]
    for row, (idxs, label) in enumerate(zip([top_idx, mid_idx, bot_idx], row_labels)):
        for col, idx in enumerate(idxs):
            show_img(axes[row, col], all_images[idx],
                     f"{all_scores[idx]:.0f}", fontsize=6)
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold",
                                rotation=90, labelpad=4)

    fig.suptitle("Simplicity bias — sorted by denoiser Jacobian trace\n"
                 "(higher score = higher estimated log likelihood = simpler)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_score_hist(all_scores, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(all_scores, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(all_scores), color="tomato", lw=1.5, label="median")
    ax.set_xlabel("Jacobian trace score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of simplicity scores", fontsize=12)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Load model
    print("Loading model...")
    config, lit = load_lit_model(MODEL_DIR, device=DEVICE)
    net = lit.ema_model.eval().to(DEVICE)
    sigma_t = torch.zeros(1, device=DEVICE, dtype=torch.float32)

    def model_fn(x):
        s = sigma_t.expand(x.shape[0])
        return net(x, s)

    # Log-spaced sigma levels
    sigmas = np.exp(np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), N_SIGMA)).tolist()
    print(f"  σ levels: {[f'{s:.3f}' for s in sigmas]}")

    # Load training images
    print("Loading training data...")
    _, train_loader, _, _ = build_loaders_from_config(
        config, batch_size=BATCH_SIZE, num_workers=2)

    all_images = []
    all_scores = []

    print(f"Scoring {N_IMAGES} images "
          f"(K={N_SIGMA} σ levels × m={N_HUTCH} Hutchinson vecs)...")

    count = 0
    for batch_imgs, _ in train_loader:
        for img in batch_imgs:
            if count >= N_IMAGES:
                break
            x = img.unsqueeze(0).to(DEVICE)

            # Update sigma closure for each level inside score_image
            sc = _score_image_unet(net, x, sigmas, N_HUTCH, DEVICE)
            all_images.append(img.cpu())
            all_scores.append(sc)
            count += 1
            if count % 50 == 0:
                print(f"  {count}/{N_IMAGES}  last score={sc:.1f}")
        if count >= N_IMAGES:
            break

    all_scores = np.array(all_scores)
    print(f"\nScores: min={all_scores.min():.1f}  max={all_scores.max():.1f}  "
          f"median={np.median(all_scores):.1f}")

    plot_summary(all_images, all_scores,
                 OUT_DIR / "simplicity_bias_grid.png", n_show=25)
    plot_score_hist(all_scores, OUT_DIR / "simplicity_bias_scores.png")
    print("\nAll done.")


def _score_image_unet(net, x_clean, sigmas, n_hutch, device):
    """Score one image across multiple σ levels."""
    n = x_clean.numel()
    total = 0.0
    for sigma in sigmas:
        s_t = torch.full((1,), sigma, device=device, dtype=torch.float32)
        noise = torch.randn_like(x_clean) * sigma
        x_noisy = (x_clean + noise).detach()

        def model_fn(x):
            return net(x, s_t)

        tr_J = hutchinson_trace(model_fn, x_noisy, n_vecs=n_hutch)
        total += (tr_J - n) / (sigma ** 2)
    return total / len(sigmas)


if __name__ == "__main__":
    main()
