"""
run_harmonic_basis.py
=====================
Compute and visualise the top-k eigenvectors of the denoiser Jacobian
  J_D = d D(x_noisy) / d x_noisy
for both neural_sheet7 (face) and UNet baseline.

These eigenvectors are the "harmonic basis" — directions in pixel space where
the model is most confident about the denoised signal.

Algorithm: Randomized SVD via VJP only (no full Jacobian stored).
  J_D is symmetric for a well-trained denoiser (it equals the posterior
  covariance / sigma^2), so VJP == JVP and we only need backward passes.

Approx cost: 2*(K+OVERSAMPLE)*N_POWER_ITER backward passes per model.

Outputs
-------
figures/harmonic_basis_neural_sheet7.png
figures/harmonic_basis_UNet.png
figures/harmonic_basis_comparison.png   (side-by-side top row)
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from recurrent_diffusion_pkg.utils import load_lit_model, build_loaders_from_config


def parse_args():
    p = argparse.ArgumentParser(description="Compute harmonic basis (Jacobian eigenvectors) of denoiser")
    p.add_argument("--unet_dir", type=str,
                   default="pretrained_model/scaling-new-face/00029_layer1_unet_small_edm_large_noattn",
                   help="Path to UNet model directory")
    p.add_argument("--sheet_dir", type=str,
                   default="pretrained_model/scaling-new-face/00024_simplify_layer1_neural_sheet7_intra_film_scalar_mul_ff_scale_large_large_noise",
                   help="Path to neural_sheet7 model directory")
    p.add_argument("--unet_only", action="store_true",
                   help="Only run UNet (skip neural_sheet7 and comparison plot)")
    p.add_argument("--out_dir",       type=str,   default="figures")
    p.add_argument("--noise_level",   type=float, default=0.3)
    p.add_argument("--n_iters_sheet", type=int,   default=4)
    p.add_argument("--k",             type=int,   default=100,  help="Number of eigenvectors")
    p.add_argument("--k_plot_stride", type=int,   default=4,    help="Plot every Nth eigenvector")
    p.add_argument("--n_oversampling",type=int,   default=10)
    p.add_argument("--n_power_iter",  type=int,   default=10)
    p.add_argument("--val_seed",      type=int,   default=0)
    p.add_argument("--val_img_idx",   type=int,   default=2,    help="Which image from val batch to use")
    return p.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Randomized SVD via VJP ────────────────────────────────────────────────────

def apply_JD(model_fn, x, V_cols):
    """
    Compute J_D^T · v for each column of V_cols ∈ R^{n × l} via VJP.
    Since J_D is symmetric, J_D^T · v = J_D · v.
    Returns matrix of shape (n, l).
    """
    n, l = V_cols.shape
    results = []
    for i in range(l):
        v = V_cols[:, i].view_as(x)
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            out   = model_fn(x_req)
            grad  = torch.autograd.grad(out, x_req, grad_outputs=v,
                                        retain_graph=False)[0]
        results.append(grad.detach().view(-1))
    return torch.stack(results, dim=1)   # (n, l)


def randomized_svd_denoiser(model_fn, x_noisy, k=20, n_oversampling=10,
                             n_power_iter=2):
    """
    Top-k eigenvectors of J_D = d D(x) / dx using randomized SVD.

    Steps
    -----
    1. Sketch:  Y = J_D · Omega   (l VJP calls, Omega random)
    2. Power iteration on Y for accuracy
    3. QR:      Q = orth(Y)
    4. Project: B = J_D · Q       (l VJP calls)
    5. SVD of small matrix B^T ∈ R^{l×n}  →  reconstruct U, S, V

    Returns
    -------
    eigvecs : (n, k) – columns are the top-k eigenvectors (right singular vectors)
    eigvals : (k,)   – corresponding singular values (≈ eigenvalues for sym J_D)
    """
    x = x_noisy.detach()
    n = x.numel()
    l = k + n_oversampling

    # Step 1: initial sketch
    Omega = torch.randn(n, l, device=x.device, dtype=x.dtype)
    Y = apply_JD(model_fn, x, Omega)            # (n, l)

    # Step 2: power iteration  Y ← (J_D)^2 · Q  repeated
    for _ in range(n_power_iter):
        Q, _ = torch.linalg.qr(Y)
        Y = apply_JD(model_fn, x, Q)            # J_D · Q
        Q, _ = torch.linalg.qr(Y)
        Y = apply_JD(model_fn, x, Q)            # J_D^2 · Q

    # Step 3: orthonormal basis of the range
    Q, _ = torch.linalg.qr(Y)                   # (n, l)

    # Step 4: project J_D onto the sketch
    B = apply_JD(model_fn, x, Q)                # (n, l) = J_D · Q
    # B ≈ J_D · Q  →  J_D ≈ B · Q^T  (for sym J_D)

    # Step 5: SVD of small (n, l) → use (l, n) for efficiency
    Bt = B.T.contiguous()                        # (l, n)
    U_hat, S, Vh = torch.linalg.svd(Bt, full_matrices=False)  # U_hat:(l,l), S:(l,), Vh:(l,n)

    # Reconstruct left singular vectors in full space
    # J_D ≈ Q · U_hat · diag(S) · Vh
    # Right singular vectors V = Vh^T
    V = Vh.T                                     # (n, l)

    return V[:, :k], S[:k]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_eigvecs(clean, eigvecs, eigvals, label, out_path, img_shape,
                 stride=1, noise_level=0.3):
    """
    Grid: clean image in slot 0, eigenvectors sampled every `stride` in slots 1+.
    eigvecs: (n, K)  eigvals: (K,)  img_shape: (C, H, W)
    """
    C, H, W = img_shape
    indices = list(range(0, eigvecs.shape[1], stride))[:24]  # cap at 24 so grid is 5×5
    k_plot  = len(indices)

    ncols = 5
    nrows = 5   # 1 clean + 24 eigenvectors = 25 slots
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axs = axs.ravel()

    # Slot 0: clean image
    img_np = clean.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img_np = (img_np - img_np.min()) / max(img_np.max() - img_np.min(), 1e-8)
    if C == 1:
        axs[0].imshow(img_np[:, :, 0], cmap="gray")
    else:
        axs[0].imshow(img_np.clip(0, 1))
    axs[0].set_title("Clean", fontsize=11)
    axs[0].axis("off")

    # Slots 1..k_plot: strided eigenvectors
    for slot, i in enumerate(indices):
        ev = eigvecs[:, i].cpu().numpy().reshape(H, W) if C == 1 else \
             eigvecs[:, i].cpu().numpy().reshape(C, H, W).mean(0)
        vmax = np.percentile(np.abs(ev), 98)
        axs[slot + 1].imshow(ev, cmap="RdBu",
                             norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
        axs[slot + 1].set_title(f"λ{i+1}={eigvals[i].item():.3f}", fontsize=10)
        axs[slot + 1].axis("off")

    for j in range(k_plot + 1, len(axs)):
        axs[j].axis("off")

    fig.suptitle(f"Harmonic basis — {label}  (σ={noise_level})", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_comparison(results, out_path, noise_level=0.3):
    """
    Side-by-side: each model gets one row, showing clean + top-8 eigenvectors.
    results: list of (label, clean, eigvecs, eigvals, img_shape)
    """
    n_show  = 8
    ncols   = n_show + 1
    nrows   = len(results)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 2.8))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, (label, clean, eigvecs, eigvals, img_shape) in enumerate(results):
        C, H, W = img_shape

        # Clean image
        img_np = clean.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        img_np = (img_np - img_np.min()) / max(img_np.max() - img_np.min(), 1e-8)
        if C == 1:
            axes[row, 0].imshow(img_np[:, :, 0], cmap="gray")
        else:
            axes[row, 0].imshow(img_np.clip(0, 1))
        axes[row, 0].set_title("Clean", fontsize=10)
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold",
                                rotation=90, labelpad=6)
        axes[row, 0].axis("off")

        for col in range(n_show):
            ev = eigvecs[:, col].cpu().numpy()
            ev = ev.reshape(H, W) if C == 1 else ev.reshape(C, H, W).mean(0)
            vmax = np.percentile(np.abs(ev), 98)
            axes[row, col + 1].imshow(ev, cmap="RdBu",
                                      norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
            axes[row, col + 1].set_title(f"λ{col+1}={eigvals[col].item():.3f}",
                                          fontsize=9)
            axes[row, col + 1].axis("off")

    fig.suptitle(f"Harmonic basis comparison  (σ={noise_level})", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model_and_fn(model_dir, noise_level, device, n_iters_sheet=4):
    """Load model, return (config, model_fn, lit_model)."""
    import json
    config, lit = load_lit_model(model_dir, device=device)
    arch = config["model_arch"]
    net  = lit.ema_model.eval()

    if arch == "neural_sheet7":
        for lvl in net.levels:
            lvl.reset_wnorm()
        sigma = torch.full((1,), noise_level, device=device, dtype=torch.float32)
        def model_fn(x):
            return net(x, noise_labels=sigma, infer_mode=True,
                       n_iters=n_iters_sheet, n_iters_grad=2)
    elif arch == "unet":
        sigma = torch.full((1,), noise_level, device=device, dtype=torch.float32)
        def model_fn(x):
            return net(x, sigma)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    print(f"  Loaded {arch} from {model_dir.name}")
    return config, model_fn, net


def main():
    args = parse_args()
    torch.manual_seed(args.val_seed)

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)

    UNET_DIR  = Path(args.unet_dir)
    SHEET_DIR = Path(args.sheet_dir)

    model_configs = [("UNet", UNET_DIR)] if args.unet_only else [
        ("UNet",          UNET_DIR),
        ("neural_sheet7", SHEET_DIR),
    ]

    # Load val image from UNet config (works whether unet_only or not)
    print("Loading val image...")
    import json
    cfg0 = json.load(open(UNET_DIR / "config.json"))
    _, _, val_loader, _ = build_loaders_from_config(cfg0, batch_size=8, num_workers=2)
    clean_batch, _ = next(iter(val_loader))
    clean = clean_batch[[args.val_img_idx]].to(DEVICE)
    noisy = clean + torch.randn_like(clean) * args.noise_level

    comparison_results = []

    for label, model_dir in model_configs:
        print(f"\n{'='*60}\n  [{label}]")
        config, model_fn, net = load_model_and_fn(model_dir, args.noise_level, DEVICE,
                                                   n_iters_sheet=args.n_iters_sheet)

        img_size  = config.get("img_size", 64)
        grayscale = config.get("grayscale", False)
        C = 1 if grayscale else 3
        img_shape = (C, img_size, img_size)

        noisy_m = noisy[:, :C].to(DEVICE)
        if noisy_m.shape[-1] != img_size:
            import torch.nn.functional as F
            noisy_m = F.interpolate(noisy_m, size=(img_size, img_size), mode="bilinear")

        l_total = args.k + args.n_oversampling
        print(f"  Input: {noisy_m.shape}  "
              f"Running randomized SVD (k={args.k}, oversampling={args.n_oversampling}, "
              f"power_iters={args.n_power_iter})")
        print(f"  Total VJP calls ≈ {2 * l_total * (args.n_power_iter * 2 + 2)}")

        eigvecs, eigvals = randomized_svd_denoiser(
            model_fn, noisy_m,
            k=args.k, n_oversampling=args.n_oversampling, n_power_iter=args.n_power_iter,
        )
        print(f"  eigvals range: {eigvals[-1].item():.4f} … {eigvals[0].item():.4f}")

        clean_m = clean[:, :C]
        if clean_m.shape[-1] != img_size:
            import torch.nn.functional as F
            clean_m = F.interpolate(clean_m, size=(img_size, img_size), mode="bilinear")

        plot_eigvecs(clean_m, eigvecs, eigvals, label,
                     OUT_DIR / f"harmonic_basis_{label.replace(' ', '_')}.png",
                     img_shape, stride=args.k_plot_stride, noise_level=args.noise_level)
        comparison_results.append((label, clean_m, eigvecs, eigvals, img_shape))

    if not args.unet_only:
        plot_comparison(comparison_results, OUT_DIR / "harmonic_basis_comparison.png",
                        noise_level=args.noise_level)
    print("\nAll done.")


if __name__ == "__main__":
    main()
