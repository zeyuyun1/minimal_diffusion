"""
run_model_comparison.py
=======================
Compare neural_sheet7 vs. UNet denoiser/generator on the validation set.

Outputs
-------
figures/comparison_psnr_curve.png   – PSNR vs. σ (log–log) for both models
figures/comparison_generation.png   – side-by-side random samples (UNet | sheet7)

Config
------
Edit the paths at the top of the file.  Set UNET_DIR = None to skip UNet.
"""

from pathlib import Path
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from recurrent_diffusion_pkg.utils import load_lit_model, build_loaders_from_config

# ── Experiment paths ──────────────────────────────────────────────────────────
SHEET_DIR = Path("/home/zeyu/recurrent_diffusion_minimal/pretrained_model/scaling-new-face/00026_simplify_layer1_neural_sheet7_intra_film_scalar_mul_ff_scale_large_large_noise_simple_control")
UNET_DIR  = Path("/home/zeyu/recurrent_diffusion_minimal/pretrained_model/scaling-new-face/00006_layer1_unet_small_edm_large_noattn")

# ── Denoising eval settings ───────────────────────────────────────────────────
NOISE_LEVELS   = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
N_ITERS_SHEET  = 8      # recurrent iters for neural_sheet7
MAX_BATCHES    = 20     # val batches to average over (None = all)
BATCH_SIZE     = 16

# ── Generation settings ───────────────────────────────────────────────────────
N_SAMPLES        = 12
SIGMA_MAX        = 4.0
SIGMA_MIN        = 0.1
N_SIGMA_STEPS    = 40
RHO              = 7.0   # EDM polynomial schedule exponent
N_ITERS_DENOISE  = 4     # recurrent iters during generation
RECORD_EVERY     = 2
N_MEAN_BATCHES   = 100   # batches to average for mean_img start point

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Model wrapper: unified denoiser interface
# ─────────────────────────────────────────────────────────────────────────────

class DenoiserWrapper:
    """
    Wraps both neural_sheet7 and UNet so callers use a single API:
        wrapper.denoise(x_noisy, sigma_scalar)  ->  x_hat (B,C,H,W)
    """
    def __init__(self, lit_model, model_arch: str, n_iters: int = 8):
        self.net       = lit_model.ema_model.eval()
        self.arch      = model_arch
        self.n_iters   = n_iters

    @torch.no_grad()
    def denoise(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        B = x.shape[0]
        sigma_t = torch.full((B,), float(sigma), device=x.device, dtype=x.dtype)
        if self.arch == "neural_sheet7":
            return self.net(
                x, noise_labels=sigma_t,
                infer_mode=True, n_iters=self.n_iters,
            )
        elif self.arch == "unet":
            # UNetModel.forward(x, timesteps) — timesteps == sigma
            return self.net(x, sigma_t)
        else:
            raise ValueError(f"Unknown arch: {self.arch}")

    def __call__(self, x, noise_labels, infer_mode=False, n_iters=None, **kwargs):
        """Drop-in replacement for passing the wrapper as `model` arg to annealed_heun."""
        if n_iters is None:
            n_iters = self.n_iters
        orig_iters, self.n_iters = self.n_iters, n_iters
        out = self.denoise(x, float(noise_labels[0].item()))
        self.n_iters = orig_iters
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Annealed Heun sampler (works for both model types via DenoiserWrapper)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def annealed_heun_dynamics(
    model,          # DenoiserWrapper (or any callable matching neural_sheet7 API)
    shape,
    sigmas,
    n_iters_denoiser=8,
    device="cuda",
    record_per_steps=1,
    start_img=None,
):
    """
    Deterministic Heun (improved-Euler) sampler.

    model: must expose model(x, noise_labels=sigma_tensor, infer_mode=True, n_iters=K)
           DenoiserWrapper satisfies this for both neural_sheet7 and UNet.

    sigmas: 1-D tensor, descending, NOT including 0 (model not trained at 0).
    """
    B, C, H, W = shape
    device = torch.device(device)
    sigmas = torch.as_tensor(sigmas, device=device, dtype=torch.float32)

    x = (torch.randn(shape, device=device) * sigmas[0]
         if start_img is None else start_img.to(device=device, dtype=torch.float32))

    x_hist, k = [], 0

    def _denoise(x_, s_):
        sig = torch.full((B,), float(s_), device=device, dtype=torch.float32)
        return model(x_, noise_labels=sig, infer_mode=True, n_iters=n_iters_denoiser)

    for i in range(len(sigmas) - 1):
        t_cur, t_next = sigmas[i], sigmas[i + 1]

        den_cur = _denoise(x, t_cur)
        d_cur   = (x - den_cur) / t_cur

        x_euler = x + (t_next - t_cur) * d_cur

        den_next = _denoise(x_euler, t_next)
        d_prime  = (x_euler - den_next) / t_next

        x = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        k += 1
        if record_per_steps and (k % record_per_steps == 0):
            x_hist.append(x.detach().cpu())

    # Final denoise at sigma_min
    x = _denoise(x, sigmas[-1])
    x_hist.append(x.detach().cpu())
    x_hist = torch.stack(x_hist, dim=0) if x_hist else torch.empty((0,))
    return x, x_hist


# ─────────────────────────────────────────────────────────────────────────────
#  PSNR utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(x_hat: torch.Tensor, x_clean: torch.Tensor, peak: float = 1.0) -> float:
    mse = ((x_hat - x_clean) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(peak ** 2 / mse)


# ─────────────────────────────────────────────────────────────────────────────
#  Denoising evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_denoising_psnr(wrapper: DenoiserWrapper, val_loader, noise_levels, max_batches=None):
    """
    For each σ in noise_levels, add Gaussian noise to val images, denoise,
    and return mean PSNR (also returns noisy-image PSNR for reference).

    Returns
    -------
    noisy_psnr  : list[float]  – PSNR of x + noise vs. x_clean (theoretical: 10log(1/σ²))
    denoised_psnr : list[float]
    """
    noisy_psnr_all    = []
    denoised_psnr_all = []

    for sigma in noise_levels:
        psnr_noisy_buf    = []
        psnr_denoised_buf = []

        for batch_idx, (x, _) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(DEVICE, dtype=torch.float32)
            noise = torch.randn_like(x) * sigma
            x_noisy = x + noise

            x_hat = wrapper.denoise(x_noisy, sigma)

            psnr_noisy_buf.append(compute_psnr(x_noisy, x))
            psnr_denoised_buf.append(compute_psnr(x_hat,  x))

        noisy_psnr_all.append(float(np.mean(psnr_noisy_buf)))
        denoised_psnr_all.append(float(np.mean(psnr_denoised_buf)))
        print(f"  σ={sigma:.3f}  noisy={noisy_psnr_all[-1]:.2f} dB  "
              f"denoised={denoised_psnr_all[-1]:.2f} dB")

    return noisy_psnr_all, denoised_psnr_all


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_psnr_curves(results: dict, noise_levels, out_path: Path):
    """
    results: dict mapping label -> (noisy_psnr_list, denoised_psnr_list)
    """
    colors  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    sigmas = np.array(noise_levels)

    for (label, (_, denoised)), color, marker in zip(results.items(), colors, markers):
        ax.plot(sigmas, denoised, f"-{marker}", color=color, lw=2.2,
                markersize=6, markeredgewidth=0.8, markeredgecolor="white",
                label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise level  σ", fontsize=13)
    ax.set_ylabel("PSNR  (dB)", fontsize=13)
    ax.set_title("Denoising PSNR — CelebA val set", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, which="major", ls="-",  lw=0.6, alpha=0.5, color="white")
    ax.grid(True, which="minor", ls="--", lw=0.4, alpha=0.3, color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(labelsize=11, colors="#333333")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def arr_from_tensor(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float tensor → H×W×3 uint8."""
    t = t.float().cpu()
    lo, hi = t.min(), t.max()
    t = (t - lo) / max((hi - lo).item(), 1e-8)
    t = t.clamp(0, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def plot_generation_grid(samples_dict: dict, out_path: Path):
    """
    samples_dict: {label: tensor (N, C, H, W)}
    Each model gets its own row, with a colored label bar above the row.
    """
    labels     = list(samples_dict.keys())
    n_models   = len(labels)
    n_cols     = N_SAMPLES
    cell_size  = 1.7
    bar_height = 0.3   # height of label bar row in inches

    row_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    # Interleave: label-bar row + image row for each model
    n_rows  = n_models * 2
    heights = []
    for _ in range(n_models):
        heights += [bar_height, cell_size]

    fig_h = sum(heights) + 0.3
    fig_w = n_cols * cell_size
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        height_ratios=heights,
        hspace=0.03, wspace=0.03,
        left=0.01, right=0.99, top=0.97, bottom=0.01,
    )

    for row, (label, color) in enumerate(zip(labels, row_colors)):
        bar_row = row * 2
        img_row = row * 2 + 1

        # Colored label bar spanning all columns
        ax_bar = fig.add_subplot(gs[bar_row, :])
        ax_bar.set_facecolor(color)
        ax_bar.text(0.02, 0.5, label, transform=ax_bar.transAxes,
                    ha="left", va="center", fontsize=12, fontweight="bold",
                    color="white")
        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
        for sp in ax_bar.spines.values():
            sp.set_visible(False)

        imgs = samples_dict[label]
        for col in range(n_cols):
            ax = fig.add_subplot(gs[img_row, col])
            ax.imshow(arr_from_tensor(imgs[col]))
            ax.axis("off")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    print(f"Loading model from {model_dir} ...")
    config, lit = load_lit_model(model_dir, device=DEVICE)
    arch    = config["model_arch"]
    n_iters = N_ITERS_SHEET if arch == "neural_sheet7" else 1
    if arch == "neural_sheet7":
        for lvl in lit.ema_model.levels:
            lvl.reset_wnorm()
    wrapper = DenoiserWrapper(lit, arch, n_iters=n_iters)
    print(f"  arch={arch}  n_iters={n_iters}")
    return config, wrapper


def main():
    torch.manual_seed(0)

    # ── Load models ───────────────────────────────────────────────────────────
    models = {}  # label -> (config, wrapper)

    print("=" * 60)
    sheet_config, sheet_wrapper = load_model(SHEET_DIR)
    models["neural_sheet7"] = (sheet_config, sheet_wrapper)

    if UNET_DIR is not None:
        unet_config, unet_wrapper = load_model(UNET_DIR)
        models["UNet"] = (unet_config, unet_wrapper)
    else:
        print("UNET_DIR not set — skipping UNet (set UNET_DIR at top of script)")

    # ── Build per-model val loaders (models may differ in img_size / grayscale) ──
    print("\nBuilding val loaders ...")
    val_loaders = {}
    for label, (cfg, _) in models.items():
        _, _, vl, _ = build_loaders_from_config(cfg, batch_size=BATCH_SIZE, num_workers=2)
        val_loaders[label] = vl
        print(f"  [{label}]  img_size={cfg.get('img_size')}  "
              f"grayscale={cfg.get('grayscale')}  batches={len(vl)}")

    # ── Denoising PSNR curves ─────────────────────────────────────────────────
    print("\n── Denoising evaluation ──")
    psnr_results = {}
    for label, (cfg, wrapper) in models.items():
        print(f"\n  [{label}]")
        noisy_psnr, denoised_psnr = eval_denoising_psnr(
            wrapper, val_loaders[label], NOISE_LEVELS, max_batches=MAX_BATCHES
        )
        psnr_results[label] = (noisy_psnr, denoised_psnr)

    plot_psnr_curves(psnr_results, NOISE_LEVELS, OUT_DIR / "comparison_psnr_curve.png")

    # ── Generation ────────────────────────────────────────────────────────────
    print("\n── Generation ──")

    # EDM polynomial schedule (Karras et al.)
    step_indices = torch.arange(N_SIGMA_STEPS, dtype=torch.float32)
    sigmas = (SIGMA_MAX ** (1 / RHO)
              + step_indices / (N_SIGMA_STEPS - 1)
              * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO

    gen_results = {}
    for label, (cfg, wrapper) in models.items():
        img_size  = cfg.get("img_size", 128)
        grayscale = cfg.get("grayscale", True)
        C         = 1 if grayscale else 3

        # Compute mean image from train loader (same as notebook)
        print(f"\n  [{label}] Computing mean image from {N_MEAN_BATCHES} train batches ...")
        _, train_loader_gen, _, _ = build_loaders_from_config(
            cfg, batch_size=BATCH_SIZE, num_workers=2
        )
        x_ls = []
        for i, (x, _) in enumerate(train_loader_gen):
            x_ls.append(x.cpu())
            if i >= N_MEAN_BATCHES:
                break
        mean_img = torch.cat(x_ls).mean(0, keepdim=True).repeat(N_SAMPLES, 1, 1, 1)

        start_img = (mean_img + torch.randn_like(mean_img) * SIGMA_MAX).to(DEVICE)
        shape = start_img.shape
        print(f"  Generating {N_SAMPLES} samples  shape={shape} ...")

        samples, _ = annealed_heun_dynamics(
            model=wrapper,
            shape=shape,
            sigmas=sigmas.to(DEVICE),
            n_iters_denoiser=N_ITERS_DENOISE,
            device=DEVICE,
            record_per_steps=RECORD_EVERY,
            start_img=start_img,
        )
        gen_results[label] = samples.detach().cpu()
        print(f"  Done. sample range: [{samples.min():.3f}, {samples.max():.3f}]")

    plot_generation_grid(gen_results, OUT_DIR / "comparison_generation.png")

    print("\nAll done.")


if __name__ == "__main__":
    main()
