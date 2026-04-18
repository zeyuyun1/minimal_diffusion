"""
run_generation_trajectory.py
============================
Visualise the full denoising trajectory for both neural_sheet7 and UNet.

Layout
------
Each sample trajectory = one row.
Columns = evenly-spaced ODE steps; LAST column = final denoised image.
Models are separated into labelled sections.

    [sheet7 label] | t=σ_max | … | t=σ_mid | … | final
    [sheet7 label] | …                             final
    [UNet   label] | …                             final
    [UNet   label] | …                             final

Output
------
figures/comparison_generation_trajectory.png
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from recurrent_diffusion_pkg.utils import load_lit_model, build_loaders_from_config

# ── Config ────────────────────────────────────────────────────────────────────
SHEET_DIR = Path("pretrained_model/scaling-new-face/"
                 "00026_simplify_layer1_neural_sheet7_intra_"
                 "film_scalar_mul_ff_scale_large_large_noise_simple_control")
UNET_DIR  = Path("pretrained_model/scaling-new-face/"
                 "00029_layer1_unet_small_edm_large_noattn")

N_TRAJ_SAMPLES  = 4      # trajectories (rows) per model
N_TRAJ_SHOW     = 9      # intermediate steps shown (+ 1 final = 10 cols total)
SIGMA_MAX       = 4.0
SIGMA_MIN       = 0.1
N_SIGMA_STEPS   = 40
RHO             = 7.0
N_ITERS_DENOISE = 4      # recurrent iters for sheet7
N_MEAN_BATCHES  = 100
BATCH_SIZE      = 16

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


# ── Model wrapper (same as run_model_comparison.py) ──────────────────────────

class DenoiserWrapper:
    def __init__(self, lit_model, model_arch, n_iters=8):
        self.net     = lit_model.ema_model.eval()
        self.arch    = model_arch
        self.n_iters = n_iters

    @torch.no_grad()
    def denoise(self, x, sigma):
        B     = x.shape[0]
        sig_t = torch.full((B,), float(sigma), device=x.device, dtype=x.dtype)
        if self.arch == "neural_sheet7":
            return self.net(x, noise_labels=sig_t,
                            infer_mode=True, n_iters=self.n_iters)
        elif self.arch == "unet":
            return self.net(x, sig_t)
        raise ValueError(self.arch)

    def __call__(self, x, noise_labels, infer_mode=False, n_iters=None, **kw):
        return self.denoise(x, float(noise_labels[0].item()))


# ── Annealed Heun sampler — records every step ────────────────────────────────

@torch.no_grad()
def sample_with_trajectory(model, shape, sigmas, n_iters_denoiser, device, start_img):
    """
    Returns
    -------
    x_final : (B, C, H, W)
    history : list of (B, C, H, W) tensors — one per ODE step + final denoise
    sigmas_hist : list of sigma values corresponding to each history entry
    """
    B, C, H, W = shape
    sigmas  = torch.as_tensor(sigmas, device=device, dtype=torch.float32)
    x       = start_img.to(device=device, dtype=torch.float32)
    history = []
    sigmas_hist = []

    def _den(x_, s_):
        sig = torch.full((B,), float(s_), device=device, dtype=torch.float32)
        return model(x_, noise_labels=sig, infer_mode=True, n_iters=n_iters_denoiser)

    for i in range(len(sigmas) - 1):
        t_cur, t_next = sigmas[i], sigmas[i + 1]
        den_cur  = _den(x, t_cur)
        d_cur    = (x - den_cur) / t_cur
        x_euler  = x + (t_next - t_cur) * d_cur
        den_next = _den(x_euler, t_next)
        d_prime  = (x_euler - den_next) / t_next
        x = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        history.append(x.detach().cpu())
        sigmas_hist.append(float(t_next))

    # Final denoise
    x_final = _den(x, sigmas[-1])
    history.append(x_final.detach().cpu())
    sigmas_hist.append(0.0)

    return x_final.detach().cpu(), history, sigmas_hist


# ── Image display helper ──────────────────────────────────────────────────────

def to_img(t):
    """(C,H,W) tensor → H×W×3 float [0,1]."""
    t = t.float().cpu()
    lo, hi = t.min(), t.max()
    t = (t - lo) / max((hi - lo).item(), 1e-8)
    t = t.clamp(0, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t.permute(1, 2, 0).numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model(model_dir):
    print(f"  Loading {model_dir.name} ...")
    config, lit = load_lit_model(model_dir, device=DEVICE)
    arch = config["model_arch"]
    n_it = N_ITERS_DENOISE if arch == "neural_sheet7" else 1
    if arch == "neural_sheet7":
        for lvl in lit.ema_model.levels:
            lvl.reset_wnorm()
    return config, DenoiserWrapper(lit, arch, n_iters=n_it)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # ── EDM sigma schedule ────────────────────────────────────────────────────
    step_idx = torch.arange(N_SIGMA_STEPS, dtype=torch.float32)
    sigmas   = (SIGMA_MAX ** (1 / RHO)
                + step_idx / (N_SIGMA_STEPS - 1)
                * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO

    model_entries = [
        ("neural_sheet7", SHEET_DIR),
        ("UNet",          UNET_DIR),
    ]

    all_trajs   = {}
    all_sigmas  = {}

    # Each model gets its own independent seeds so rows show each model's
    # own diverse generations rather than matched pairs from identical noise.
    model_seeds = {label: base * 100 for base, (label, _) in enumerate(model_entries)}

    for label, model_dir in model_entries:
        print(f"\n{'='*60}\n  [{label}]")
        config, wrapper = load_model(model_dir)

        img_size  = config.get("img_size", 128)
        grayscale = config.get("grayscale", True)
        C = 1 if grayscale else 3

        # Mean image start point
        print(f"  Computing mean image ...")
        _, train_loader, _, _ = build_loaders_from_config(
            config, batch_size=BATCH_SIZE, num_workers=2)
        x_ls = []
        for i, (x, _) in enumerate(train_loader):
            x_ls.append(x.cpu())
            if i >= N_MEAN_BATCHES:
                break
        mean_img = torch.cat(x_ls).mean(0, keepdim=True)   # (1, C, H, W)

        trajs = []
        for s in range(N_TRAJ_SAMPLES):
            torch.manual_seed(model_seeds[label] + s)
            start = (mean_img.expand(1, -1, -1, -1)
                     + torch.randn(1, C, img_size, img_size) * SIGMA_MAX)

            print(f"  Sample {s+1}/{N_TRAJ_SAMPLES} ...", end=" ", flush=True)
            _, history, sig_hist = sample_with_trajectory(
                wrapper,
                shape=(1, C, img_size, img_size),
                sigmas=sigmas,
                n_iters_denoiser=N_ITERS_DENOISE if label == "neural_sheet7" else 1,
                device=DEVICE,
                start_img=start,
            )
            trajs.append(history)
            print(f"done  ({len(history)} steps)")

        all_trajs[label]  = trajs
        all_sigmas[label] = sig_hist

    # ── Build figure ──────────────────────────────────────────────────────────
    N_COLS      = N_TRAJ_SHOW + 1   # intermediate steps + final
    N_ROWS      = N_TRAJ_SAMPLES * len(model_entries)
    CELL        = 1.5
    LABEL_W     = 1.4    # width of left label column in inches
    BAR_H       = 0.35   # height of model-section header bar

    row_colors  = {"neural_sheet7": "#2196F3", "UNet": "#FF5722"}

    fig_w  = LABEL_W + N_COLS * CELL
    # each model section: 1 bar row + N_TRAJ_SAMPLES image rows
    fig_h  = len(model_entries) * (BAR_H + N_TRAJ_SAMPLES * CELL) + 0.4
    fig    = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Outer: one row per model section
    n_outer = len(model_entries)
    section_heights = [BAR_H + N_TRAJ_SAMPLES * CELL] * n_outer
    outer_gs = gridspec.GridSpec(
        n_outer, 1, figure=fig,
        height_ratios=section_heights,
        hspace=0.08,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
    )

    for sec_idx, (label, model_dir) in enumerate(model_entries):
        trajs    = all_trajs[label]
        sig_hist = all_sigmas[label]
        color    = row_colors[label]
        n_total  = len(trajs[0])

        # Subsample indices: N_TRAJ_SHOW evenly from [0, n_total-2], then -1 (final)
        intermediate_idx = np.round(
            np.linspace(0, n_total - 2, N_TRAJ_SHOW)
        ).astype(int).tolist()
        show_idx = intermediate_idx + [n_total - 1]  # final denoised last

        # Inner GridSpec for this section: 1 header bar + N_TRAJ_SAMPLES rows
        inner_rows   = 1 + N_TRAJ_SAMPLES
        inner_heights = [BAR_H] + [CELL] * N_TRAJ_SAMPLES
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            inner_rows, N_COLS + 1,   # +1 for label column
            subplot_spec=outer_gs[sec_idx],
            height_ratios=inner_heights,
            hspace=0.04, wspace=0.03,
            width_ratios=[LABEL_W] + [CELL] * N_COLS,
        )

        # Section header bar
        ax_bar = fig.add_subplot(inner_gs[0, :])
        ax_bar.set_facecolor(color)
        ax_bar.text(0.01, 0.5, label,
                    transform=ax_bar.transAxes,
                    ha="left", va="center",
                    fontsize=12, fontweight="bold", color="white")
        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
        for sp in ax_bar.spines.values():
            sp.set_visible(False)

        # Column headers (sigma values) — only for first section
        if sec_idx == 0:
            for col_pos, step in enumerate(show_idx):
                s_val = sig_hist[step]
                title = "final" if step == n_total - 1 else f"σ={s_val:.2f}"
                # embed as title of the first-row image axes (set later)
                ax_bar.text(
                    (LABEL_W + (col_pos + 0.5) * CELL) / fig_w,
                    -0.15,
                    title,
                    transform=fig.transFigure,
                    ha="center", va="top", fontsize=7, color="#555555",
                )

        # Image rows
        for row_idx, history in enumerate(trajs):
            # Left label cell
            ax_lbl = fig.add_subplot(inner_gs[row_idx + 1, 0])
            ax_lbl.set_facecolor("#f0f0f0")
            ax_lbl.text(0.5, 0.5, f"sample\n{row_idx+1}",
                        transform=ax_lbl.transAxes,
                        ha="center", va="center",
                        fontsize=8, color="#444444")
            ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
            for sp in ax_lbl.spines.values():
                sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.5)

            for col_pos, step in enumerate(show_idx):
                ax = fig.add_subplot(inner_gs[row_idx + 1, col_pos + 1])
                frame = history[step][0]    # (C, H, W)
                ax.imshow(to_img(frame))
                ax.set_xticks([]); ax.set_yticks([])

                # Highlight final column with colored border
                if step == n_total - 1:
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_edgecolor(color)
                        sp.set_linewidth(2.5)
                else:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#dddddd")
                        sp.set_linewidth(0.4)

    out_path = OUT_DIR / "comparison_generation_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out_path}")
    print("All done.")


if __name__ == "__main__":
    main()
