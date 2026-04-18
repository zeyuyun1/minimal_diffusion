"""
run_noise_label_sweep.py
========================
Feed a clean image into the model with varying noise LABELS (no actual noise
added to the image). Shows how the model's output changes as the noise label
increases — ideally the clean circle is preserved while the corrupted circle
is completed/regularized toward the learned prior.

Output: figures/noise_label_sweep_<tag>.pdf
"""

import argparse
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms

from recurrent_diffusion_pkg.utils import load_lit_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str,
                   default="pretrained_model/scaling-VH-new-2/"
                           "00008_simple_sheet7_simple_control_small_noise_long_iter")
    p.add_argument("--clean_stimulus",   type=str, default="stimulus/shape_Circle_clean.png")
    p.add_argument("--corrupt_stimulus", type=str, default="stimulus/shape_Circle_perturbed.png")
    p.add_argument("--out_dir",  type=str, default="figures")
    p.add_argument("--tag",      type=str, default="00008_circle")
    p.add_argument("--n_iters",  type=int, default=8)
    p.add_argument("--noise_labels", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])
    p.add_argument("--ff_scale",   type=float, default=1.0,
                   help="Scale res_in (feedforward drive) by this factor (1.0 = normal)")
    p.add_argument("--panel_size", type=float, default=3.0)
    return p.parse_args()


@contextmanager
def patch_ff_scale(net, ff_scale: float):
    original = net.forward_inter

    def patched_forward_inter(
        self, x, x_in, a, upstream_grad, decoded,
        noise_emb_ls=None, T=0, return_history=False,
        ablate_noi=None, ablate_mode="spatial_mean", measurement=None,
    ):
        if self.constraint_energy == "SC" and decoded[0] is not None:
            res_in = self.encoder_0(x - self.decoder_0(decoded))
        else:
            res_in = self.encoder_0(x)
        res_in = [r * ff_scale for r in res_in]
        self.forward_dynamics(a, res_in, upstream_grad, decoded, noise_emb_ls, T, ablate_noi, ablate_mode, reverse=False)
        snap1 = self._snapshot(a, decoded, upstream_grad) if return_history else None
        self.forward_dynamics(a, res_in, upstream_grad, decoded, noise_emb_ls, T, ablate_noi, ablate_mode, reverse=True)
        snap2 = self._snapshot(a, decoded, upstream_grad) if return_history else None
        if return_history:
            return [snap1, snap2]
        return None

    net.forward_inter = types.MethodType(patched_forward_inter, net)
    try:
        yield
    finally:
        net.forward_inter = original



def main():
    args = parse_args()
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {Path(args.model_dir).name} ...")
    config, lit_model = load_lit_model(Path(args.model_dir), device=DEVICE, ckpt_name=None)
    net = lit_model.ema_model.eval()
    for lvl in net.levels:
        lvl.reset_wnorm()

    import torch.nn as _nn
    _first_conv = next(m for m in net.levels[0].modules() if isinstance(m, _nn.Conv2d))
    n_channels = _first_conv.in_channels

    # ── Load stimuli ──────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Grayscale(1) if n_channels == 1 else transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    stimuli = [
        ("clean circle",     transform(Image.open(args.clean_stimulus))),
        ("corrupted circle", transform(Image.open(args.corrupt_stimulus))),
    ]
    print(f"Stimuli: {[n for n,_ in stimuli]}")

    noise_labels = args.noise_labels
    n_labels = len(noise_labels)
    n_stim   = len(stimuli)

    # ── Run model for each noise label ────────────────────────────────────────
    # Stack both stimuli into one batch, run once per noise label
    imgs = torch.stack([t for _, t in stimuli]).to(DEVICE)  # [2, C, H, W]

    # results[label_idx][stim_idx] = {"denoised": tensor, "history": [...]}
    results = []
    for sigma_val in noise_labels:
        sigma = torch.full((n_stim,), sigma_val, device=DEVICE, dtype=imgs.dtype)
        print(f"  noise_label={sigma_val:.3f}  ff_scale={args.ff_scale} ...", end=" ", flush=True)
        with patch_ff_scale(net, args.ff_scale):
            with torch.no_grad():
                history = net(
                    imgs, noise_labels=sigma,
                    infer_mode=True, n_iters=args.n_iters, return_history=True,
                )
        results.append(history)
        print("done")

    # ── Build PDF ─────────────────────────────────────────────────────────────
    # Page layout: rows = stimuli, cols = noise labels
    # Row 0: input image (same for all cols, shown once)
    # Rows 1+: denoised output + Gabor overlay per noise label

    out_path = OUT_DIR / f"noise_label_sweep_{args.tag}.pdf"
    print(f"\nWriting {out_path} ...")

    with PdfPages(out_path) as pdf:
        for stim_idx, (stim_name, stim_tensor) in enumerate(stimuli):
            print(f"  [{stim_idx+1}/{n_stim}] {stim_name}")

            fig, axes = plt.subplots(
                1, n_labels + 1,
                figsize=((n_labels + 1) * args.panel_size, args.panel_size),
                facecolor="black",
            )
            fig.patch.set_facecolor("black")
            for ax in axes.flat:
                ax.set_facecolor("black")
                ax.axis("off")

            # Col 0: input image
            inp_arr = stim_tensor[0].numpy() if n_channels == 1 else stim_tensor.permute(1,2,0).numpy()
            axes[0].imshow(inp_arr, cmap="gray" if n_channels == 1 else None, vmin=0, vmax=1)
            axes[0].set_title("input", fontsize=7, color="white", pad=2)

            for col, (sigma_val, history) in enumerate(zip(noise_labels, results)):
                den = history[-1]["denoised"][stim_idx]
                den_arr = den[0].cpu().float().numpy() if n_channels == 1 else den.cpu().permute(1,2,0).float().numpy()
                den_arr = (den_arr - den_arr.min()) / max(den_arr.max() - den_arr.min(), 1e-8)
                axes[col + 1].imshow(den_arr, cmap="gray" if n_channels == 1 else None, vmin=0, vmax=1)
                axes[col + 1].set_title(f"σ={sigma_val:.2f}", fontsize=7, color="white", pad=2)

            fig.suptitle(
                f"{stim_name}  |  noise label sweep  |  ff_scale={args.ff_scale}  |  n_iters={args.n_iters}",
                fontsize=9, color="white", y=1.05,
            )
            plt.tight_layout(pad=0.3)
            pdf.savefig(fig, bbox_inches="tight", facecolor="black")
            plt.close(fig)
            print(f"    done")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
