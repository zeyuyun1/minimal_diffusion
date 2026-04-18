# Recurrent Diffusion — Minimal Codebase

Minimal reproduction of the experiments in the paper. All scripts are run from the repo root.

## Setup

```bash
pip install torch torchvision matplotlib pillow numpy scipy
```

Pretrained model checkpoints are in `pretrained_model/`. Stimulus images are in `stimulus/`. All figures are saved to `figures/`.

---

## Figure 1 — Generalization Behavior of a Standard Diffusion Model

### 1.A — Perturbation Generalization (`harmonic_basis_UNet.png`)

Shows the top eigenvectors of the denoiser Jacobian J_D = dD(x)/dx (the "harmonic basis") for a UNet trained on faces. These are the directions in pixel space where the model is most sensitive/confident. Smooth, low-frequency patterns indicate the model generalizes to structured perturbations.

**Output:** `figures/harmonic_basis_UNet.png`

**Code:** `run_harmonic_basis.py`

```bash
python run_harmonic_basis.py --unet_only
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--unet_dir` | `pretrained_model/scaling-new-face/00029_layer1_unet_small_edm_large_noattn` | UNet model directory |
| `--noise_level` | `0.3` | Noise level at which to compute eigenvectors |
| `--k` | `100` | Number of eigenvectors to compute |
| `--k_plot_stride` | `4` | Plot every Nth eigenvector (25 shown) |
| `--n_power_iter` | `10` | Power iteration steps (more = more accurate) |
| `--val_img_idx` | `2` | Which validation image to use |

To also generate the neural_sheet7 comparison (`harmonic_basis_comparison.png`):
```bash
python run_harmonic_basis.py
```

---

### 1.B — Simplicity Bias (`simplicity_bias_unet.png`)

Ranks training images together with OOD shape stimuli (circle, square, triangle) by estimated log-likelihood under the UNet denoiser:

```
score(x) = -E_{σ,ε}[ ||D(x + σε, σ) − x||² ]
```

Simpler images (shapes) rank highest — the model assigns higher likelihood to out-of-distribution stimuli that are structurally simpler than faces.

**Output:** `figures/simplicity_bias_unet.png`

**Code:** `run_simplicity_bias_combined.py`

```bash
python run_simplicity_bias_combined.py --unet_only
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--unet_dir` | `pretrained_model/scaling-VH-new-2/00033_layer1_unet_small_edm_noatt` | UNet model directory |
| `--stim_dir` | `stimulus` | Directory with shape stimuli |
| `--n_train_imgs` | `300` | Number of training faces to include in ranking |

To generate the combined UNet + Sheet7 comparison figure:
```bash
python run_simplicity_bias_combined.py
```
Output: `figures/simplicity_bias_combined.png`

---

---

## Figure 2 — Manifold Hypothesis

These figures illustrate the core modeling assumptions: data lives as sparse signal on a low-dimensional manifold; we tile the manifold and approximate the signal distribution on it.

> **TODO (collaborator):** Add the code and instructions for each sub-figure below.

### 2.A — Data Distribution

Visualization of the data distribution on the manifold.

> **TODO (collaborator):** Add script path, output filename, and run command.

---

### 2.B — Causal Generation

Shows how to generate a signal in pixel space from a 4-sparse signal on the manifold.

> **TODO (collaborator):** Add script path, output filename, and run command.

---

### 2.C — Sparse Coding Visualization

How the manifold is tiled and how signals on the manifold are approximated via sparse coding.

> **TODO (collaborator):** Add script path, output filename, and run command.

---

### 2.D — Different Distributions on the Manifold

Visualizes multiple distinct signal distributions that can arise on the manifold.

> **TODO (collaborator):** Add script path, output filename, and run command.

---

## Directory Structure

```
recurrent_diffusion_minimal/
├── pretrained_model/          # Model checkpoints
│   ├── scaling-new-face/      # Models trained on CelebA faces
│   └── scaling-VH-new-2/      # Models trained on natural images (VH dataset)
├── stimulus/                  # Shape stimuli (circle, square, triangle, etc.)
├── figures/                   # All output figures saved here
├── recurrent_diffusion_pkg/   # Core model and utility code
│   ├── model.py               # neural_sheet7 and UNet architectures
│   ├── needle_plot.py         # Gabor fitting and overlay rendering
│   └── utils.py               # Model loading utilities
├── run_harmonic_basis.py      # Figure 1.A
└── run_simplicity_bias_combined.py  # Figure 1.B
```
