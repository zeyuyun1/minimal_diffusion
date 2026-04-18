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

## Figure 3 — Comparison: Recurrent Sheet vs. Standard UNet

### 3.A — PSNR Curve (`comparison_psnr_curve.png`)

Denoising PSNR vs. noise level σ (log–log) for both models evaluated on the CelebA validation set.

**Output:** `figures/comparison_psnr_curve.png`

**Code:** `run_model_comparison.py`

```bash
python run_model_comparison.py
```

Key options (edit constants at the top of the file):
| Constant | Default | Description |
|----------|---------|-------------|
| `SHEET_DIR` | `pretrained_model/scaling-new-face/00026_...` | neural_sheet7 model directory |
| `UNET_DIR` | `pretrained_model/scaling-new-face/00006_...` | UNet model directory |
| `N_ITERS_SHEET` | `8` | Recurrent iterations for neural_sheet7 |
| `MAX_BATCHES` | `20` | Val batches to average over |

---

### 3.B — Generation Comparison (`comparison_generation.png`)

Side-by-side random samples from UNet and neural_sheet7, generated via annealed Heun ODE sampler starting from the mean training image plus Gaussian noise.

**Output:** `figures/comparison_generation.png`

**Code:** `run_model_comparison.py` (same script as 3.A — both figures are produced in one run)

```bash
python run_model_comparison.py
```

---

### 3.C — Generation Trajectory (`comparison_generation_trajectory.png`)

Full denoising trajectories for both models: each row is one sample, columns show evenly-spaced ODE steps from σ_max to the final denoised image.

**Output:** `figures/comparison_generation_trajectory.png`

**Code:** `run_generation_trajectory.py`

```bash
python run_generation_trajectory.py
```

Key options (edit constants at the top of the file):
| Constant | Default | Description |
|----------|---------|-------------|
| `N_TRAJ_SAMPLES` | `4` | Number of sample rows per model |
| `N_TRAJ_SHOW` | `9` | Intermediate steps shown per row |
| `N_ITERS_DENOISE` | `4` | Recurrent iterations for neural_sheet7 |

---

### 3.D — Simplicity Bias Comparison (`simplicity_bias_combined.png`)

Both models ranked together: training faces + OOD shape stimuli sorted by estimated log-likelihood. Shows that both models assign higher likelihood to simpler, out-of-distribution stimuli.

**Output:** `figures/simplicity_bias_combined.png`

**Code:** `run_simplicity_bias_combined.py`

```bash
python run_simplicity_bias_combined.py
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--sheet_dir` | `pretrained_model/scaling-new-face/00026_...` | neural_sheet7 model directory |
| `--unet_dir` | `pretrained_model/scaling-VH-new-2/00033_...` | UNet model directory |
| `--n_train_imgs` | `300` | Number of training faces to include in ranking |
| `--stim_dir` | `stimulus` | Directory with shape stimuli |

---

### 3.E — Harmonic Basis Comparison (`harmonic_basis_neural_sheet7.png` + `harmonic_basis_UNet.png`)

Top eigenvectors of the denoiser Jacobian for both models shown side by side. neural_sheet7 eigenvectors tend to be more spatially localized than UNet's smooth low-frequency basis.

**Outputs:** `figures/harmonic_basis_UNet.png`, `figures/harmonic_basis_neural_sheet7.png`, `figures/harmonic_basis_comparison.png`

**Code:** `run_harmonic_basis.py`

```bash
python run_harmonic_basis.py
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--unet_dir` | `pretrained_model/scaling-new-face/00029_...` | UNet model directory |
| `--sheet_dir` | `pretrained_model/scaling-new-face/00024_...` | neural_sheet7 model directory |
| `--noise_level` | `0.3` | Noise level at which to compute eigenvectors |
| `--k` | `100` | Number of eigenvectors to compute |
| `--n_power_iter` | `10` | Power iteration steps (more = more accurate) |

---

---

## Figure 5 — Decomposition of the Harmonic Basis

The harmonic basis (Jacobian eigenvectors from Figure 1.A / 3.E) measures how a pixel perturbation propagates through the model. Here we show that this Jacobian calculation decomposes into perturbations in the **latent space** — specifically, each eigenvector corresponds to how a single Gabor filter's activation spreads to neighbouring neurons under the model's learned lateral connections.

This equivalence is visualised concretely: a target neuron near the circle boundary is identified, and we show how its activation pattern spreads across the image under the recurrent dynamics, producing the same structured, contour-following pattern as a pixel-space Jacobian eigenvector.

**Output:** `figures/gen_combined_shape_Circle_perturbed.png`

**Code:** `run_contour_experiments.py`

```bash
python run_contour_experiments.py
```

**Model:** `pretrained_model/scaling-VH-new-2/00008_simple_sheet7_simple_control_small_noise_long_iter`

Key options (edit constants at the top of the file):
| Constant | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | `pretrained_model/scaling-VH-new-2/00008_...` | Model directory |
| `STIMULI` | `["shape_Circle_perturbed"]` | List of stimulus stems to process |
| `NOISE_LEVEL` | `0.4` | Noise level for denoising |
| `N_ITERS` | `8` | Recurrent iterations |

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
├── run_harmonic_basis.py      # Figures 1.A, 3.E
├── run_simplicity_bias_combined.py  # Figures 1.B, 3.D
├── run_model_comparison.py    # Figures 3.A, 3.B
├── run_generation_trajectory.py    # Figure 3.C
├── run_denoising_history.py   # Denoising history + Gabor overlay
├── run_contour_experiments.py # Figure 5
├── run_ff_scale_sweep.py      # Causal intervention: feedforward scale sweep
├── run_noise_label_sweep.py   # Causal intervention: noise label sweep
└── run_pyramid_corruption.py  # Causal intervention: pyramid-level boundary corruption
```
