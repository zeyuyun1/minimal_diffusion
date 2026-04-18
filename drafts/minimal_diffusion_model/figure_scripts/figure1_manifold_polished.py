#!/usr/bin/env python3

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from torch.utils.data import DataLoader, Dataset


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
PNG_PATH = OUTPUT_DIR / "manifold_hypothesis_polished.png"
PDF_PATH = OUTPUT_DIR / "manifold_hypothesis_polished.pdf"

BG = "#FCFCFB"
TEXT = "#1E2430"
MUTED = "#5B6573"
RED = "#E9857E"
BLUE = "#5B7CFA"
GRAY = "#9AA1AA"
DARK_GRAY = "#59606A"
BAND_FILL = "#ECEFF4"
BAND_EDGE = "#B7BDC7"
GOOD = "#1B7F5B"
MID = "#C27A00"
BAD = "#B54708"

THETA_LANDMARKS = np.linspace(0, 2 * math.pi, 28, endpoint=False)
RADIAL_LANDMARKS = np.linspace(0.08, 0.92, 5)
DATA_SEED = 0


class AsymmetricCrossDataset(Dataset):
    def __init__(self, num_samples=5000, d_dim=128):
        self.num_samples = num_samples
        self.d_dim = d_dim
        self.grid = torch.linspace(0, 2 * math.pi, d_dim + 1)[:-1]

        self.std_mu = 0.1
        self.std_sigma = 0.05

        mu_base = torch.rand(num_samples, 1) * 2 * math.pi
        sigma_base_major = 0.1 + 0.3 * torch.rand(num_samples, 1)
        sigma_base_minor = 0.1 + 0.3 * torch.rand(num_samples, 1)

        mu_n = (mu_base + torch.randn(num_samples, 1) * self.std_mu) % (2 * math.pi)
        sigma_n = torch.clamp(sigma_base_major + torch.randn(num_samples, 1) * self.std_sigma, min=0.15, max=0.45)
        scale_n = torch.full((num_samples, 1), 1.2)

        mu_s = (mu_base + math.pi + torch.randn(num_samples, 1) * self.std_mu) % (2 * math.pi)
        sigma_s = torch.clamp(sigma_base_major + torch.randn(num_samples, 1) * self.std_sigma, min=0.15, max=0.45)
        scale_s = torch.full((num_samples, 1), 1.2)

        mu_e = (mu_base + math.pi / 2 + torch.randn(num_samples, 1) * self.std_mu) % (2 * math.pi)
        sigma_e = torch.clamp(sigma_base_minor + torch.randn(num_samples, 1) * self.std_sigma, min=0.15, max=0.45)
        scale_e = torch.full((num_samples, 1), 0.7)

        mu_w = (mu_base - math.pi / 2 + torch.randn(num_samples, 1) * self.std_mu) % (2 * math.pi)
        sigma_w = torch.clamp(sigma_base_minor + torch.randn(num_samples, 1) * self.std_sigma, min=0.15, max=0.45)
        scale_w = torch.full((num_samples, 1), 0.7)

        coords_n = torch.cat([mu_n, sigma_n, scale_n], dim=1)
        coords_s = torch.cat([mu_s, sigma_s, scale_s], dim=1)
        coords_e = torch.cat([mu_e, sigma_e, scale_e], dim=1)
        coords_w = torch.cat([mu_w, sigma_w, scale_w], dim=1)
        self.manifold_coords = torch.stack([coords_n, coords_s, coords_e, coords_w], dim=1)

        components = [
            (mu_n, sigma_n, scale_n),
            (mu_s, sigma_s, scale_s),
            (mu_e, sigma_e, scale_e),
            (mu_w, sigma_w, scale_w),
        ]
        self.data = self.generate(components)

    def generate(self, components):
        batch_size = components[0][0].shape[0]
        signal = torch.zeros(batch_size, self.d_dim, device=self.grid.device)
        for mu, sigma, amplitude in components:
            mu = mu.view(batch_size, 1)
            sigma = sigma.view(batch_size, 1)
            loc = ((self.grid - mu + math.pi) % (2 * math.pi)) - math.pi
            base_blade = torch.exp(-torch.abs(loc) / sigma)
            signal += base_blade
        return signal

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.manifold_coords[idx]


    def get_grid_samples(self, mu_steps=72, sigma_steps=10):
        mu_vals = torch.linspace(0, 2 * math.pi, mu_steps + 1)[:-1]
        sigma_vals = torch.linspace(0.15, 0.45, sigma_steps)

        mu_grid, sigma_grid = torch.meshgrid(mu_vals, sigma_vals, indexing="ij")
        mu_flat = mu_grid.flatten().unsqueeze(1)
        sigma_flat = sigma_grid.flatten().unsqueeze(1)
        grid_coords = torch.cat([mu_flat, sigma_flat], dim=1)

        amplitude_flat = torch.ones_like(mu_flat)
        components = [(mu_flat, sigma_flat, amplitude_flat)]
        grid_data = self.generate(components)
        return grid_data, grid_coords


class TiedTranspose(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)


class SCInference(nn.Module):
    def __init__(self, data_dim, hidden_dim, eta=0.1, lam=0.0, learning_horizontal=True):
        super().__init__()
        self.eta = eta
        self.lam = lam
        self.Phi = nn.Linear(hidden_dim, data_dim, bias=False)
        self.Phi_T = TiedTranspose(self.Phi)
        self.M = nn.Linear(hidden_dim, hidden_dim, bias=False) if learning_horizontal else None

    def forward(self, x, a_prev, noise_emb):
        prior_term = self.M(a_prev) if (a_prev is not None and self.M is not None) else 0.0
        constraint_ff = self.Phi_T(x - self.Phi(a_prev)) if a_prev is not None else self.Phi_T(x)
        if a_prev is None:
            a = self.eta * (constraint_ff + noise_emb * self.lam)
        else:
            a = a_prev + self.eta * (constraint_ff + noise_emb * (self.lam + prior_term))
        return F.relu(a)


def run_inference(model, batch, steps=5000):
    a = None
    with torch.no_grad():
        for _ in range(steps):
            a = model(batch, a, 1)
            recon = model.Phi(a)
    return a, recon


def notebook_pipeline_outputs(seed=DATA_SEED):
    torch.manual_seed(seed)

    dataset = AsymmetricCrossDataset(num_samples=10000, d_dim=128)
    dataloader_generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, generator=dataloader_generator)

    grid_dictionary, grid_coords = dataset.get_grid_samples(mu_steps=40, sigma_steps=5)
    sc_net = SCInference(
        data_dim=128,
        hidden_dim=grid_dictionary.shape[0],
        eta=0.01,
        lam=-0.1,
        learning_horizontal=False,
    )
    sc_net.Phi.weight.data = grid_dictionary.T.clone() * 0.5

    sample_batch_0, manifold_indices_0 = next(iter(dataloader))
    a_recon, recon_recon = run_inference(sc_net, sample_batch_0)

    batch_size = 4
    h_blades = 4
    independent_components = []
    coord_list = []
    for _ in range(h_blades):
        mu_rnd = torch.rand(batch_size, 1) * (2 * math.pi)
        sigma_rnd = 0.15 + 0.3 * torch.rand(batch_size, 1)
        amp_rnd = torch.ones(batch_size, 1)
        independent_components.append((mu_rnd, sigma_rnd, amp_rnd))
        coord_list.append(torch.cat([mu_rnd, sigma_rnd, amp_rnd], dim=1))
    ind_batch = dataset.generate(independent_components)
    ind_coords = torch.stack(coord_list, dim=1)
    a_geo, recon_geo = run_inference(sc_net, ind_batch)

    with torch.no_grad():
        a_shuffle = a_geo[:, torch.randperm(len(a_geo[1]))]
        recon_shuffle = sc_net.Phi(a_shuffle)

    sample_batch_1, manifold_indices_1 = next(iter(dataloader))
    a_full, recon_full = run_inference(sc_net, sample_batch_1)

    return {
        "dataset": dataset,
        "grid": dataset.grid.cpu().numpy(),
        "grid_coords": grid_coords.cpu().numpy(),
        "sample_batch_0": sample_batch_0.cpu().numpy(),
        "manifold_indices_0": manifold_indices_0.cpu().numpy(),
        "a_recon": a_recon.cpu().numpy(),
        "recon_recon": recon_recon.cpu().numpy(),
        "ind_batch": ind_batch.cpu().numpy(),
        "ind_coords": ind_coords.cpu().numpy(),
        "a_geo": a_geo.cpu().numpy(),
        "recon_geo": recon_geo.cpu().numpy(),
        "a_shuffle": a_shuffle.cpu().numpy(),
        "recon_shuffle": recon_shuffle.cpu().numpy(),
        "sample_batch_1": sample_batch_1.cpu().numpy(),
        "manifold_indices_1": manifold_indices_1.cpu().numpy(),
        "a_full": a_full.cpu().numpy(),
        "recon_full": recon_full.cpu().numpy(),
    }


def wrapped_angle(theta, center):
    return (theta - center + math.pi) % (2 * math.pi) - math.pi


def build_cross_star(blades, theta):
    signal = np.zeros_like(theta)
    for mu, sigma, amplitude in blades:
        signal += amplitude * np.exp(-np.abs(wrapped_angle(theta, mu)) / sigma)
    return signal


def star_outline(blades, base_radius=0.74, radial_scale=0.82, n_points=720):
    theta = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    radius = base_radius + radial_scale * build_cross_star(blades, theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.append(x, x[0]), np.append(y, y[0])


def draw_star(ax, blades, color, *, lw=2.6, alpha=1.0, linestyle="-", zorder=3):
    x, y = star_outline(blades)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, linestyle=linestyle, zorder=zorder)
    ax.set_aspect("equal")
    extent = max(np.abs(x).max(), np.abs(y).max(), 1.42)
    lim = 1.00 * extent
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")


def ring_xy(signal, grid, base_radius=1.5):
    signal = np.asarray(signal, dtype=float)
    grid = np.asarray(grid, dtype=float)
    signal_closed = np.append(signal, signal[0])
    theta_closed = np.append(grid, grid[0])
    radius = base_radius + signal_closed
    x = radius * np.cos(theta_closed)
    y = radius * np.sin(theta_closed)
    return x, y


def draw_ring_signal(ax, signal, grid, color, *, lw=2.6, alpha=1.0, linestyle="-", zorder=3, limits=None):
    x, y = ring_xy(signal, grid)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, linestyle=linestyle, zorder=zorder)
    ax.set_aspect("equal")
    if limits is None:
        extent = max(np.abs(x).max(), np.abs(y).max(), 1.5)
        lim = 1.02 * extent
    else:
        lim = limits
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")


def signal_from_components(components, grid):
    grid = np.asarray(grid, dtype=float)
    signal = np.zeros_like(grid)
    for theta, sigma, amplitude in components:
        loc = ((grid - theta + math.pi) % (2 * math.pi)) - math.pi
        signal += amplitude * np.exp(-np.abs(loc) / sigma)
    return signal


def ellipse_axes(radial_pos):
    a_inner, a_outer = 0.95, 1.85
    b_inner, b_outer = 0.30, 0.82
    a = a_inner + radial_pos * (a_outer - a_inner)
    b = b_inner + radial_pos * (b_outer - b_inner)
    return a, b


def band_point(theta, radial_pos):
    a, b = ellipse_axes(radial_pos)
    return np.array([a * np.cos(theta), b * np.sin(theta)])


def band_normal(theta, radial_pos):
    a, b = ellipse_axes(radial_pos)
    normal = np.array([np.cos(theta) / a, np.sin(theta) / b])
    normal /= np.linalg.norm(normal)
    return normal


def draw_band(ax, *, theta_landmarks=None, radial_landmarks=None, show_landmarks=False, show_centerline=False):
    if theta_landmarks is None:
        theta_landmarks = THETA_LANDMARKS
    if radial_landmarks is None:
        radial_landmarks = RADIAL_LANDMARKS
    theta_landmarks = np.asarray(theta_landmarks, dtype=float)
    radial_landmarks = np.asarray(radial_landmarks, dtype=float)

    theta = np.linspace(0, 2 * math.pi, 500)
    outer = np.stack([ellipse_axes(1.0)[0] * np.cos(theta), ellipse_axes(1.0)[1] * np.sin(theta)], axis=1)
    inner = np.stack([ellipse_axes(0.0)[0] * np.cos(theta), ellipse_axes(0.0)[1] * np.sin(theta)], axis=1)[::-1]
    patch = Polygon(
        np.vstack([outer, inner]),
        closed=True,
        facecolor=BAND_FILL,
        edgecolor="none",
        alpha=1.0,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.plot(outer[:, 0], outer[:, 1], color=BAND_EDGE, lw=1.25, alpha=0.85)
    ax.plot(inner[:, 0], inner[:, 1], color=BAND_EDGE, lw=1.15, alpha=0.80)

    if show_centerline:
        mid = np.stack([ellipse_axes(0.5)[0] * np.cos(theta), ellipse_axes(0.5)[1] * np.sin(theta)], axis=1)
        ax.plot(mid[:, 0], mid[:, 1], color="#D1D6DE", lw=0.9, linestyle="--", alpha=0.9)

    if show_landmarks:
        for radial_pos in radial_landmarks:
            a, b = ellipse_axes(radial_pos)
            ax.plot(a * np.cos(theta), b * np.sin(theta), color="#D4D9E1", lw=0.55, alpha=0.85, zorder=1)

        for theta_val in theta_landmarks:
            x0, y0 = band_point(theta_val, radial_landmarks[0])
            x1, y1 = band_point(theta_val, radial_landmarks[-1])
            ax.plot([x0, x1], [y0, y1], color="#D4D9E1", lw=0.45, alpha=0.7, zorder=1)

        xs, ys = [], []
        for radial_pos in radial_landmarks:
            a, b = ellipse_axes(radial_pos)
            xs.extend(a * np.cos(theta_landmarks))
            ys.extend(b * np.sin(theta_landmarks))
        ax.scatter(xs, ys, s=9, color=DARK_GRAY, alpha=0.78, linewidths=0, zorder=2)

    ax.set_xlim(-1.88, 1.88)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_spikes(ax, spikes, color, *, lw=1.9, zorder=4, alpha=1.0, end_marker=True, direction=(0.0, 1.0), length_scale=0.43):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    for theta, radial_pos, amplitude in spikes:
        start = band_point(theta, radial_pos)
        end = start + length_scale * amplitude * direction
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )
        if end_marker:
            ax.scatter([end[0]], [end[1]], s=18, color=color, alpha=alpha, linewidths=0, zorder=zorder + 1)


def sigma_to_band(sigma):
    return np.clip((sigma - 0.15) / 0.30, 0.0, 1.0)


def blades_to_spikes(blades):
    return [(mu, sigma_to_band(sigma), 0.95 * amplitude) for mu, sigma, amplitude in blades]


def panel_heading(ax, letter, title, subtitle=None, *, title_fs=12.8, subtitle_fs=8.7, center_title=False):
    ax.axis("off")
    ax.text(0.00, 0.90, letter, fontsize=17, fontweight="bold", color=TEXT, ha="left", va="top")
    ax.text(0.50 if center_title else 0.085, 0.90, title, fontsize=title_fs, fontweight="semibold", color=TEXT, ha="center" if center_title else "left", va="top")
    if subtitle:
        ax.text(0.00, 0.36, subtitle, fontsize=subtitle_fs, color=MUTED, ha="left", va="top", linespacing=1.15)


def arrow_between(fig, ax_left, ax_right, *, color="#71767F", gap_frac=0.52, center_frac=0.5, y_shift=0.0, text=None, text_offset=0.012):
    left_box = ax_left.get_position()
    right_box = ax_right.get_position()
    gap = right_box.x0 - left_box.x1
    y_mid = 0.5 * (left_box.y0 + left_box.y1) + y_shift * (left_box.y1 - left_box.y0)
    x_mid = left_box.x1 + center_frac * gap
    arrow_len = gap_frac * gap
    start = (x_mid - 0.5 * arrow_len, y_mid)
    end = (x_mid + 0.5 * arrow_len, y_mid)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=1.05,
        color=color,
        alpha=0.95,
        transform=fig.transFigure,
        zorder=20,
    )
    fig.add_artist(arrow)
    if text:
        fig.text(
            0.5 * (start[0] + end[0]),
            start[1] + text_offset,
            text,
            fontsize=9.3,
            color=MUTED,
            ha="center",
            va="bottom",
        )


def data_x_to_fig(fig, ax, x, y=0.0):
    xy = ax.transData.transform((x, y))
    return fig.transFigure.inverted().transform(xy)[0]


def arrow_between_content(fig, ax_left, ax_right, left_x, right_x, *, color="#71767F", gap_frac=0.34, y_shift=0.0):
    left_box = ax_left.get_position()
    right_box = ax_right.get_position()
    y_mid = 0.5 * (left_box.y0 + left_box.y1) + y_shift * (left_box.y1 - left_box.y0)
    gap = right_x - left_x
    x_mid = 0.5 * (left_x + right_x)
    tri_w = min(gap_frac * gap, 0.0080)
    aspect = fig.get_figwidth() / fig.get_figheight()
    tri_h = tri_w * aspect * (2.0 / math.sqrt(3.0))
    triangle = Polygon(
        [
            (x_mid - 0.5 * tri_w, y_mid - 0.5 * tri_h),
            (x_mid - 0.5 * tri_w, y_mid + 0.5 * tri_h),
            (x_mid + 0.5 * tri_w, y_mid),
        ],
        closed=True,
        facecolor=color,
        edgecolor="none",
        alpha=0.95,
        transform=fig.transFigure,
        zorder=20,
    )
    fig.add_artist(triangle)


def label_chip(ax, title):
    ax.axis("off")
    ax.text(
        0.50,
        0.50,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.9,
        color=TEXT,
        fontweight="semibold",
        linespacing=1.0,
    )


def angular_distance(a, b):
    return np.abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def nearest_theta_index(theta):
    return int(np.argmin(angular_distance(THETA_LANDMARKS, theta)))


def nearest_radial_index(radial_pos):
    return int(np.argmin(np.abs(RADIAL_LANDMARKS - radial_pos)))


def spike_from_indices(theta_idx, radial_idx, amplitude):
    return (THETA_LANDMARKS[int(theta_idx) % len(THETA_LANDMARKS)], RADIAL_LANDMARKS[int(np.clip(radial_idx, 0, len(RADIAL_LANDMARKS) - 1))], amplitude)


def local_dictionary_cluster(mu, sigma, amplitude):
    theta_idx = nearest_theta_index(mu)
    radial_idx = nearest_radial_index(sigma_to_band(sigma))
    specs = [
        (theta_idx - 1, radial_idx, 0.20 * amplitude),
        (theta_idx, radial_idx, 0.66 * amplitude),
        (theta_idx + 1, radial_idx, 0.20 * amplitude),
    ]
    if radial_idx < len(RADIAL_LANDMARKS) - 1 and sigma_to_band(sigma) > RADIAL_LANDMARKS[radial_idx]:
        specs.append((theta_idx, radial_idx + 1, 0.08 * amplitude))
    elif radial_idx > 0:
        specs.append((theta_idx, radial_idx - 1, 0.08 * amplitude))
    return [spike_from_indices(i, j, amp) for i, j, amp in specs]


def gallery_blades():
    rng = np.random.default_rng(7)
    blades = []
    for _ in range(12):
        angle = rng.uniform(0.0, math.pi / 2.0)
        major_sigma = rng.uniform(0.22, 0.38)
        minor_sigma = rng.uniform(0.15, 0.28)
        major_amp = rng.uniform(1.00, 1.16)
        minor_amp = rng.uniform(0.62, 0.82)
        jitter = rng.normal(0.0, 0.08, size=4)
        blades.append(
            [
                (angle + jitter[0], major_sigma, major_amp),
                (angle + math.pi + jitter[1], major_sigma, major_amp),
                (angle + math.pi / 2 + jitter[2], minor_sigma, minor_amp),
                (angle - math.pi / 2 + jitter[3], minor_sigma, minor_amp),
            ]
        )
    return blades


def cluster_from_true_blades(blades):
    clusters = []
    for mu, sigma, amplitude in blades:
        clusters.extend(local_dictionary_cluster(mu, sigma, amplitude))
    return clusters


def factorial_spikes():
    return [
        spike_from_indices(0, 2, 0.92),
        spike_from_indices(2, 1, 0.52),
        spike_from_indices(6, 0, 0.80),
        spike_from_indices(12, 1, 0.62),
        spike_from_indices(18, 0, 0.94),
        spike_from_indices(22, 2, 0.58),
    ]


def geometric_only_spikes():
    return [
        spike_from_indices(1, 3, 0.34),
        spike_from_indices(2, 3, 0.54),
        spike_from_indices(3, 3, 0.31),
        spike_from_indices(9, 1, 0.30),
        spike_from_indices(10, 1, 0.50),
        spike_from_indices(11, 1, 0.28),
        spike_from_indices(20, 2, 0.34),
        spike_from_indices(21, 2, 0.56),
        spike_from_indices(22, 2, 0.32),
    ]


def cooccurrence_spikes():
    true_blades = [
        (0.42, 0.34, 1.10),
        (0.42 + math.pi, 0.34, 1.10),
        (0.42 + math.pi / 2, 0.21, 0.70),
        (0.42 - math.pi / 2, 0.21, 0.70),
    ]
    return cluster_from_true_blades(true_blades)


def spikes_to_blades(spikes):
    return [(theta, 0.16 + 0.30 * radial_pos, amplitude) for theta, radial_pos, amplitude in spikes]


def normalize_sigma(sigma, sigma_min, sigma_max):
    return (sigma - sigma_min) / (sigma_max - sigma_min + 1e-8)


def notebook_landmarks(grid_coords):
    theta_landmarks = np.unique(grid_coords[:, 0])
    sigma_levels = np.unique(grid_coords[:, 1])
    sigma_min = float(sigma_levels.min())
    sigma_max = float(sigma_levels.max())
    radial_landmarks = normalize_sigma(sigma_levels, sigma_min, sigma_max)
    return theta_landmarks, radial_landmarks, sigma_min, sigma_max


def notebook_true_spikes(query_coords, sigma_min, sigma_max):
    spikes = []
    for mu, sigma, amplitude in query_coords:
        radial_pos = normalize_sigma(sigma, sigma_min, sigma_max)
        gt_len = 0.6 * ((sigma * 2.0) / 1.2)
        spikes.append((float(mu), float(radial_pos), float(gt_len)))
    return spikes


def notebook_sparse_spikes(grid_coords, sparse_code, sigma_min, sigma_max, threshold=1e-3):
    max_a = float(np.max(np.abs(sparse_code)))
    spike_scale = 0.6 / max_a if max_a > 0 else 0.0
    spikes = []
    for (mu, sigma), value in zip(grid_coords, sparse_code):
        if abs(value) > threshold:
            radial_pos = normalize_sigma(sigma, sigma_min, sigma_max)
            spikes.append((float(mu), float(radial_pos), float(value * spike_scale)))
    return spikes


def ring_limit_from_batch(batch_data, base_radius=1.5, margin=0.2):
    return base_radius + float(np.max(batch_data)) + margin


def main():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.8, 4.25))
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.462, 0.538],
        hspace=0.0,
        left=0.04,
        right=0.985,
        top=0.992,
        bottom=0.015,
    )

    outputs = notebook_pipeline_outputs()
    signal_grid = outputs["grid"]
    grid_coords = outputs["grid_coords"]
    theta_landmarks, radial_landmarks, sigma_min, sigma_max = notebook_landmarks(grid_coords)

    panel_a_signals = outputs["sample_batch_0"][:8] - 1.0
    panel_a_limit = ring_limit_from_batch(panel_a_signals)

    example_idx = 4
    true_query = outputs["manifold_indices_0"][example_idx]
    true_signal = outputs["sample_batch_0"][example_idx] - 1.0
    recon_signal = outputs["recon_recon"][example_idx] - 1.0
    true_spikes = notebook_true_spikes(true_query, sigma_min, sigma_max)
    recon_spikes = notebook_sparse_spikes(grid_coords, outputs["a_recon"][example_idx], sigma_min, sigma_max)
    panel_c_limit = ring_limit_from_batch(np.stack([true_signal, recon_signal], axis=0))

    viz_idx = 1
    d_specs = [
        (
            "Factorial",
            notebook_sparse_spikes(grid_coords, outputs["a_shuffle"][viz_idx], sigma_min, sigma_max),
            outputs["recon_shuffle"][viz_idx] - 1.0,
        ),
        (
            "Geometric",
            notebook_sparse_spikes(grid_coords, outputs["a_geo"][viz_idx], sigma_min, sigma_max),
            outputs["recon_geo"][viz_idx] - 1.0,
        ),
        (
            "Co-occurrence",
            notebook_sparse_spikes(grid_coords, outputs["a_full"][viz_idx], sigma_min, sigma_max),
            outputs["recon_full"][viz_idx] - 1.0,
        ),
    ]

    top = outer[0, 0].subgridspec(1, 3, width_ratios=[2.36, 4.40, 4.42], wspace=0.09)

    panel_a = top[0, 0].subgridspec(2, 1, height_ratios=[0.026, 1.0], hspace=0.00)
    panel_a_head = fig.add_subplot(panel_a[0, 0])
    panel_heading(panel_a_head, "A", "Data distribution", title_fs=10.9, center_title=True)
    sub_a = panel_a[1, 0].subgridspec(2, 4, wspace=0.045, hspace=-0.50)
    for idx, signal in enumerate(panel_a_signals):
        ax = fig.add_subplot(sub_a[idx // 4, idx % 4])
        draw_ring_signal(ax, signal, signal_grid, RED, lw=2.0, alpha=0.96, limits=panel_a_limit)

    panel_b = top[0, 1].subgridspec(3, 1, height_ratios=[0.026, 0.082, 1.0], hspace=0.00)
    panel_b_head = fig.add_subplot(panel_b[0, 0])
    panel_heading(panel_b_head, "B", "Latent manifold", title_fs=10.9, center_title=True)
    sub_b = panel_b[2, 0].subgridspec(1, 2, width_ratios=[1.78, 0.92], wspace=0.08)
    ax_b_band = fig.add_subplot(sub_b[0, 0])
    draw_band(ax_b_band, theta_landmarks=theta_landmarks, radial_landmarks=radial_landmarks, show_landmarks=False, show_centerline=False)
    draw_spikes(ax_b_band, true_spikes, RED, lw=2.4, length_scale=1.0)
    ax_b_star = fig.add_subplot(sub_b[0, 1])
    draw_ring_signal(ax_b_star, true_signal, signal_grid, RED, lw=2.8, limits=panel_c_limit)

    panel_c = top[0, 2].subgridspec(3, 1, height_ratios=[0.026, 0.082, 1.0], hspace=0.00)
    panel_c_head = fig.add_subplot(panel_c[0, 0])
    panel_heading(panel_c_head, "C", "Sparse coding reconstruction", title_fs=10.9, center_title=True)
    sub_c = panel_c[2, 0].subgridspec(1, 2, width_ratios=[1.84, 1.01], wspace=0.07)
    ax_c_band = fig.add_subplot(sub_c[0, 0])
    draw_band(ax_c_band, theta_landmarks=theta_landmarks, radial_landmarks=radial_landmarks, show_landmarks=True, show_centerline=False)
    draw_spikes(ax_c_band, true_spikes, RED, lw=2.1, alpha=0.95, length_scale=1.0)
    draw_spikes(ax_c_band, recon_spikes, BLUE, lw=1.55, alpha=0.95, length_scale=1.0)
    ax_c_star = fig.add_subplot(sub_c[0, 1])
    draw_ring_signal(ax_c_star, true_signal, signal_grid, RED, lw=2.8, alpha=0.72, limits=panel_c_limit)
    draw_ring_signal(ax_c_star, recon_signal, signal_grid, BLUE, lw=2.2, limits=panel_c_limit)

    panel_d_anchor = fig.add_subplot(outer[1, 0])
    panel_d_anchor.axis("off")
    panel_d_anchor.text(0.00, 0.99, "D", fontsize=17, fontweight="bold", color=TEXT, ha="left", va="top")
    panel_d_anchor.text(0.50, 0.99, "Sparse coding priors", fontsize=10.9, fontweight="semibold", color=TEXT, ha="center", va="top")
    sub_d = outer[1, 0].subgridspec(1, 3, wspace=0.11)

    d_band_axes = []
    d_star_axes = []
    for idx, (title, spikes, generated_signal) in enumerate(d_specs):
        panel_limit = ring_limit_from_batch(np.expand_dims(generated_signal, axis=0))
        panel = sub_d[0, idx].subgridspec(3, 1, height_ratios=[0.145, 1.0, 0.09], hspace=0.00)
        content = panel[1, 0].subgridspec(1, 2, width_ratios=[1.72, 0.98], wspace=0.08)
        ax_band = fig.add_subplot(content[0, 0])
        draw_band(ax_band, theta_landmarks=theta_landmarks, radial_landmarks=radial_landmarks, show_landmarks=True, show_centerline=False)
        draw_spikes(ax_band, spikes, BLUE, lw=1.95, length_scale=1.0)
        ax_star = fig.add_subplot(content[0, 1])
        draw_ring_signal(ax_star, generated_signal, signal_grid, BLUE, lw=2.4, limits=panel_limit)
        label_ax = fig.add_subplot(panel[2, 0])
        label_chip(label_ax, title)
        d_band_axes.append(ax_band)
        d_star_axes.append(ax_star)

    fig.canvas.draw()
    b_x, _ = ring_xy(true_signal, signal_grid)
    c_true_x, _ = ring_xy(true_signal, signal_grid)
    c_recon_x, _ = ring_xy(recon_signal, signal_grid)
    arrow_between_content(
        fig,
        ax_b_band,
        ax_b_star,
        data_x_to_fig(fig, ax_b_band, 1.85),
        data_x_to_fig(fig, ax_b_star, float(np.min(b_x))),
        gap_frac=0.36,
    )
    arrow_between_content(
        fig,
        ax_c_band,
        ax_c_star,
        data_x_to_fig(fig, ax_c_band, 1.85),
        data_x_to_fig(fig, ax_c_star, float(min(np.min(c_true_x), np.min(c_recon_x)))),
        gap_frac=0.34,
    )
    for (ax_band, ax_star), (_, _, generated_signal) in zip(zip(d_band_axes, d_star_axes), d_specs):
        d_x, _ = ring_xy(generated_signal, signal_grid)
        arrow_between_content(
            fig,
            ax_band,
            ax_star,
            data_x_to_fig(fig, ax_band, 1.85),
            data_x_to_fig(fig, ax_star, float(np.min(d_x))),
            gap_frac=0.32,
        )

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
