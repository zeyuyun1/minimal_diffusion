import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimizer import weight_norm


def _gauss_kernel2d(ks: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    return g2.view(1, 1, ks, ks)


class GaussianPyramidEncoder(nn.Module):
    """Multi-level Gaussian/Laplacian pyramid encoder/decoder."""

    def __init__(
        self,
        levels=2,
        kernel_size=5,
        sigma=1.2,
        concat_to_channels=True,
        learnable=False,
        reflect_pad=True,
    ):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.ks = int(kernel_size)
        self.sigma = float(sigma)
        self.concat = concat_to_channels
        self.reflect_pad = reflect_pad

        self.register_buffer("k2d", _gauss_kernel2d(self.ks, self.sigma))
        self.learnable = learnable
        self.k2d_param = nn.Parameter(self.k2d.clone()) if learnable else None

    def _blur(self, x):
        _, c, _, _ = x.shape
        k = self.k2d_param if self.learnable else self.k2d
        k = k.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        pad = self.ks // 2
        if self.reflect_pad:
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            pad = 0
        return F.conv2d(x, k, padding=pad, groups=x.size(1))

    def forward(self, x):
        _, _, h, w = x.shape
        xs = x
        residuals = []

        for _ in range(self.levels):
            b_next = F.avg_pool2d(self._blur(xs), kernel_size=2, stride=2)
            up_next = F.interpolate(b_next, size=xs.shape[-2:], mode="bilinear", align_corners=False)
            residuals.append(xs - up_next)
            xs = b_next

        b_last = xs
        pyr = residuals + [b_last]

        if not self.concat:
            return pyr

        ups = []
        for r_s in residuals:
            if r_s.shape[-2:] != (h, w):
                r_s = F.interpolate(r_s, size=(h, w), mode="bilinear", align_corners=False)
            ups.append(r_s)
        ups.append(F.interpolate(b_last, size=(h, w), mode="bilinear", align_corners=False))
        return torch.cat(ups, dim=1)

    def decode(self, pyr):
        assert isinstance(pyr, (list, tuple))
        assert len(pyr) == self.levels + 1

        residuals = pyr[:-1]
        current = pyr[-1]

        for r_s in reversed(residuals):
            if current.shape[-2:] != r_s.shape[-2:]:
                current = F.interpolate(current, size=r_s.shape[-2:], mode="bilinear", align_corners=False)
            current = r_s + current
        return current


def sample_uniformly(n1, n2):
    return random.randint(int(n1), int(n2))


def _sample_discrete_laplace_unbounded(center=0, b=2.0):
    q = math.exp(-1.0 / float(b))
    p = 1.0 - q
    g1 = np.random.geometric(p)
    g2 = np.random.geometric(p)
    return int(g1 - g2)


def sample_uniformly_with_long_tail(n1, n2, b=2.5, mixer_value=0.0, center=None):
    n1, n2 = int(n1), int(n2)
    if center is None:
        center = (n1 + n2) // 2
    if random.random() < float(1 - mixer_value):
        return sample_uniformly(n1, n2)
    k = center + _sample_discrete_laplace_unbounded(center=0, b=b)
    return int(max(n1, k))


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class TiedTransposeConv(nn.Module):
    """Conv-transpose tied to a reference conv weight."""

    def __init__(self, conv, output_padding=1):
        super().__init__()
        self.conv = conv
        self.output_padding = output_padding

    def forward(self, x):
        return F.conv_transpose2d(
            x,
            self.conv.weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            output_padding=self.output_padding,
            groups=self.conv.groups,
        )

class encoder_node7(nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis,
        kernel_size=7,
        stride=2,
        padding=3,
        eta=0.5,
        output_padding=1,
        wnorm=True,
    ):
        super().__init__()

        if in_channels:
            self.encoder = nn.Conv2d(
                in_channels,
                num_basis,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        else:
            self.encoder = self.decoder = None

        self.eta = torch.tensor(eta)
        self.num_basis = num_basis
        self._wnorm = bool(wnorm)

        if self._wnorm:
            if self.encoder is not None:
                self.encoder, self._enc_wn = weight_norm(self.encoder, names=["weight"], dim=0)

    def reset_wnorm(self):
        if not getattr(self, "_wnorm", False):
            return
        if self.encoder is not None:
            self._enc_wn.reset(self.encoder)

    def forward(
        self,
        a_prev=None,
        x_c=None,
        noise_emb=None,
    ):
        self.reset_wnorm()
        constraint_ff = self.encoder(x_c) if x_c is not None and self.encoder is not None else None

        if a_prev is None and constraint_ff is None:
            return None, None
        if a_prev is None:
            a_prev = torch.zeros_like(constraint_ff)
        if constraint_ff is None:
            constraint_ff = torch.zeros_like(a_prev)
        noise_emb = noise_emb if noise_emb is not None else 0

        a = a_prev + self.eta * (constraint_ff * noise_emb[0])
        decoded = self.decoder(a) if self.decoder is not None else None
        return a, decoded


class prior_node7(nn.Module):
    def __init__(
        self,
        num_basis_prev,
        num_basis,
        eta=0.5,
        learning_horizontal=True,
        relu_6=True,
        wnorm=True,
        k_inter=None,
        down=True,
        tied_transpose=True,
    ):
        super().__init__()
        m_stride = 2 if down else 1
        output_padding = 1 if down else 0

        if learning_horizontal:
            self.M_inter = nn.Conv2d(num_basis, num_basis, kernel_size=3, padding=1, bias=False)
        else:
            self.M_inter = None

        if num_basis_prev:
            self.M_intra = nn.Conv2d(
                num_basis_prev, num_basis, stride=m_stride, kernel_size=3, padding=1, bias=False
            )
            if tied_transpose:
                self.M_intra_T = TiedTransposeConv(self.M_intra, output_padding=output_padding)
            else:
                self.M_intra_T = nn.ConvTranspose2d(
                    num_basis,
                    num_basis_prev,
                    stride=m_stride,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    output_padding=output_padding,
                )
        else:
            self.M_intra = self.M_intra_T = None

        self.tied_transpose = tied_transpose
        self.eta = torch.tensor(eta)
        self.relu = nn.ReLU6() if relu_6 else nn.ReLU()
        self.num_basis = num_basis
        self._wnorm = bool(wnorm)
        self.threshold_prior = nn.Parameter(torch.zeros(1, num_basis, 1, 1))

        if self._wnorm:
            if self.M_inter is not None:
                self.M_inter, self._M_inter_wn = weight_norm(self.M_inter, names=["weight"], dim=0, k=k_inter)
            if self.M_intra is not None:
                self.M_intra, self._M_intra_wn = weight_norm(self.M_intra, names=["weight"], dim=0)
            if self.M_intra_T is not None and not self.tied_transpose:
                self.M_intra_T, self._M_intra_T_wn = weight_norm(self.M_intra_T, names=["weight"], dim=0)

    def reset_wnorm(self):
        if not getattr(self, "_wnorm", False):
            return
        if self.M_inter is not None:
            self._M_inter_wn.reset(self.M_inter)
        if self.M_intra is not None:
            self._M_intra_wn.reset(self.M_intra)
        if self.M_intra_T is not None and not self.tied_transpose:
            self._M_intra_T_wn.reset(self.M_intra_T)

    def forward(self, a_p_1=None, a_prev=None, top_signal=None, noise_emb=None):
        self.reset_wnorm()
        noise_prior = noise_emb[1] if isinstance(noise_emb, (list, tuple)) and len(noise_emb) > 1 else 0
        noise_threshold = noise_emb[2] if isinstance(noise_emb, (list, tuple)) and len(noise_emb) > 2 else 0

        feedback = top_signal if top_signal is not None else 0
        feedup = self.M_intra(a_p_1) if a_p_1 is not None and self.M_intra is not None else 0
        a_inter = self.M_inter(a_prev) if self.M_inter is not None and a_prev is not None else 0
        a_prev = a_prev if a_prev is not None else 0

        prior_term = feedup + a_inter + feedback
        a = a_prev + self.eta * (prior_term * noise_prior + self.threshold_prior * noise_threshold)
        a = self.relu(a)
        upstream_grad = self.M_intra_T(a) if self.M_intra_T is not None else 0
        return a, upstream_grad


class neural_node7(nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis_prev,
        num_basis,
        kernel_size=7,
        stride=2,
        padding=3,
        eta=0.5,
        output_padding=1,
        learning_horizontal=True,
        bias=False,
        relu_6=True,
        wnorm=True,
        constraint_energy="SC",
        k_inter=None,
        down=True,
        tied_transpose=True,
        film=False,
        ignore_ff_control=False,
        n_hid_layers=1,
    ):
        super().__init__()
        self.encoder_node = encoder_node7(
            in_channels=in_channels,
            num_basis=num_basis,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            eta=eta,
            output_padding=output_padding,
            wnorm=wnorm,
        )
        self.prior_node = prior_node7(
            num_basis_prev=num_basis_prev,
            num_basis=num_basis,
            eta=eta,
            learning_horizontal=learning_horizontal,
            relu_6=relu_6,
            wnorm=wnorm,
            k_inter=k_inter,
            down=down,
            tied_transpose=tied_transpose,
        )

        self.num_basis = num_basis
        self.decoder = self.encoder_node.decoder
        # self.M_intra_T = self.prior_node.M_intra_T

    def reset_wnorm(self):
        self.encoder_node.reset_wnorm()
        self.prior_node.reset_wnorm()

    def forward(
        self,
        a_p_1=None,
        a_prev=None,
        x_c=None,
        x_in=None,
        top_signal=None,
        noise_emb=None,
        T=0,
        steer_components=None,
        return_feature=False,
    ):
        a, _ = self.encoder_node(a_prev=a_prev, x_c=x_c, noise_emb=noise_emb)
        a, upstream_grad = self.prior_node(a_p_1=a_p_1, a_prev=a, top_signal=top_signal, noise_emb=noise_emb)
        decoded = self.decoder(a) if self.decoder is not None else None
        return a, upstream_grad, decoded, None


class neural_sheet7(nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis,
        kernel_size=7,
        stride=2,
        output_padding=1,
        learning_horizontal=True,
        eta_base=0.1,
        jfb_no_grad_iters=None,
        jfb_with_grad_iters=None,
        channel_mult_emb=2,
        channel_mult_noise=1,
        jfb_reuse_solution=0,
        mixer_value=0.0,
        noise_embedding=True,
        bias=True,
        relu_6=False,
        T=0.1,
        constraint_energy="SC",
        per_dim_threshold=True,
        positive_threshold=False,
        multiscale=False,
        intra=True,
        k_inter=None,
        n_hid_layers=1,
        tied_transpose=True,
        film=False,
        scalar_mul=False,
        ff_scale=False,
        control_groups=None,
        ignore_ff_control=False,
        simple_control=False,
    ):
        super().__init__()

        self.eta_base = eta_base
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.mixer_value = mixer_value
        self.T = T
        self.positive_threshold = positive_threshold
        self.multiscale = multiscale
        self.num_scales = len(num_basis)
        self.constraint_energy = constraint_energy
        self.film = film
        self.ff_scale = ff_scale
        # None keeps legacy behavior (per-dim if enabled, else scalar).
        # Set to an int (e.g. 1, 4, 8) for grouped control modulation.
        self.control_groups = control_groups

        if multiscale:
            self.encoder_0 = GaussianPyramidEncoder(levels=self.num_scales - 1, concat_to_channels=False)
            self.decoder_0 = self.encoder_0.decode
            self.injection_ls = [in_channels] * self.num_scales
        else:
            if len(num_basis) == 1:
                self.encoder_0 = lambda x: [x]
                self.decoder_0 = lambda x: x[0]
                self.injection_ls = [in_channels]
            else:
                self.encoder_0 = lambda x: [x] + [None for _ in range(self.num_scales - 1)]
                self.decoder_0 = lambda x: x[0]
                self.injection_ls = [in_channels] + [None] * (self.num_scales - 1)

        levels = []
        prev_channels = None
        counter = 0
        for i, nb in enumerate(num_basis):
            in_channels = self.injection_ls[i]
            levels.append(
                neural_node7(
                    in_channels,
                    prev_channels,
                    nb,
                    kernel_size=kernel_size,
                    eta=eta_base,
                    stride=stride,
                    padding=kernel_size // 2 if i > 0 else 3,
                    output_padding=output_padding,
                    learning_horizontal=learning_horizontal,
                    bias=bias,
                    relu_6=relu_6,
                    constraint_energy=constraint_energy,
                    k_inter=k_inter,
                    tied_transpose=tied_transpose,
                    n_hid_layers=n_hid_layers,
                )
            )
            counter += 1

            if intra:
                prev_channels = nb
        self.levels = nn.ModuleList(levels)

        self.n_levels = len(levels)
        self.noise_embedding = noise_embedding
        
        if noise_embedding:
            emb_channels = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise

            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            out_dims = [node.num_basis if per_dim_threshold else 1 for node in self.levels]
            # self.affines = nn.ModuleList([nn.Linear(emb_channels, d, bias=True) for d in out_dims])
            self.affines = nn.ModuleList(
                [nn.ModuleList([nn.Linear(emb_channels, d, bias=True) for d in out_dims]) for _ in range(1 if simple_control else 3)]
            )

    def _reset_all_wnorm(self):
        if not getattr(self, "_wnorm", False):
            return
        for lvl in self.levels:
            if hasattr(lvl, "reset_wnorm"):
                lvl.reset_wnorm()

    
    def film_modulation(self, noise_labels, i):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(len(noise_labels), 2, -1).flip(1).reshape(len(noise_labels), -1)
        emb = F.relu(self.map_layer0(emb))
        normalize_emb = []
        for j in range(len(self.affines)):
            ctrl_j = (1 + torch.tanh(self.affines[j][i](emb))) * 2  # [B, d_i]
            ctrl_j = ctrl_j.unsqueeze(-1).unsqueeze(-1)          # [B, d_i, 1, 1]
            normalize_emb.append(ctrl_j)

        if len(normalize_emb) == 1:
            normalize_emb = [1,normalize_emb[0],normalize_emb[0]]
        return normalize_emb

    
    def forward(
        self,
        x,
        a=None,
        upstream_grad=None,
        noise_labels=None,
        T=None,
        return_feature=False,
        infer_mode=False,
        n_iters=None,
        neuron_steer=None,
        return_history=False,
        class_labels=None,
        gamma_scale=None,
        ablate_noi=None,
        ablate_mode="spatial_mean",
        measurement = None, # (measurement, operator : image -> measurement)
    ):
        bsz = x.shape[0]
        if T is None:
            T = self.T
        self._reset_all_wnorm()

        if noise_labels is None:
            noise_labels = torch.zeros(bsz, device=x.device, dtype=x.dtype)

        if self.noise_embedding:
            noise_emb_ls = [self.film_modulation(noise_labels, i) for i in range(self.n_levels)]
        else:
            noise_emb_ls = [self.lambda_bias for _ in range(self.n_levels)]

        x_in = None

        if a is None:
            a = [None] * self.n_levels

        if upstream_grad is None:
            upstream_grad = [None] * self.n_levels
        decoded = [None] * self.n_levels

        n0 = sample_uniformly_with_long_tail(
            self.jfb_no_grad_iters[0],
            self.jfb_no_grad_iters[1],
            mixer_value=self.mixer_value,
        )
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        if infer_mode:
            n0 = n_iters
        history = [] if return_history else None
        if n0 > 0:
            with torch.no_grad():
                for _ in range(n0):
                    snaps = self.forward_inter(
                        x,
                        x_in,
                        a,
                        upstream_grad,
                        decoded,
                        noise_emb_ls,
                        T=T,
                        return_history=return_history,
                        ablate_noi=ablate_noi,
                        ablate_mode=ablate_mode,
                        measurement=measurement,
                    )
                    if return_history:
                        history.extend(snaps)

        a = [ai.detach() if ai is not None else None for ai in a]
        for _ in range(m1):
            snaps = self.forward_inter(
                x,
                x_in,
                a,
                upstream_grad,
                decoded,
                noise_emb_ls,
                ablate_noi=ablate_noi,
                ablate_mode=ablate_mode,
                measurement=measurement,
                return_history=return_history,
            )
            if return_history:
                history.extend(snaps)

        decoded = [decoded[i] for i in range(self.n_levels)]

        if return_history:
            return history
        if return_feature:
            a = [ai.detach().clone() if ai is not None else None for ai in a]
            upstream_grad = [ui.detach().clone() if isinstance(ui, torch.Tensor) else 0 for ui in upstream_grad]
            return {
                "a": a,
                "decoded": decoded,
                "upstream_grad": upstream_grad,
                "denoised": self.decoder_0(decoded),
            }
        return self.decoder_0(decoded)

    def forward_dynamics(self,a,res_in,upstream_grad,decoded,noise_emb_ls,T,ablate_noi,ablate_mode, reverse = False):
        order = range(self.n_levels - 1, -1, -1) if reverse else range(self.n_levels)
        for i in order:
            inp = None if i == 0 else a[i - 1]
            top_signal = upstream_grad[i + 1] if i < self.n_levels - 1 else None
            a[i], upstream_grad[i], decoded[i], _ = self.levels[i](
                inp,
                a_prev=a[i],
                x_c=res_in[i],
                top_signal=top_signal,
                noise_emb=noise_emb_ls[i],
                T=T,
            )
            a[i] = self._apply_noi_ablation(a[i], i, ablate_noi=ablate_noi, ablate_mode=ablate_mode)

    def _snapshot(self, a, decoded, upstream_grad):
        """Detached clone of full state for history recording."""
        return {
            "a": [ai.detach().clone() if ai is not None else None for ai in a],
            "decoded": [d.detach().clone() if d is not None else None for d in decoded],
            "upstream_grad": [u.detach().clone() if isinstance(u, torch.Tensor) else u for u in upstream_grad],
            "denoised": self.decoder_0([d.detach() if d is not None else None for d in decoded]),
        }

    def forward_inter(
        self,
        x,
        x_in,
        a,
        upstream_grad,
        decoded,
        noise_emb_ls=None,
        T=0,
        return_history=False,
        ablate_noi=None,
        ablate_mode="spatial_mean",
        measurement=None,
    ):
        if self.constraint_energy == "SC" and decoded[0] is not None:
            res_in = self.encoder_0(x - self.decoder_0(decoded))
        else:
            res_in = self.encoder_0(x)

        self.forward_dynamics(a, res_in, upstream_grad, decoded, noise_emb_ls, T, ablate_noi, ablate_mode, reverse=False)
        snap1 = self._snapshot(a, decoded, upstream_grad) if return_history else None
        self.forward_dynamics(a, res_in, upstream_grad, decoded, noise_emb_ls, T, ablate_noi, ablate_mode, reverse=True)
        snap2 = self._snapshot(a, decoded, upstream_grad) if return_history else None

        if return_history:
            return [snap1, snap2]
        return None

    def _apply_noi_ablation(self, a_i, level_idx, ablate_noi=None, ablate_mode="spatial_mean"):
        """
        Ablate selected channels in a level latent state.

        Args:
            a_i: Tensor [B, C, H, W]
            level_idx: int level index
            ablate_noi: dict[level_idx] -> indices to ablate
            ablate_mode: "spatial_mean" | "batch_spatial_mean" | "zero"
        """
        if a_i is None or ablate_noi is None or level_idx not in ablate_noi:
            return a_i

        idx = torch.as_tensor(ablate_noi[level_idx], device=a_i.device, dtype=torch.long)
        if idx.numel() == 0:
            return a_i
        idx = idx[(idx >= 0) & (idx < a_i.size(1))]
        if idx.numel() == 0:
            return a_i

        out = a_i.clone()
        if ablate_mode == "zero":
            out[:, idx] = 0
        elif ablate_mode == "batch_spatial_mean":
            fill = out[:, idx].mean(dim=(0, 2, 3), keepdim=True)
            out[:, idx] = fill.expand_as(out[:, idx])
        else:  # default: "spatial_mean"
            fill = out[:, idx].mean(dim=(2, 3), keepdim=True)
            out[:, idx] = fill.expand_as(out[:, idx])
        return out


    # if return_history:
    #     history = []
    #     for i in range(len(self.levels)):
    #         x_in = self.encoder(x)
    #         inp = None if i == 0 else a[i - 1]
    #         top_signal = upstream_grad[i + 1] if i < self.n_levels - 1 else None
    #         x_in_i = x_in[self.injection_dic[i]] if i in self.injection_dic else None
    #         x_c = res_in[self.injection_dic[i]] if i in self.injection_dic else None
    #         _, _, _, feature = self.levels[i](
    #             inp,
    #             a_prev=a[i],
    #             x_c=x_c,
    #             top_signal=top_signal,
    #             x_in=x_in_i,
    #             noise_emb=noise_emb_ls[i],
    #             T=T,
    #             return_feature=return_history,
    #         )
    #         history.append(feature)
    #     return [history]



class encoder_node6(nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis,
        kernel_size=7,
        stride=2,
        padding=3,
        eta=0.5,
        output_padding=1,
        wnorm=True,
        relu_6=True,
    ):
        super().__init__()

        if in_channels:
            self.encoder = nn.Conv2d(
                in_channels,
                num_basis,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        else:
            self.encoder = self.decoder = None

        self.eta = torch.tensor(eta)
        self.num_basis = num_basis
        self._wnorm = bool(wnorm)
        self.relu = nn.ReLU6() if relu_6 else nn.ReLU()

        if self._wnorm:
            if self.encoder is not None:
                self.encoder, self._enc_wn = weight_norm(self.encoder, names=["weight"], dim=0)

    def reset_wnorm(self):
        if not getattr(self, "_wnorm", False):
            return
        if self.encoder is not None:
            self._enc_wn.reset(self.encoder)

    def forward(
        self,
        a_prev=None,
        x_c=None,
        noise_emb=None,
    ):
        self.reset_wnorm()
        if isinstance(x_c, tuple):
            x_c, x_m = x_c
            constraint_m = self.encoder(x_m) if (x_m is not None and self.encoder is not None) else 0
        else:
            x_m = None
            constraint_m = 0
        constraint_ff = self.encoder(x_c) if x_c is not None and self.encoder is not None else None

        if a_prev is None and constraint_ff is None:
            return None, None
        if a_prev is None:
            a_prev = torch.zeros_like(constraint_ff)
        if constraint_ff is None:
            constraint_ff = torch.zeros_like(a_prev)
        noise_emb = noise_emb if noise_emb is not None else 0

        a = a_prev + self.eta * (constraint_ff * noise_emb[0] + constraint_m * noise_emb[3])
        return a


class prior_node6(nn.Module):
    def __init__(
        self,
        num_basis_prev,
        num_basis,
        eta=0.5,
        learning_horizontal=True,
        relu_6=True,
        wnorm=True,
        k_inter=None,
        down=True,
        tied_transpose=True,
    ):
        super().__init__()
        m_stride = 2 if down else 1
        output_padding = 1 if down else 0

        if learning_horizontal:
            self.M_inter = nn.Conv2d(num_basis, num_basis, kernel_size=3, padding=1, bias=False)
        else:
            self.M_inter = None

        if num_basis_prev:
            self.M_intra = nn.Conv2d(
                num_basis_prev, num_basis, stride=m_stride, kernel_size=3, padding=1, bias=False
            )
            if tied_transpose:
                self.M_intra_T = TiedTransposeConv(self.M_intra, output_padding=output_padding)
            else:
                self.M_intra_T = nn.ConvTranspose2d(
                    num_basis,
                    num_basis_prev,
                    stride=m_stride,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    output_padding=output_padding,
                )
        else:
            self.M_intra = self.M_intra_T = None

        self.tied_transpose = tied_transpose
        self.eta = torch.tensor(eta)
        self.relu = nn.ReLU6() if relu_6 else nn.ReLU()
        self.num_basis = num_basis
        self._wnorm = bool(wnorm)
        # self.threshold_prior = nn.Parameter(torch.zeros(1, num_basis, 1, 1))

        if self._wnorm:
            if self.M_inter is not None:
                self.M_inter, self._M_inter_wn = weight_norm(self.M_inter, names=["weight"], dim=0, k=k_inter)
            if self.M_intra is not None:
                self.M_intra, self._M_intra_wn = weight_norm(self.M_intra, names=["weight"], dim=0)
            if self.M_intra_T is not None and not self.tied_transpose:
                self.M_intra_T, self._M_intra_T_wn = weight_norm(self.M_intra_T, names=["weight"], dim=0)

    def reset_wnorm(self):
        if not getattr(self, "_wnorm", False):
            return
        if self.M_inter is not None:
            self._M_inter_wn.reset(self.M_inter)
        if self.M_intra is not None:
            self._M_intra_wn.reset(self.M_intra)
        if self.M_intra_T is not None and not self.tied_transpose:
            self._M_intra_T_wn.reset(self.M_intra_T)

    def forward(self, s_p_1=None, a_prev=None, s_prev=None, top_signal=None, noise_emb=None):
        self.reset_wnorm()
        noise_prior = noise_emb[1] if isinstance(noise_emb, (list, tuple)) and len(noise_emb) > 1 else 0
        noise_threshold = noise_emb[2] if isinstance(noise_emb, (list, tuple)) and len(noise_emb) > 2 else 0

        feedback = top_signal if top_signal is not None else 0
        feedup = self.M_intra(s_p_1) if s_p_1 is not None and self.M_intra is not None else 0
        a_inter = self.M_inter(s_prev) if self.M_inter is not None and s_prev is not None else 0
        a_prev = a_prev if a_prev is not None else 0

        prior_term = feedup + a_inter + feedback
        a = a_prev + self.eta * (prior_term * noise_prior)
        upstream_grad = self.M_intra_T(a) if self.M_intra_T is not None else 0
        return a, upstream_grad


class neural_node6(nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis_prev,
        num_basis,
        kernel_size=7,
        stride=2,
        padding=3,
        eta=0.5,
        output_padding=1,
        learning_horizontal=True,
        bias=False,
        relu_6=True,
        wnorm=True,
        constraint_energy="SC",
        k_inter=None,
        down=True,
        tied_transpose=True,
        film=False,
        ignore_ff_control=False,
        n_hid_layers=1,
    ):
        super().__init__()
        self.encoder_node = encoder_node6(
            in_channels=in_channels,
            num_basis=num_basis,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            eta=eta,
            output_padding=output_padding,
            wnorm=wnorm,
            relu_6=relu_6,
        )
        self.prior_node = prior_node6(
            num_basis_prev=num_basis_prev,
            num_basis=num_basis,
            eta=eta,
            learning_horizontal=learning_horizontal,
            relu_6=relu_6,
            wnorm=wnorm,
            k_inter=k_inter,
            down=down,
            tied_transpose=tied_transpose,
        )

        self.num_basis = num_basis
        self.decoder = self.encoder_node.decoder
        self.relu = nn.ReLU6() if relu_6 else nn.ReLU()
        self.threshold_prior = nn.Parameter(torch.zeros(1, num_basis, 1, 1))
        # self.M_intra_T = self.prior_node.M_intra_T

    def reset_wnorm(self):
        self.encoder_node.reset_wnorm()
        self.prior_node.reset_wnorm()

    def _shrink(self, u, noise_emb=None):
        if u is None:
            return None
        noise_threshold = noise_emb[2] if isinstance(noise_emb, (list, tuple)) and len(noise_emb) > 2 else 0
        return self.relu(u - self.threshold_prior * noise_threshold)

    def forward(
        self,
        s_p_1=None,
        a_prev=None,
        x_c=None,
        x_in=None,
        top_signal=None,
        noise_emb=None,
        T=0,
        steer_components=None,
        return_feature=False,
    ):
        a = self.encoder_node(a_prev=a_prev, x_c=x_c, noise_emb=noise_emb)
        s = self._shrink(a, noise_emb=noise_emb)
        a, upstream_grad = self.prior_node(s_p_1=s_p_1, a_prev=a, s_prev=s, top_signal=top_signal, noise_emb=noise_emb)
        s = self._shrink(a, noise_emb=noise_emb)
        decoded = self.decoder(s) if self.decoder is not None else None
        return a, s, upstream_grad, decoded, None


class neural_sheet6(nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis,
        kernel_size=7,
        stride=2,
        output_padding=1,
        learning_horizontal=True,
        eta_base=0.1,
        jfb_no_grad_iters=None,
        jfb_with_grad_iters=None,
        channel_mult_emb=2,
        channel_mult_noise=1,
        jfb_reuse_solution=0,
        mixer_value=0.0,
        noise_embedding=True,
        bias=True,
        relu_6=False,
        T=0.1,
        constraint_energy="SC",
        per_dim_threshold=True,
        positive_threshold=False,
        multiscale=False,
        intra=True,
        k_inter=None,
        n_hid_layers=1,
        tied_transpose=True,
        film=False,
        scalar_mul=False,
        ff_scale=False,
        control_groups=None,
        ignore_ff_control=False,
    ):
        super().__init__()

        self.eta_base = eta_base
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.mixer_value = mixer_value
        self.T = T
        self.positive_threshold = positive_threshold
        self.multiscale = multiscale
        self.num_scales = len(num_basis)
        self.constraint_energy = constraint_energy
        self.film = film
        self.ff_scale = ff_scale
        # None keeps legacy behavior (per-dim if enabled, else scalar).
        # Set to an int (e.g. 1, 4, 8) for grouped control modulation.
        self.control_groups = control_groups

        if multiscale:
            self.encoder_0 = GaussianPyramidEncoder(levels=self.num_scales - 1, concat_to_channels=False)
            self.decoder_0 = self.encoder_0.decode
            self.injection_ls = [in_channels] * self.num_scales
        else:
            if len(num_basis) == 1:
                self.encoder_0 = lambda x: [x]
                self.decoder_0 = lambda x: x[0]
                self.injection_ls = [in_channels]
            else:
                self.encoder_0 = lambda x: [x] + [None for _ in range(self.num_scales - 1)]
                self.decoder_0 = lambda x: x[0]
                self.injection_ls = [in_channels] + [None] * (self.num_scales - 1)

        levels = []
        prev_channels = None
        counter = 0
        for i, nb in enumerate(num_basis):
            in_channels = self.injection_ls[i]
            levels.append(
                neural_node6(
                    in_channels,
                    prev_channels,
                    nb,
                    kernel_size=kernel_size,
                    eta=eta_base,
                    stride=stride,
                    padding=kernel_size // 2 if i > 0 else 3,
                    output_padding=output_padding,
                    learning_horizontal=learning_horizontal,
                    bias=bias,
                    relu_6=relu_6,
                    constraint_energy=constraint_energy,
                    k_inter=k_inter,
                    tied_transpose=tied_transpose,
                    n_hid_layers=n_hid_layers,
                )
            )
            counter += 1

            if intra:
                prev_channels = nb
        self.levels = nn.ModuleList(levels)

        self.n_levels = len(levels)
        self.noise_embedding = noise_embedding
        
        if noise_embedding:
            emb_channels = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            out_dims = []
            for li, node in enumerate(self.levels):
                c_i = int(node.num_basis)
                if self.control_groups is None:
                    d_i = c_i if per_dim_threshold else 1
                else:
                    g_i = max(1, int(self.control_groups))
                    d_i = min(g_i, c_i)
                out_dims.append(d_i)

                # Register channel->group map for grouped control (d_i in (1, c_i) ).
                if 1 < d_i < c_i:
                    idx = torch.div(
                        torch.arange(c_i, dtype=torch.long) * d_i,
                        c_i,
                        rounding_mode="floor",
                    )
                    self.register_buffer(f"control_group_idx_{li}", idx, persistent=True)

            self.affines = nn.ModuleList(
                [nn.ModuleList([nn.Linear(emb_channels, d, bias=True) for d in out_dims]) for _ in range(4)]
            )
            for affine in self.affines:
                for lin in affine:
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)

    def _reset_all_wnorm(self):
        if not getattr(self, "_wnorm", False):
            return
        for lvl in self.levels:
            if hasattr(lvl, "reset_wnorm"):
                lvl.reset_wnorm()


    def film_modulation(self, noise_labels, i):

        emb = self.map_noise(noise_labels)

        emb = emb.reshape(len(noise_labels), 2, -1).flip(1).reshape(len(noise_labels), -1)
        emb = F.relu(self.map_layer0(emb))

        c_i = int(self.levels[i].num_basis)
        normalize_emb = []
        for j in range(4):
            ctrl_j = (1 + torch.tanh(self.affines[j][i](emb))) * 2  # [B, d_i]
            d_i = int(ctrl_j.shape[1])
            if d_i == 1 or d_i == c_i:
                pass
            else:
                idx = getattr(self, f"control_group_idx_{i}")
                ctrl_j = ctrl_j[:, idx]  # [B, C]
            normalize_emb.append(ctrl_j)

        final_emb = [ctrl.unsqueeze(2).unsqueeze(3).to(ctrl.dtype) for ctrl in normalize_emb]
        return final_emb

    
    def forward(
        self,
        x,
        a=None,
        s=None,
        upstream_grad=None,
        noise_labels=None,
        T=None,
        return_feature=False,
        infer_mode=False,
        n_iters=None,
        neuron_steer=None,
        return_history=False,
        class_labels=None,
        gamma_scale=None,
        ablate_noi=None,
        ablate_mode="spatial_mean",
        measurement = None, # (measurement, operator : image -> measurement)
    ):
        bsz = x.shape[0]
        if T is None:
            T = self.T
        self._reset_all_wnorm()

        if noise_labels is None:
            noise_labels = torch.zeros(bsz, device=x.device, dtype=x.dtype)

        if self.noise_embedding:
            noise_emb_ls = [self.film_modulation(noise_labels, i) for i in range(self.n_levels)]
        else:
            noise_emb_ls = [self.lambda_bias for _ in range(self.n_levels)]

        x_in = None

        if a is None:
            a = [None] * self.n_levels
        if s is None:
            s = [None] * self.n_levels
        if upstream_grad is None:
            upstream_grad = [None] * self.n_levels
        decoded = [None] * self.n_levels

        n0 = sample_uniformly_with_long_tail(
            self.jfb_no_grad_iters[0],
            self.jfb_no_grad_iters[1],
            mixer_value=self.mixer_value,
        )
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        if infer_mode:
            n0 = n_iters
        if n0 > 0:
            with torch.no_grad():
                history = []
                for _ in range(n0):
                    features = self.forward_inter(
                        x,
                        x_in,
                        a,
                        s,
                        upstream_grad,
                        decoded,
                        noise_emb_ls,
                        T=T,
                        return_history=return_history,
                        ablate_noi=ablate_noi,
                        ablate_mode=ablate_mode,
                        measurement=measurement,
                    )
                    if return_history:
                        history.extend(features)

        a = [ai.detach() if ai is not None else None for ai in a]
        # print(m1)
        for _ in range(m1):
            self.forward_inter(
                x,
                x_in,
                a,
                s,
                upstream_grad,
                decoded,
                noise_emb_ls,
                ablate_noi=ablate_noi,
                ablate_mode=ablate_mode,
                measurement=measurement,
            )

        decoded = [decoded[i] for i in range(self.n_levels)]

        if return_feature:
            # if return_history:
                # return history
            a = [ai.detach().clone() if ai is not None else None for ai in a]
            upstream_grad = [ui.detach().clone() if isinstance(ui, torch.Tensor) else 0 for ui in upstream_grad]
            return {
                "a": a,
                "decoded": decoded,
                "upstream_grad": upstream_grad,
                "denoised": self.decoder(decoded),
            }
        return self.decoder_0(decoded)

    def forward_dynamics(self,a,s,res_in,upstream_grad,decoded,noise_emb_ls,T,ablate_noi,ablate_mode, res_m=None, reverse = False):
        order = range(self.n_levels - 1, -1, -1) if reverse else range(self.n_levels)
        for i in order:
            inp = None if i == 0 else a[i - 1] # wrong because this should be s[i-1]
            top_signal = upstream_grad[i + 1] if i < self.n_levels - 1 else None
            a[i], s[i], upstream_grad[i], decoded[i], _ = self.levels[i](
                s_p_1 = inp,
                a_prev=a[i],
                x_c=(res_in[i], res_m[i]) if res_m is not None else res_in[i],
                top_signal=top_signal,
                noise_emb=noise_emb_ls[i],
                T=T,
            )
            a[i] = self._apply_noi_ablation(a[i], i, ablate_noi=ablate_noi, ablate_mode=ablate_mode)

    def forward_inter(
        self,
        x,
        x_in,
        a,
        s,
        upstream_grad,
        decoded,
        noise_emb_ls=None,
        T=0,
        return_history=False,
        ablate_noi=None,
        ablate_mode="spatial_mean",
        measurement=None,
    ):
        if self.constraint_energy == "SC" and decoded[0] is not None:
            res_in = self.encoder_0(x - self.decoder_0(decoded))
        else:
            res_in = self.encoder_0(x)

        if measurement is not None and decoded[0] is not None:
            measurement, operator = measurement
            x_hat = self.decoder_0(decoded)
            meas_res = measurement - operator.forward(x_hat)
            res_m = operator.transpose(meas_res)
            res_m = self.encoder_0(res_m)
        else:
            res_m = None
            
        # TODO, save two intermediate back below and return it as part of history, list of two state...
        self.forward_dynamics(a,s,res_in,upstream_grad,decoded,noise_emb_ls,T,ablate_noi,ablate_mode, res_m=res_m, reverse = False)
        self.forward_dynamics(a,s,res_in,upstream_grad,decoded,noise_emb_ls,T,ablate_noi,ablate_mode, res_m=res_m, reverse = True)

        return None

    def _apply_noi_ablation(self, a_i, level_idx, ablate_noi=None, ablate_mode="spatial_mean"):
        """
        Ablate selected channels in a level latent state.

        Args:
            a_i: Tensor [B, C, H, W]
            level_idx: int level index
            ablate_noi: dict[level_idx] -> indices to ablate
            ablate_mode: "spatial_mean" | "batch_spatial_mean" | "zero"
        """
        if a_i is None or ablate_noi is None or level_idx not in ablate_noi:
            return a_i

        idx = torch.as_tensor(ablate_noi[level_idx], device=a_i.device, dtype=torch.long)
        if idx.numel() == 0:
            return a_i
        idx = idx[(idx >= 0) & (idx < a_i.size(1))]
        if idx.numel() == 0:
            return a_i

        out = a_i.clone()
        if ablate_mode == "zero":
            out[:, idx] = 0
        elif ablate_mode == "batch_spatial_mean":
            fill = out[:, idx].mean(dim=(0, 2, 3), keepdim=True)
            out[:, idx] = fill.expand_as(out[:, idx])
        else:  # default: "spatial_mean"
            fill = out[:, idx].mean(dim=(2, 3), keepdim=True)
            out[:, idx] = fill.expand_as(out[:, idx])
        return out


    # if return_history:
    #     history = []
    #     for i in range(len(self.levels)):
    #         x_in = self.encoder(x)
    #         inp = None if i == 0 else a[i - 1]
    #         top_signal = upstream_grad[i + 1] if i < self.n_levels - 1 else None
    #         x_in_i = x_in[self.injection_dic[i]] if i in self.injection_dic else None
    #         x_c = res_in[self.injection_dic[i]] if i in self.injection_dic else None
    #         _, _, _, feature = self.levels[i](
    #             inp,
    #             a_prev=a[i],
    #             x_c=x_c,
    #             top_signal=top_signal,
    #             x_in=x_in_i,
    #             noise_emb=noise_emb_ls[i],
    #             T=T,
    #             return_feature=return_history,
    #         )
    #         history.append(feature)
    #     return [history]










# Borrowed from https://github.com/openai/guided-diffusion
from abc import abstractmethod


def t_conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.ConvTranspose1d(*args, **kwargs)
    if dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    if dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            out = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            out = F.interpolate(x, scale_factor=2, mode="nearest")
        if x.shape[-1] == x.shape[-2] == 3:
            out = F.pad(out, (1, 0, 1, 0))
        if self.use_conv:
            out = self.conv(out)
        return out


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.ReLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
            nn.SiLU(),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        return h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        time_emb_factor=4,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * time_emb_factor
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 7, padding=3, stride=2), nn.ReLU())]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.feature_resize = nn.ModuleList([])
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                self.feature_resize.append(conv_nd(dims, ich, ch, 1))
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        # Keep encoder-skip projection order aligned with hs.pop() order.
        # Reversing here causes channel mismatches in decoder skip fusion.

        self.out = nn.Sequential(
            nn.ReLU(),
            zero_module(t_conv_nd(dims, input_ch, out_channels, 7, padding=3, stride=2, output_padding=1)),
        )

    def forward(self, x, timesteps, y=None, class_labels=None, **kwargs):
        # Keep compatibility with existing loss calls that pass class_labels.
        if y is None and class_labels is not None:
            y = class_labels

        if self.num_classes is None:
            y = None
        else:
            assert y is not None and y.shape == (x.shape[0],), (
                "must specify y with shape [N] for class-conditional UNet"
            )

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        for module, resizer in zip(self.output_blocks, self.feature_resize):
            h = resizer(hs.pop()) + h
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


def UNetBig(image_size, in_channels=3, out_channels=3, base_width=192, num_classes=None, use_attention=True):
    if image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size in (32,):
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    if use_attention:
        attention_resolutions = "28,14,7" if image_size == 28 else "32,16,8"
        attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]
    else:
        attention_ds = []

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=base_width,
        out_channels=out_channels,
        num_res_blocks=3,
        attention_resolutions=tuple(attention_ds),
        dropout=0.1,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )


def UNet(image_size, in_channels=3, out_channels=3, base_width=64, num_classes=None, use_attention=True):
    if image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size in (32,):
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    if use_attention:
        attention_resolutions = "28,14,7" if image_size == 28 else "32,16,8"
        attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]
    else:
        attention_ds = []

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=base_width,
        out_channels=out_channels,
        num_res_blocks=3,
        attention_resolutions=tuple(attention_ds),
        dropout=0.1,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=True,
        use_new_attention_order=True,
    )


def UNetSmall(image_size, in_channels=3, out_channels=3, base_width=32, num_classes=None, use_attention=True):
    if image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    if use_attention:
        attention_resolutions = "28,14,7" if image_size == 28 else "32,16,8"
        attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]
    else:
        attention_ds = []

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=base_width,
        out_channels=out_channels,
        num_res_blocks=1,
        attention_resolutions=tuple(attention_ds),
        time_emb_factor=2,
        dropout=0.1,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=32,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=True,
        use_new_attention_order=True,
    )


class ToySCInference(nn.Module):
    """Single-step sparse-coding inference on flat vectors."""

    def __init__(self, data_dim: int, hidden_dim: int, eta: float = 0.1, lam: float = 0.0, learning_horizontal=True):
        super().__init__()
        self.eta = eta
        # self.lam = lam
        self.lam = torch.nn.Parameter(torch.tensor(lam))
        # define lam as a tensor of shape (1,hidden_dim,1,1)
        # self.lam = torch.nn.Parameter(torch.ones(1, hidden_dim) * lam)
        self.Phi = nn.Linear(hidden_dim, data_dim, bias=False)
        self.Phi_T = _LinearTiedTranspose(self.Phi)
        self.M = nn.Linear(hidden_dim, hidden_dim, bias=False) if learning_horizontal else None

    def forward(self, x: torch.Tensor, a_prev: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        prior_term = self.M(a_prev) if (a_prev is not None and self.M is not None) else 0.0
        constraint_ff = self.Phi_T(x - self.Phi(a_prev)) if a_prev is not None else self.Phi_T(x)
        if a_prev is None:
            a = self.eta * (constraint_ff + noise_emb * self.lam)
        else:
            a = a_prev + self.eta * (constraint_ff + noise_emb * (self.lam + prior_term))
        return F.relu(a)


class _LinearTiedTranspose(nn.Module):
    """Linear layer tied to transpose of another linear layer."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight.t(), None)


class ToySCDEQ(nn.Module):
    """DEQ-like sparse coding denoiser for toy vector data."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        eta: float = 0.1,
        lam: float = 0.0,
        noise_channels: int = None,
        emb_channels: int = None,
        jfb_no_grad_iters: tuple = (5, 10),
        jfb_with_grad_iters: tuple = (1, 3),
        mixer_value: float = 0.0,
        learning_horizontal: bool = False,
    ):
        super().__init__()
        if noise_channels is None:
            noise_channels = max(2, (hidden_dim // 2) * 2)
        if emb_channels is None:
            emb_channels = hidden_dim
        self.sc_net = ToySCInference(
            data_dim,
            hidden_dim,
            eta=eta,
            lam=lam,
            learning_horizontal=learning_horizontal,
        )
        self.jfb_no_grad_iters = tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = tuple(jfb_with_grad_iters)
        self.mixer_value = mixer_value

        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
        # Scalar FiLM-like gate, matching notebook behavior.
        self.affines = nn.Linear(emb_channels, 1, bias=True)

    def modulation(self, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(len(noise_labels), 2, -1).flip(1).reshape(len(noise_labels), -1)
        emb = F.relu(self.map_layer0(emb))
        return (1.0 + torch.tanh(self.affines(emb))) * 2.0

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor = None, return_feature=False, **kwargs):
        # 1. Store the original shape for the final reconstruction
        original_shape = x.shape
        bsz = original_shape[0]
        device = x.device

        # 2. Flatten: [B, C, H, W] -> [B, C*H*W]
        # start_dim=1 ensures we don't flatten the Batch dimension (dim 0)
        if x.ndim > 2:
            x = x.flatten(start_dim=1)

        # ... (Processing logic remains the same) ...
        if noise_labels is None:
            noise_labels = torch.zeros(bsz, device=device, dtype=x.dtype)
        else:
            noise_labels = noise_labels.to(device=device, dtype=x.dtype)

        noise_emb = self.modulation(noise_labels)
        
        # Sparse coding iterations
        a = None
        n0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1])
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])
        
        with torch.no_grad():
            for _ in range(n0):
                a = self.sc_net(x, a, noise_emb)
        for _ in range(m1):
            a = self.sc_net(x, a, noise_emb)

        # Decode back to data space (currently [B, flat_dim])
        decoded = self.sc_net.Phi(a)

        # 3. Restore the original shape
        # We still use .view() here because it is the most efficient way 
        # to expand a flat tensor back into its multi-dimensional structure.
        if len(original_shape) > 2:
            decoded = decoded.view(original_shape)

        if return_feature:
            return decoded, a
        return decoded