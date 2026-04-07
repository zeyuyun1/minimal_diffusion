import torch
import torch.nn.functional as F


class PyramidEDMLoss:
    """
    EDM loss computed in pyramid space.

    Instead of supervising only the final reconstructed image, this loss
    supervises each level's decoded output against the corresponding Laplacian
    pyramid band of the clean image.  This gives every level a direct learning
    signal, preventing upper (coarser/semantic) levels from being starved of
    gradients.

    Architecture assumption
    -----------------------
    The model is a `neural_sheet3` with `multiscale=True`.
      - `net.encoder(x)` returns a list of Laplacian pyramid bands
        `[band_0, band_1, ..., band_L]`  (full-res detail → coarsest base)
      - Calling `net(..., return_feature=True)` returns a dict with key
        `"decoded"` = `[dec_0, dec_1, ..., dec_L]`  matching pyramid bands
        spatially.
      - `net.decoder(decoded)` reconstructs the full image.

    Loss
    ----
    total = w_main * MSE(final_recon, clean)
          + sum_i  w_level[i] * MSE(decoded[i], pyr_target[i])

    Both terms use EDM-style weighting if `edm_weighting=True`.

    level_weights
    -------------
    List/tuple of per-level loss weights.  len must equal n_levels.
    Default `None` uses equal weight 1.0 for every level.
    A sensible coarser-upweighted default is e.g. [0.5, 1.0, 2.0] for
    three levels so semantic (coarse) levels see stronger gradient.
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.25,
        noise_range=(None, None),
        edm_weighting: bool = False,
        level_weights=None,
        main_weight: float = 1.0,
    ):
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.noise_min, self.noise_max = noise_range
        self.edm_weighting = edm_weighting
        self.level_weights = level_weights   # None → equal 1.0 per level
        self.main_weight = float(main_weight)

    def _edm_weight(self, sigma):
        """Return broadcastable loss-weight tensor from sigma [B,1,1,1]."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def __call__(self, net, images: torch.Tensor, class_labels=None):
        device = images.device
        B = images.shape[0]

        # ---- sample sigma ----
        rnd = torch.randn(B, 1, 1, 1, device=device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        if self.noise_min is not None:
            sigma = sigma.clamp_min(self.noise_min)
        if self.noise_max is not None:
            sigma = sigma.clamp_max(self.noise_max)

        y = images
        n = torch.randn_like(y) * sigma
        y_noisy = y + n
        noise_labels = sigma.flatten()

        # ---- pyramid target from clean image ----
        # net.encoder is GaussianPyramidEncoder (multiscale=True)
        with torch.no_grad():
            pyr_target = net.encoder(y)   # list of bands, each [B, C, H_i, W_i]

        # ---- forward with per-level decoded outputs ----
        kw = dict(noise_labels=noise_labels, return_feature=True)
        if class_labels is not None:
            kw["class_labels"] = class_labels
        features = net(y_noisy, **kw)

        denoised = features["denoised"]   # final reconstructed image [B,3,H,W]
        decoded  = features["decoded"]    # list of level outputs

        n_levels = len(decoded)

        # ---- per-level weights ----
        if self.level_weights is None:
            lw = [1.0] * n_levels
        else:
            lw = list(self.level_weights)
            if len(lw) != n_levels:
                # broadcast scalar or truncate/pad with last value
                lw = (lw + [lw[-1]] * n_levels)[:n_levels]

        # ---- EDM weight ----
        edm_w = self._edm_weight(sigma) if self.edm_weighting else 1.0

        # ---- main loss on final reconstruction ----
        per_pix = self.main_weight * edm_w * (denoised - y) ** 2

        # ---- per-level auxiliary losses ----
        for i, (dec_i, tgt_i, w_i) in enumerate(zip(decoded, pyr_target, lw)):
            if dec_i is None or w_i == 0.0:
                continue
            # Spatial size may differ by ±1 due to stride/padding; align.
            if dec_i.shape[-2:] != tgt_i.shape[-2:]:
                tgt_i = F.interpolate(
                    tgt_i, size=dec_i.shape[-2:], mode="bilinear", align_corners=False
                )
            # edm_w is [B,1,1,1]; need to match spatial size of dec_i
            edm_w_i = (
                F.interpolate(edm_w, size=dec_i.shape[-2:], mode="nearest")
                if self.edm_weighting
                else 1.0
            )
            level_se = w_i * edm_w_i * (dec_i - tgt_i) ** 2
            # up-sample to full-res so per_pix stays [B,C,H,W] shaped
            if level_se.shape[-2:] != per_pix.shape[-2:]:
                level_se = F.interpolate(
                    level_se, size=per_pix.shape[-2:], mode="bilinear", align_corners=False
                )
            per_pix = per_pix + level_se

        return per_pix


class SimpleNoiseLoss:
    """
    Blind denoising loss with configurable σ sampling and σ-aware weighting.

    - sigma_dist: 'uniform' (default) or 'lognormal' (Song-style)
    - weighting:
        'none'     -> unweighted MSE
        'inv_var'  -> 1 / (σ^2 + eps)         (strongly downweights large σ)
        'edm'      -> (σ^2 + σ_data^2) / (σ^2 σ_data^2 + eps)
        'power'    -> (σ / σ_data)^(-weight_power)

    Returns per-pixel squared error (optionally weighted). Keep reduction outside
    to match your current training loop.
    """
    def __init__(self,
                 noise_range=(0.01, 1.0),
                 sigma_dist='lognormal',
                 # Lognormal params (Song): σ = exp(P_mean + P_std * N(0,1))
                 P_mean=-1.2,
                 P_std=1.2,
                 clamp_lognormal_to_range=True,
                 # Weighting options
                 weighting='edm',
                 sigma_data=0.25,
                 weight_power=2.0,
                 eps=1e-8):
        self.noise_min, self.noise_max = noise_range
        if self.noise_min is None or self.noise_max is None:
            raise ValueError("SimpleNoiseLoss requires finite (sigma_min, sigma_max).")
        if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

        self.sigma_dist = sigma_dist.lower()
        if self.sigma_dist not in ('uniform', 'lognormal'):
            raise ValueError("sigma_dist must be 'uniform' or 'lognormal'.")

        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.clamp_lognormal_to_range = bool(clamp_lognormal_to_range)

        self.weighting = weighting.lower()
        if self.weighting not in ('none', 'inv_var', 'edm', 'power'):
            raise ValueError("weighting must be 'none', 'inv_var', 'edm', or 'power'.")

        self.sigma_data = float(sigma_data)
        self.weight_power = float(weight_power)
        self.eps = float(eps)

    def _sample_sigma(self, B, device):
        if self.sigma_dist == 'uniform':
            if self.noise_max > self.noise_min:
                u = torch.rand(B, 1, 1, 1, device=device)
                sigma = self.noise_min + u * (self.noise_max - self.noise_min)
            else:
                sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)
        else:
            # Song-style lognormal sampling
            rnd = torch.randn(B, 1, 1, 1, device=device)
            sigma = (rnd * self.P_std + self.P_mean).exp()
            if self.clamp_lognormal_to_range:
                sigma = sigma.clamp(min=self.noise_min, max=self.noise_max)
        return sigma

    def _loss_weight(self, sigma):
        """Return a broadcastable loss weight tensor shaped (B,1,1,1)."""
        if self.weighting == 'none':
            return 1.0
        elif self.weighting == 'inv_var':
            # Classic inverse-variance weighting to counter σ^2 growth of MSE.
            return 1.0 / (sigma**2 + self.eps)
        elif self.weighting == 'edm':
            # Karras EDM-style: behaves like 1/σ^2 at large σ but is finite at tiny σ.
            sd2 = self.sigma_data ** 2
            return (sigma**2 + sd2) / (sigma**2 * sd2 + self.eps)
        elif self.weighting == 'power':
            # Generalized power law around a reference σ_data.
            # p=2 recovers ~1/σ^2 (up to a constant).
            # Note: add eps in denominator via clamp to avoid blowup at tiny σ.
            return (torch.clamp(sigma, min=self.eps) / self.sigma_data).pow(-self.weight_power)
        else:
            raise RuntimeError("Unreachable")

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        sigma = self._sample_sigma(B, device)          # (B,1,1,1)
        noise = torch.randn_like(images) * sigma       # additive Gaussian noise
        y_noisy = images + noise

        D_yn = net(y_noisy)                            # no conditioning

        per_pixel_se = (D_yn - images) ** 2            # (B,C,H,W)
        w = self._loss_weight(sigma)                   # (B,1,1,1), broadcasts
        return per_pixel_se * w


class SimpleUniformNoiseLoss:
    """
    Uniform-in-σ denoising loss with *no conditioning*.
    Calls D_yn = net(y + n) only. Returns per-pixel squared error.
    """
    def __init__(self, noise_range=(0.01, 1.0)):
        self.noise_min, self.noise_max = noise_range
        if self.noise_min is None or self.noise_max is None:
            raise ValueError("SimpleUniformNoiseLoss requires finite (sigma_min, sigma_max).")
        if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        # σ ~ Uniform[min, max] (fixed if min == max)
        if self.noise_max > self.noise_min:
            u = torch.rand(B, 1, 1, 1, device=device)
            sigma = self.noise_min + u * (self.noise_max - self.noise_min)
        else:
            sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)

        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n)                # <-- no sigma/labels/aug passed

        return (D_yn - y) ** 2


class EDMLossNoCond:
    """
    EDM loss with log-normal σ sampling + EDM weighting.
    Calls net(y+n, sigma) where 'net' is the EDMPrecond wrapper.
    """
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2,
                 sigma_data: float = 0.25, noise_range=(None, None), edm_weighting=False):
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.noise_min, self.noise_max = noise_range
        self.edm_weighting = edm_weighting

    def __call__(self, net, images: torch.Tensor,a=None,upstream_grad=None,return_feature=True,class_labels=None):
        device = images.device
        B = images.shape[0]
        sigma_shape = [B] + [1] * (images.ndim - 1)

        # σ ~ LogNormal(P_mean, P_std)
        rnd = torch.randn(*sigma_shape, device=device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        if self.noise_min is not None:
            sigma = sigma.clamp_min(self.noise_min)
        if self.noise_max is not None:
            sigma = sigma.clamp_max(self.noise_max)

        y = images
        n = torch.randn_like(y) * sigma
        # if return_feature:
        #     feature = net(y + n, noise_labels=sigma.flatten(),a=a,upstream_grad=upstream_grad,return_feature=True)  # EDM-preconditioned forward
        #     D_yn = feature["denoised"]
        #     a = feature["a"]
        #     upstream_grad = feature["upstream_grad"]
        # else:
            # a=upstream_grad=None
        if class_labels is not None:
            D_yn = net(y + n, noise_labels=sigma.flatten(),class_labels=class_labels)  # EDM-preconditioned forward
        else:
            D_yn = net(y + n, noise_labels=sigma.flatten())  # EDM-preconditioned forward

        if self.edm_weighting:
        # EDM weighting
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            return weight * ((D_yn - y) ** 2)
        else:
            return (D_yn - y) ** 2


class DSMLossNoCond:
    """
    Denoising Score Matching (DSM) loss with log-normal σ sampling.
    Assumes `net(y_tilde, noise_labels=sigma.flatten())` returns
    the score estimate ∂/∂y_tilde log p_σ(y_tilde) with the same shape as y_tilde.
    """
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2,
                 sigma_data: float = 0.25, noise_range=(None, None),
                 edm_weighting: bool = False):
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.noise_min, self.noise_max = noise_range
        self.edm_weighting = edm_weighting

    def __call__(self, net, images: torch.Tensor):
        device = images.device
        B = images.shape[0]

        # Sample σ ~ LogNormal(P_mean, P_std)
        rnd = torch.randn(B, 1, 1, 1, device=device)
        sigma = (rnd * self.P_std + self.P_mean).exp()  # [B,1,1,1]
        if self.noise_min is not None:
            sigma = sigma.clamp_min(self.noise_min)
        if self.noise_max is not None:
            sigma = sigma.clamp_max(self.noise_max)

        y = images  # clean
        n = torch.randn_like(y) * sigma  # Gaussian noise with std σ
        y_tilde = y + n                  # noisy input

        # Model score estimate s_θ(y_tilde, σ)
        # net should return a tensor with same shape as y_tilde
        scores = net(y_tilde, noise_labels=sigma.flatten())

        # DSM target: - (1/σ^2) * (y_tilde - y) = - n / σ^2 = -ε/σ
        target = - n / (sigma ** 2)      # broadcasts over C,H,W

        # Per-pixel squared error between score and target
        loss = (scores - target) ** 2

        if self.edm_weighting:
            # Optional EDM-style weighting (same as your EDMLossNoCond)
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            loss = weight * loss  # broadcasts over channels/spatial

        # Return per-pixel loss; caller can do .mean() or .sum()
        return loss



# class SimpleUniformNoiseLoss:
#     """
#     Uniform-in-σ denoising loss with *no conditioning*.
#     Calls D_yn = net(y + n) only. Returns per-pixel squared error.
#     """
#     def __init__(self, noise_range=(0.01, 1.0)):
#         self.noise_min, self.noise_max = noise_range
#         if self.noise_min is None or self.noise_max is None:
#             raise ValueError("SimpleUniformNoiseLoss requires finite (sigma_min, sigma_max).")
#         if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
#             raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

#     def __call__(self, net, images):
#         device = images.device
#         B = images.shape[0]

#         # σ ~ Uniform[min, max] (fixed if min == max)
#         if self.noise_max > self.noise_min:
#             u = torch.rand(B, 1, 1, 1, device=device)
#             sigma = self.noise_min + u * (self.noise_max - self.noise_min)
#         else:
#             sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)

#         y = images
#         n = torch.randn_like(y) * sigma
#         D_yn = net(y + n)                # <-- no sigma/labels/aug passed

#         return (D_yn - y) ** 2


# class EDMStyleXPredLoss:
#     """
#     EDM-style x-prediction loss for a net that takes ONLY an image tensor.
#     - Samples σ ~ Uniform[min,max]
#     - Normalizes input with c_in = 1/sqrt(σ^2 + σ_data^2) before calling net
#     - Unscales the net output back to image domain
#     - Applies EDM weighting: λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
#     Returns a per-pixel squared error map (same shape as images).
#     """
#     def __init__(self, noise_range=(0.01, 1.0), sigma_data=0.5, eps=1e-8):
#         sigma_min, sigma_max = noise_range
#         if sigma_min is None or sigma_max is None:
#             raise ValueError("Provide finite (sigma_min, sigma_max).")
#         if not (sigma_min > 0 and sigma_max >= sigma_min):
#             raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")
#         self.sigma_min   = float(sigma_min)
#         self.sigma_max   = float(sigma_max)
#         self.sigma_data  = float(sigma_data)
#         self.eps         = float(eps)

#     def __call__(self, net, images):
#         device = images.device
#         B = images.shape[0]

#         # --- sample σ ~ Uniform[min,max] ---
#         if self.sigma_max > self.sigma_min:
#             u = torch.rand(B, 1, 1, 1, device=device, dtype=images.dtype)
#             sigma = self.sigma_min + u * (self.sigma_max - self.sigma_min)
#         else:
#             sigma = torch.full((B, 1, 1, 1), self.sigma_min, device=device, dtype=images.dtype)

#         # --- make noisy input y + n ---
#         y = images
#         n = torch.randn_like(y) * sigma
#         y_noisy = y + n

#         # --- EDM input normalization ---
#         # c_in = 1 / sqrt(σ^2 + σ_data^2)
#         c_in = 1.0 / torch.sqrt(sigma * sigma + self.sigma_data * self.sigma_data)

#         # net sees only normalized inputs
#         y_in = c_in * y_noisy
#         y_hat_scaled = net(y_in)

#         # --- unscale prediction back to image domain ---
#         # x_hat = y_hat_scaled / c_in
#         x_hat = y_hat_scaled / (c_in + self.eps)

#         # --- EDM x-pred weighting ---
#         # λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
#         weight = (sigma * sigma + self.sigma_data * self.sigma_data) / (
#             (sigma * self.sigma_data + self.eps) ** 2
#         )

#         # per-pixel loss map
#         loss_map = weight * (x_hat - y) ** 2
#         return loss_map

class SimpleUniformNoiseLoss:
    """
    Uniform-in-σ denoising loss with *no conditioning*.
    Calls D_yn = net(y + n) only. Returns per-pixel squared error.
    """
    def __init__(self, noise_range=(0.01, 1.0)):
        self.noise_min, self.noise_max = noise_range
        if self.noise_min is None or self.noise_max is None:
            raise ValueError("SimpleUniformNoiseLoss requires finite (sigma_min, sigma_max).")
        if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        # σ ~ Uniform[min, max] (fixed if min == max)
        if self.noise_max > self.noise_min:
            u = torch.rand(B, 1, 1, 1, device=device)
            sigma = self.noise_min + u * (self.noise_max - self.noise_min)
        else:
            sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)

        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n)                # <-- no sigma/labels/aug passed

        return (D_yn - y) ** 2


class EDMStyleXPredLoss:
    """
    EDM-style x-prediction loss for a net that takes ONLY an image tensor.
    - Samples σ ~ Uniform[min,max]
    - Normalizes input with c_in = 1/sqrt(σ^2 + σ_data^2) before calling net
    - Unscales the net output back to image domain
    - Applies EDM weighting: λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
    Returns a per-pixel squared error map (same shape as images).
    """
    def __init__(self, noise_range=(0.01, 1.0), sigma_data=0.5, eps=1e-8):
        sigma_min, sigma_max = noise_range
        if sigma_min is None or sigma_max is None:
            raise ValueError("Provide finite (sigma_min, sigma_max).")
        if not (sigma_min > 0 and sigma_max >= sigma_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")
        self.sigma_min   = float(sigma_min)
        self.sigma_max   = float(sigma_max)
        self.sigma_data  = float(sigma_data)
        self.eps         = float(eps)

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        # --- sample σ ~ Uniform[min,max] ---
        if self.sigma_max > self.sigma_min:
            u = torch.rand(B, 1, 1, 1, device=device, dtype=images.dtype)
            sigma = self.sigma_min + u * (self.sigma_max - self.sigma_min)
        else:
            sigma = torch.full((B, 1, 1, 1), self.sigma_min, device=device, dtype=images.dtype)

        # --- make noisy input y + n ---
        y = images
        n = torch.randn_like(y) * sigma
        y_noisy = y + n

        x_hat = net(y_noisy)

        # --- EDM x-pred weighting ---
        # λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
        weight = (sigma * sigma + self.sigma_data * self.sigma_data) / (
            (sigma * self.sigma_data + self.eps) ** 2
        )

        # per-pixel loss map
        loss_map = weight * (x_hat - y) ** 2
        return loss_map
