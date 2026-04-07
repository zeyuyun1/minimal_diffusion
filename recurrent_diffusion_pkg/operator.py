
from abc import ABC, abstractmethod
import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn
import torch
from functools import partial
# from motionblur.motionblur import Kernel
# from .fastmri_utils import fft2c_new, ifft2c_new

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, in_channels=3, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.in_channels = int(in_channels)
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                self.kernel_size,
                stride=1,
                padding=0,
                bias=False,
                groups=self.in_channels,
            )
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
                
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

# @register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    """
    Gaussian blur operator parameterized by normalized strength in [0, 1].
    strength=0 -> light blur, strength=1 -> heavy blur.
    """
    def __init__(self, kernel_size, strength, device, in_channels=3, std_range=(0.2, 4.0)):
        self.device = device
        self.kernel_size = kernel_size
        self.strength = float(strength)
        self.in_channels = int(in_channels)
        std_min, std_max = std_range
        self.std = float(std_min + self.strength * (std_max - std_min))
        self.conv = Blurkernel(
            blur_type='gaussian',
            kernel_size=kernel_size,
            std=self.std,
            in_channels=self.in_channels,
            device=device
        ).to(device)
        kernel = self.conv.get_kernel()
        self.conv.update_weights(kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        # For symmetric Gaussian kernel with same boundary handling, A^T ≈ A
        return self.conv(data)

# @register_operator(name='random_inpainting')
class RandomInpaintingOperator(LinearOperator):
    """
    A(x) = m ⊙ x, where m in {0,1} is a binary mask (1=observed, 0=missing).
    Drop rate = fraction of pixels missing.
    """
    def __init__(self, in_shape, strength, device, per_channel=False, fixed_mask=True, seed=None, drop_range=(0.1, 0.9)):
        """
        in_shape: (C,H,W) or (B,C,H,W) accepted; mask stored as (1,C,H,W)
        strength: normalized corruption strength in [0,1]
                  mapped to drop_rate in drop_range.
        per_channel: if False, same mask for all channels (recommended)
        fixed_mask: if True, sample mask once and reuse across calls
        seed: optional int
        """
        self.device = device
        self.strength = float(strength)
        dmin, dmax = drop_range
        self.drop_rate = float(dmin + self.strength * (dmax - dmin))
        self.drop_rate = min(max(self.drop_rate, 0.0), 0.999999)
        self.per_channel = per_channel
        self.fixed_mask = fixed_mask
        self.seed = seed

        if len(in_shape) == 4:
            _, C, H, W = in_shape
        else:
            C, H, W = in_shape

        self.C, self.H, self.W = C, H, W
        self._mask = None
        self._mask_B = None
        if fixed_mask:
            self._mask = self._sample_mask()

    def _sample_mask(self, batch_size=1):
        if self.seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
        else:
            g = None

        if self.per_channel:
            # independent mask per channel
            m = torch.rand((batch_size, self.C, self.H, self.W), device=self.device, generator=g)
        else:
            # same spatial mask for all channels
            m = torch.rand((batch_size, 1, self.H, self.W), device=self.device, generator=g)
            m = m.expand(batch_size, self.C, self.H, self.W)

        # m=1 observed, m=0 missing
        observed_prob = 1.0 - self.drop_rate
        m = (m < observed_prob).to(torch.float32)
        return m

    def get_mask(self, batch_size=None):
        B = 1 if batch_size is None else int(batch_size)
        if self.fixed_mask:
            if self._mask is None or self._mask_B != B:
                self._mask = self._sample_mask(batch_size=B)
                self._mask_B = B
            return self._mask
        return self._sample_mask(batch_size=B)

    def forward(self, data, **kwargs):
        # data: (B,C,H,W)
        m = kwargs.get("mask", None)
        if m is None:
            m = self.get_mask()
        if m.shape[0] == 1 and data.shape[0] != 1:
            m = m.expand(data.shape[0], -1, -1, -1)
        return data * m

    def transpose(self, data, **kwargs):
        # For masking operator, A^T = A
        return self.forward(data, **kwargs)

    def project(self, data, measurement, **kwargs):
        """
        Enforce known pixels equal measurement:
          x <- (1-m)⊙x + m⊙y
        """
        m = kwargs.get("mask", None)
        if m is None:
            m = self.get_mask()
        if m.shape[0] == 1 and data.shape[0] != 1:
            m = m.expand(data.shape[0], -1, -1, -1)
        return data * (1.0 - m) + measurement * m


class RandomBoxInpaintingOperator(LinearOperator):
    """
    Batch-wise random rectangle inpainting with FIXED masks per batch.
    m=1 known, m=0 hole. Different box per sample, but constant across calls.
    """
    def __init__(self, in_shape, strength, device="cuda", per_channel=False, seed=None, fixed_mask=True):
        self.device = device
        self.strength = float(strength)
        self.per_channel = per_channel
        self.seed = seed
        self.fixed_mask = fixed_mask

        if len(in_shape) == 4:
            _, C, H, W = in_shape
        else:
            C, H, W = in_shape
        self.C, self.H, self.W = C, H, W

        # strength -> hole fraction
        min_frac, max_frac = 0.1, 0.7
        self.box_frac = min_frac + self.strength * (max_frac - min_frac)

        # cache
        self._mask = None
        self._mask_B = None

    def _sample_mask(self, B):
        if self.seed is not None:
            g = torch.Generator(device=self.device).manual_seed(self.seed)
        else:
            g = None

        C, H, W = self.C, self.H, self.W
        bh = max(1, int(H * self.box_frac))
        bw = max(1, int(W * self.box_frac))

        y0 = torch.randint(0, max(1, H - bh + 1), (B,), device=self.device, generator=g)
        x0 = torch.randint(0, max(1, W - bw + 1), (B,), device=self.device, generator=g)

        m = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)
        for i in range(B):
            yi0 = int(y0[i].item())
            xi0 = int(x0[i].item())
            m[i, :, yi0:yi0 + bh, xi0:xi0 + bw] = 0.0

        m = m.expand(B, C, H, W).contiguous()
        return m

    def get_mask(self, batch_size):
        if not self.fixed_mask:
            return self._sample_mask(batch_size)

        # fixed mask: cache per batch size
        if self._mask is None or self._mask_B != batch_size:
            self._mask = self._sample_mask(batch_size)
            self._mask_B = batch_size
        return self._mask

    def forward(self, data, **kwargs):
        m = kwargs.get("mask", None)
        if m is None:
            m = self.get_mask(data.shape[0])
        return data * m

    def transpose(self, data, **kwargs):
        return self.forward(data, **kwargs)

    def project(self, data, measurement, **kwargs):
        m = kwargs.get("mask", None)
        if m is None:
            m = self.get_mask(data.shape[0])
        return data * (1.0 - m) + measurement * m
        
class ColorizationOperator(LinearOperator):
    """
    Measurement is grayscale (3ch replicated).
    strength controls interpolation between identity and grayscale.
    """
    def __init__(self, strength, device="cuda"):
        self.device = device
        self.strength = float(strength)
        # RGB->luma coefficients
        self.w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1,3,1,1)

    def _to_gray3(self, x):
        y = (x * self.w).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return y.repeat(1,3,1,1)                   # (B,3,H,W)

    def forward(self, data, **kwargs):
        # returns the measurement y
        gray = self._to_gray3(data)
        if self.strength <= 0:
            return data
        if self.strength >= 1:
            return gray
        # partial color removal as "corruption"
        return (1 - self.strength) * data + self.strength * gray

    def transpose(self, data, **kwargs):
        # Adjoint for this mixing is itself (symmetric linear map)
        # It's not a strict projector, but works for backprojection injection.
        return data

    def project(self, data, measurement, **kwargs):
        # No hard projection is well-defined here; return data unchanged.
        return data