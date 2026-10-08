
import math

import torch
import torch.nn.functional as F


class GaussianGranularityTargets:
    """
    Build coarse/mid/fine reconstruction targets.

    Input: [B, C, T]
    Output: dict[str, Tensor[B, C, T]]

    Smoothing is expressed in seconds so that
    different sampling rates remain comparable.
    """

    def __init__(
        self,
        sample_rate: int,
        sigmas_seconds=None,
    ):
        self.sample_rate = sample_rate

        self.sigmas_seconds = sigmas_seconds or {
            "coarse": 0.25,
            "mid": 0.05,
            "fine": 0.0,
        }

    def smooth(self, x, sigma_seconds):
        if sigma_seconds == 0:
            return x

        sigma = sigma_seconds * self.sample_rate
        radius = max(1, math.ceil(3 * sigma))

        positions = torch.arange(
            -radius,
            radius + 1,
            device=x.device,
            dtype=x.dtype,
        )

        kernel = torch.exp(
            -0.5 * (positions / sigma) ** 2
        )

        kernel = kernel / kernel.sum()

        channels = x.shape[1]

        weights = kernel.view(1, 1, -1)
        weights = weights.repeat(channels, 1, 1)

        padded = F.pad(
            x,
            (radius, radius),
            mode="reflect",
        )

        return F.conv1d(
            padded,
            weights,
            groups=channels,
        )

    def __call__(self, x):
        if x.ndim != 3:
            raise ValueError("Expected [B, C, T].")

        return {
            level: self.smooth(x, sigma)
            for level, sigma in self.sigmas_seconds.items()
        }

