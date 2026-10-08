
import math

import torch
import torch.nn as nn


class FSQQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ).

    Input:   [B, M, input_dim]
    Codes:   [B, M, code_dim]
    Indices: [B, M] (discrete token IDs)
    """

    def __init__(
        self,
        input_dim: int = 128,
        levels=(4, 4, 4, 4, 4, 4),
        eps: float = 1e-3,
    ):
        super().__init__()

        if any(level < 2 for level in levels):
            raise ValueError("Each level must be >= 2.")

        self.eps = eps
        self.code_dim = len(levels)
        self.codebook_size = math.prod(levels)

        self.to_code = nn.Linear(input_dim, self.code_dim)

        self.register_buffer(
            "levels",
            torch.tensor(levels, dtype=torch.long),
        )

        self.register_buffer(
            "basis",
            torch.tensor(
                [math.prod(levels[:i]) for i in range(len(levels))],
                dtype=torch.long,
            ),
        )

    def forward(self, x):
        # Project continuous embeddings into FSQ dimensions.
        z = self.to_code(x)

        # Bound values before quantization.
        levels = self.levels.to(z.dtype)
        half_l = (levels - 1) * (1 - self.eps) / 2

        offset = (self.levels % 2 == 0).to(z.dtype) * 0.5
        shift = torch.tan(offset / half_l)

        bounded = torch.tanh(z + shift) * half_l - offset

        # Straight-through rounding for gradient propagation.
        rounded = bounded + (bounded.round() - bounded).detach()

        half_width = (self.levels // 2).to(z.dtype)
        codes = rounded / half_width

        indices = self.codes_to_indices(codes.detach())

        return codes, indices

    def codes_to_indices(self, codes):
        half_width = (self.levels // 2).to(codes.dtype)

        digits = torch.round(
            codes * half_width + half_width
        ).long()

        return (digits * self.basis).sum(dim=-1)

    def indices_to_codes(self, indices):
        digits = (
            torch.div(
                indices.unsqueeze(-1),
                self.basis,
                rounding_mode="floor",
            )
            % self.levels
        )

        half_width = self.levels // 2

        return (
            (digits - half_width).float()
            / half_width.float()
        )

