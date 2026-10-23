"""Sequential temporal refinement controller for TimeTok V3."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TemporalRefinementAgent(nn.Module):
    """
    Observes coarse tokens and previously revealed mid/fine tokens.

    levels[B, 16]:
        0 = coarse
        1 = mid
        2 = fine

    Predicts gain per added token for:
        output[..., 0] = coarse -> mid
        output[..., 1] = mid -> fine
    """

    def __init__(self, dim=128):
        super().__init__()

        # coarse: 6 dimensions
        # mid: 3 x 6 = 18
        # fine: 4 x 6 = 24
        # current level: 3
        # temporal position: 2
        self.embed = nn.Sequential(
            nn.Linear(53, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 3,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.context = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, codes, levels):
        b = codes.shape[0]

        if tuple(codes.shape[1:]) != (128, 6):
            raise ValueError(
                f"Unexpected FSQ shape: {codes.shape}"
            )

        if tuple(levels.shape) != (b, 16):
            raise ValueError(
                f"Unexpected levels: {levels.shape}"
            )

        coarse = codes[:, :16].float()

        mid = codes[:, 16:64].float().reshape(
            b, 16, 18
        )

        fine = codes[:, 64:128].float().reshape(
            b, 16, 24
        )

        # CRITICAL:
        # The controller cannot observe tokens
        # that have not been revealed yet.
        mid = mid * (
            levels >= 1
        ).unsqueeze(-1)

        fine = fine * (
            levels >= 2
        ).unsqueeze(-1)

        stage = F.one_hot(
            levels.long(),
            num_classes=3,
        ).float()

        theta = (
            torch.arange(
                16, device=codes.device
            ) + 0.5
        ) / 16

        pos = torch.stack(
            (
                torch.sin(2 * math.pi * theta),
                torch.cos(2 * math.pi * theta),
            ),
            dim=-1,
        )[None].expand(b, -1, -1)

        features = torch.cat(
            (
                coarse,
                mid,
                fine,
                stage,
                pos,
            ),
            dim=-1,
        )

        hidden = self.embed(features)
        hidden = self.context(hidden)

        return self.head(hidden)
