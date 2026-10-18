"""Temporally aligned hierarchical physiological tokenizer.

16 coarse + 48 mid + 64 fine = 128 FSQ tokens.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from adaptive_physio.models.timetok_physio_v2 import (
    FSQ,
    reconstruction_loss,
    reconstruction_stats,
)

LEVELS = (16, 64, 128)


class LocalEncoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 48, 11, stride=2, padding=5),
            nn.GELU(),
            nn.Conv1d(48, 96, 9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(96, channels, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GELU(),
        )

        self.coarse_norm = nn.LayerNorm(channels)
        self.mid_norm = nn.LayerNorm(channels)
        self.fine_norm = nn.LayerNorm(channels)

        self.to_coarse = nn.Linear(channels, 6)
        self.to_mid = nn.Linear(channels, 6)
        self.to_fine = nn.Linear(channels, 6)

        for head in (
            self.to_coarse,
            self.to_mid,
            self.to_fine,
        ):
            nn.init.normal_(head.weight, std=0.08)
            nn.init.zeros_(head.bias)

        # Three zero-mean contrasts for each group of 4.
        basis = torch.tensor([
            [0.5,  1 / math.sqrt(2), 0.0],
            [0.5, -1 / math.sqrt(2), 0.0],
            [-0.5, 0.0, 1 / math.sqrt(2)],
            [-0.5, 0.0, -1 / math.sqrt(2)],
        ])

        self.register_buffer(
            "mid_basis", basis, persistent=False
        )

    def forward(self, x):
        f = self.net(x)
        b, c, _ = f.shape

        coarse = F.adaptive_avg_pool1d(
            f, 16
        ).transpose(1, 2)

        mid = F.adaptive_avg_pool1d(
            f, 64
        ).transpose(1, 2).reshape(b, 16, 4, c)

        fine = F.adaptive_avg_pool1d(
            f, 128
        ).transpose(1, 2).reshape(b, 64, 2, c)

        # 16 coarse representations.
        z0 = self.to_coarse(
            self.coarse_norm(coarse) + coarse
        )

        # 3 detail coefficients per coarse region.
        mid_detail = torch.einsum(
            "bjid,im->bjmd",
            mid.float(),
            self.mid_basis,
        ).to(coarse.dtype)

        z1 = self.to_mid(
            self.mid_norm(mid_detail) + mid_detail
        ).reshape(b, 48, 6)

        # 1 contrast per mid region.
        fine_detail = (
            fine[:, :, 0] - fine[:, :, 1]
        ) / math.sqrt(2)

        z2 = self.to_fine(
            self.fine_norm(fine_detail) + fine_detail
        ).reshape(b, 64, 6)

        return torch.cat(
            (z0, z1, z2), dim=1
        )


class LocalDecoder(nn.Module):
    def __init__(self, length=3000, channels=96):
        super().__init__()

        self.length = length
        self.channels = channels

        self.from_coarse = nn.Linear(6, channels)
        self.from_mid = nn.Linear(6, channels)
        self.from_fine = nn.Linear(6, channels)

        self.mid_gain = nn.Parameter(
            torch.tensor(1.0)
        )
        self.fine_gain = nn.Parameter(
            torch.tensor(1.0)
        )

        basis = torch.tensor([
            [0.5,  1 / math.sqrt(2), 0.0],
            [0.5, -1 / math.sqrt(2), 0.0],
            [-0.5, 0.0, 1 / math.sqrt(2)],
            [-0.5, 0.0, -1 / math.sqrt(2)],
        ])

        self.register_buffer(
            "mid_basis", basis, persistent=False
        )

        # Local phase features within temporal cells.
        position = torch.arange(
            length, dtype=torch.float32
        )

        features = []

        for cells, harmonics in (
            (16, (1,)),
            (64, (1,)),
            (128, (1, 2, 4, 7)),
        ):
            phase = (
                position * cells / length
            ).remainder(1.0)

            for h in harmonics:
                features.extend([
                    torch.sin(
                        2 * math.pi * h * phase
                    ),
                    torch.cos(
                        2 * math.pi * h * phase
                    ),
                ])

        pos = torch.stack(features)[None]

        self.register_buffer(
            "pos", pos, persistent=False
        )

        self.decode = nn.Sequential(
            nn.Conv1d(
                channels + pos.shape[1],
                96, 5, padding=2
            ),
            nn.GELU(),
            nn.Conv1d(96, 64, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, codes, keep):
        if keep not in LEVELS:
            raise ValueError(
                f"Invalid granularity: {keep}"
            )

        b = codes.shape[0]

        # First 16 tokens cover the entire window.
        coarse = self.from_coarse(
            codes[:, :16]
        )

        mid = coarse.repeat_interleave(
            4, dim=1
        )

        # 48 additional detail tokens.
        if keep >= 64:
            detail = self.from_mid(
                codes[:, 16:64].reshape(
                    b, 16, 3, 6
                )
            )

            detail = torch.einsum(
                "bjmd,im->bjid",
                detail.float(),
                self.mid_basis,
            )

            mid = mid + self.mid_gain * (
                detail.reshape(
                    b, 64, self.channels
                )
            )

        fine = mid.repeat_interleave(
            2, dim=1
        )

        # 64 additional detail tokens.
        if keep >= 128:
            detail = self.from_fine(
                codes[:, 64:128]
            ) / math.sqrt(2)

            pair = torch.stack(
                (detail, -detail), dim=2
            ).reshape(
                b, 128, self.channels
            )

            fine = fine + self.fine_gain * pair

        # Each dense position has a known temporal cell.
        dense = F.interpolate(
            fine.transpose(1, 2),
            size=self.length,
            mode="nearest",
        )

        dense = torch.cat(
            (
                dense,
                self.pos.expand(
                    b, -1, -1
                ).to(dense.dtype),
            ),
            dim=1,
        )

        return self.decode(dense)


class PhysioTokenizerV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = LocalEncoder()
        self.fsq = FSQ()
        self.reconstructor = LocalDecoder()

    def tokenize(self, x):
        return self.fsq(
            self.encoder(x)
        )

    def forward(self, x, keep):
        codes, ids = self.tokenize(x)

        reconstruction = self.reconstructor(
            codes, keep
        )

        return reconstruction, ids
