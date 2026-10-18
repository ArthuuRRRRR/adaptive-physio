"""TimeTok-inspired physiological tokenizer V2."""

import math
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = (16, 64, 128)


class AttentionBlock(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, allowed=None):
        b, n, d = x.shape
        z = self.qkv(self.norm1(x)).reshape(
            b, n, 3, self.heads, d // self.heads
        )
        q, k, v = z.permute(
            2, 0, 3, 1, 4
        ).unbind(0)

        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed
        )
        x = x + self.proj(
            a.transpose(1, 2).reshape(b, n, d)
        )
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    def __init__(
        self, length=3000, patch=25,
        dim=256, depth=3
    ):
        super().__init__()
        assert length % patch == 0

        self.patches = length // patch
        self.patch = patch

        self.patch_proj = nn.Linear(patch, dim)
        self.patch_pos = nn.Parameter(
            torch.randn(1, self.patches, dim) * 0.02
        )
        self.registers = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02
        )
        self.register_pos = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            AttentionBlock(dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Linear(dim, 6)

        # Broader initialization to explore FSQ bins.
        nn.init.normal_(
            self.to_latent.weight, std=0.04
        )
        nn.init.zeros_(self.to_latent.bias)

        p = self.patches
        allowed = torch.zeros(
            p + 128, p + 128,
            dtype=torch.bool
        )
        allowed[:p, :p] = True
        allowed[p:, :p] = True
        allowed[p:, p:] = torch.ones(
            128, 128, dtype=torch.bool
        ).tril()

        self.register_buffer(
            "allowed", allowed, persistent=False
        )

    def forward(self, x):
        b = x.shape[0]
        assert x.shape[1:] == (
            1, self.patches * self.patch
        )

        patches = self.patch_proj(
            x.reshape(
                b, self.patches, self.patch
            )
        ) + self.patch_pos

        regs = (
            self.registers.expand(b, -1, -1)
            + self.register_pos
        )

        h = torch.cat((patches, regs), dim=1)

        for block in self.blocks:
            h = block(h, self.allowed)

        return self.to_latent(
            self.norm(h[:, self.patches:])
        )


class FSQ(nn.Module):
    def __init__(self):
        super().__init__()

        lv = torch.tensor(
            [4] * 6, dtype=torch.long
        )
        self.register_buffer(
            "levels", lv, persistent=False
        )
        self.register_buffer(
            "basis",
            torch.tensor(
                [4 ** i for i in range(6)],
                dtype=torch.long
            ),
            persistent=False
        )

    def forward(self, z):
        lv = self.levels.to(z.dtype)
        half = (lv - 1) * 1.001 / 2
        shift = torch.atanh(
            torch.full_like(half, 0.5) / half
        )

        bounded = (
            torch.tanh(z.float() + shift.float())
            * half.float() - 0.5
        )

        # Straight-through quantization.
        quant = bounded + (
            bounded.round() - bounded
        ).detach()

        codes = (quant / 2).to(z.dtype)

        digit = (
            quant.detach().long() + 2
        ).clamp(0, 3)

        ids = (
            digit * self.basis
        ).sum(-1)

        return codes, ids

    def decode_ids(self, ids):
        digit = (
            ids[..., None] // self.basis
        ) % self.levels

        return (digit.float() - 2) / 2


class PrefixDecoder(nn.Module):
    """Deterministic reconstruction from active tokens."""

    def __init__(
        self, length=3000, patch=25,
        dim=256, depth=3
    ):
        super().__init__()
        self.patch = patch
        self.patches = length // patch

        self.queries = nn.Parameter(
            torch.randn(
                1, self.patches, dim
            ) * 0.02
        )
        self.query_pos = nn.Parameter(
            torch.randn(
                1, self.patches, dim
            ) * 0.02
        )

        self.token_in = nn.Linear(6, dim)
        self.token_pos = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02
        )

        layer = nn.TransformerDecoderLayer(
            dim, 4, dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(
            layer, depth,
            norm=nn.LayerNorm(dim)
        )
        self.to_patch = nn.Linear(dim, patch)

    def forward(self, codes, keep):
        assert keep in LEVELS
        b = codes.shape[0]

        # Only real active tokens.
        memory = (
            self.token_in(codes[:, :keep])
            + self.token_pos[:, :keep]
        )
        queries = (
            self.queries + self.query_pos
        ).expand(b, -1, -1)

        y = self.decoder(queries, memory)
        return self.to_patch(y).reshape(
            b, 1, -1
        )


class FlowDecoder(nn.Module):
    def __init__(
        self, length=3000, patch=25,
        dim=256, depth=3
    ):
        super().__init__()
        self.patch = patch
        self.patches = length // patch

        self.signal_in = nn.Linear(patch, dim)
        self.signal_pos = nn.Parameter(
            torch.randn(
                1, self.patches, dim
            ) * 0.02
        )
        self.token_in = nn.Linear(6, dim)
        self.token_pos = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.level_emb = nn.Embedding(3, dim)

        layer = nn.TransformerDecoderLayer(
            dim, 4, dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(
            layer, depth,
            norm=nn.LayerNorm(dim)
        )
        self.to_patch = nn.Linear(dim, patch)

        nn.init.zeros_(self.to_patch.weight)
        nn.init.zeros_(self.to_patch.bias)

    def forward(self, x_t, t, codes, keep):
        assert keep in LEVELS
        b = x_t.shape[0]

        n = (
            self.signal_in(
                x_t.reshape(
                    b, self.patches, self.patch
                )
            ) + self.signal_pos
        )

        half = n.shape[-1] // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half, device=t.device
            ).float() / half
        )

        angle = (
            t.float()[:, None]
            * 1000.0
            * frequencies[None]
        )
        emb = torch.cat(
            (angle.sin(), angle.cos()), dim=1
        ).to(n.dtype)

        n = (
            n
            + self.time_mlp(emb)[:, None, :]
            + self.level_emb.weight[
                LEVELS.index(keep)
            ][None, None, :]
        )

        memory = (
            self.token_in(codes[:, :keep])
            + self.token_pos[:, :keep]
        )

        output = self.decoder(n, memory)
        return self.to_patch(output).reshape(
            b, 1, -1
        )


class PhysioTokenizerV2(nn.Module):
    def __init__(
        self, length=3000,
        patch=25, dim=256
    ):
        super().__init__()
        self.encoder = Encoder(
            length, patch, dim
        )
        self.fsq = FSQ()
        self.reconstructor = PrefixDecoder(
            length, patch, dim
        )
        self.flow = FlowDecoder(
            length, patch, dim
        )

    def tokenize(self, x):
        return self.fsq(self.encoder(x))


def reconstruction_loss(
    pred, real,
    spectral_weight=0.25,
    derivative_weight=0.1
):
    """Waveform + derivative + multiresolution STFT."""
    a = pred.float()
    b = real.float()

    temporal = F.l1_loss(a, b)

    derivative = F.l1_loss(
        a[..., 1:] - a[..., :-1],
        b[..., 1:] - b[..., :-1],
    )

    spectral = a.new_zeros(())

    # FFT always in float32.
    with torch.autocast(
        device_type=a.device.type,
        enabled=False
    ):
        for size in (64, 256, 512):
            window = torch.hann_window(
                size, device=a.device,
                dtype=torch.float32
            )

            sa = torch.stft(
                a[:, 0],
                n_fft=size,
                hop_length=size // 4,
                window=window,
                return_complex=True
            ).abs()

            sb = torch.stft(
                b[:, 0],
                n_fft=size,
                hop_length=size // 4,
                window=window,
                return_complex=True
            ).abs()

            spectral = spectral + F.l1_loss(
                torch.log1p(sa),
                torch.log1p(sb)
            ) / 3

    total = (
        temporal
        + derivative_weight * derivative
        + spectral_weight * spectral
    )

    return total, {
        "mae": temporal.detach(),
        "spec": spectral.detach(),
        "diff": derivative.detach(),
    }


def reconstruction_stats(pred, target):
    p = pred.float().flatten(1)
    t = target.float().flatten(1)

    mae = (p - t).abs().mean(1)
    rmse = (p - t).square().mean(1).sqrt()

    pc = p - p.mean(1, keepdim=True)
    tc = t - t.mean(1, keepdim=True)

    corr = (pc * tc).sum(1) / (
        (
            pc.square().sum(1)
            * tc.square().sum(1)
        ).sqrt().clamp_min(1e-8)
    )

    return mae, rmse, corr


def gentle_target(x, keep, hz=100):
    """Mild, physiology-aware flow targets."""
    sigma_seconds = {
        16: 0.04,
        64: 0.01,
        128: 0.0,
    }[keep]

    if sigma_seconds == 0:
        return x

    sigma = sigma_seconds * hz
    radius = math.ceil(sigma * 3)

    pts = torch.arange(
        -radius, radius + 1,
        device=x.device
    ).float()

    weights = torch.exp(
        -0.5 * (pts / sigma).square()
    )
    weights = (
        weights / weights.sum()
    ).reshape(1, 1, -1)

    with torch.autocast(
        device_type=x.device.type,
        enabled=False
    ):
        return F.conv1d(
            F.pad(
                x.float(),
                (radius, radius),
                mode="reflect"
            ),
            weights
        )


def flow_loss(flow, codes, x, keep):
    clean = gentle_target(x, keep)
    noise = torch.randn_like(clean)

    t = torch.rand(
        clean.shape[0],
        device=clean.device
    )

    mixed = (
        t[:, None, None] * noise
        + (1 - t[:, None, None]) * clean
    )

    velocity = flow(
        mixed, t, codes, keep
    )

    return F.mse_loss(
        velocity.float(),
        (noise - clean).float()
    )
