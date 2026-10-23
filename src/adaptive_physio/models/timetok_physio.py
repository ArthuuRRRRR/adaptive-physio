"""TimeTok-inspired hierarchical physiological tokenizer."""

import math
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = (16, 64, 128)


class SelfAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(
            b, n, 3, self.heads, d // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )
        return self.proj(
            y.transpose(1, 2).reshape(b, n, d)
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        return x + self.ff(self.n2(x))


class AdaTransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(
            dim, elementwise_affine=False
        )
        self.attn = SelfAttention(dim, heads)
        self.n2 = nn.LayerNorm(
            dim, elementwise_affine=False
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    @staticmethod
    def apply_mod(x, shift, scale):
        return (
            x * (1 + scale[:, None, :])
            + shift[:, None, :]
        )

    def forward(self, x, cond):
        s1, c1, g1, s2, c2, g2 = self.mod(
            cond
        ).chunk(6, dim=-1)

        x = x + g1[:, None, :] * self.attn(
            self.apply_mod(self.n1(x), s1, c1)
        )
        x = x + g2[:, None, :] * self.ff(
            self.apply_mod(self.n2(x), s2, c2)
        )
        return x


class RegisterEncoder(nn.Module):
    def __init__(
        self,
        length=3000,
        patch=25,
        dim=256,
        depth=3,
        heads=4,
    ):
        super().__init__()
        assert length % patch == 0

        self.length = length
        self.patch = patch
        self.n_patches = length // patch

        self.patch_in = nn.Linear(patch, dim)
        self.patch_pos = nn.Parameter(
            torch.randn(1, self.n_patches, dim) * 0.02
        )
        self.registers = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02
        )
        self.register_pos = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.to_latents = nn.Linear(dim, 6)

        nn.init.zeros_(self.to_latents.weight)
        nn.init.zeros_(self.to_latents.bias)

        p = self.n_patches
        allowed = torch.zeros(
            p + 128, p + 128, dtype=torch.bool
        )
        allowed[:p, :p] = True
        allowed[p:, :p] = True
        allowed[p:, p:] = torch.ones(
            128, 128, dtype=torch.bool
        ).tril()

        self.register_buffer(
            "attention_allowed",
            allowed,
            persistent=False,
        )

    def forward(self, x):
        assert x.ndim == 3
        assert x.shape[1:] == (1, self.length)

        b = x.shape[0]
        patches = x.reshape(
            b, self.n_patches, self.patch
        )

        p = (
            self.patch_in(patches)
            + self.patch_pos
        )
        r = (
            self.registers.expand(b, -1, -1)
            + self.register_pos
        )

        h = torch.cat([p, r], dim=1)

        for block in self.blocks:
            h = block(h, self.attention_allowed)

        return self.to_latents(
            self.norm(h[:, self.n_patches:])
        )


class FSQ(nn.Module):
    def __init__(
        self,
        levels=(4, 4, 4, 4, 4, 4),
        eps=1e-3,
    ):
        super().__init__()
        self.eps = eps

        self.register_buffer(
            "levels",
            torch.tensor(levels, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "basis",
            torch.tensor(
                [
                    math.prod(levels[:i])
                    for i in range(len(levels))
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.codebook_size = math.prod(levels)

    def forward(self, latents):
        z = latents.float()
        lv = self.levels.to(dtype=z.dtype)

        half_l = (
            (lv - 1) * (1.0 + self.eps) / 2
        )
        offset = (
            (self.levels % 2 == 0).to(z.dtype)
            * 0.5
        )
        shift = torch.atanh(offset / half_l)

        bounded = (
            (z + shift).tanh() * half_l - offset
        )

        rounded = (
            bounded
            + (bounded.round() - bounded).detach()
        )
        half_width = (
            self.levels // 2
        ).to(z.dtype)

        codes = rounded / half_width

        digits = (
            rounded.detach().long()
            + self.levels // 2
        ).clamp(min=0)
        digits = torch.minimum(
            digits, self.levels - 1
        )
        ids = (digits * self.basis).sum(dim=-1)

        return codes, ids

    def indices_to_codes(self, ids):
        digits = (
            (ids[..., None] // self.basis)
            % self.levels
        )
        return (
            digits.float()
            - (self.levels // 2)
        ) / (self.levels // 2)


class NestedTokens(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.project = nn.Linear(6, dim)
        self.pos = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02
        )
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

    def forward(
        self, codes, keep, half_gradient=True
    ):
        b, n, _ = codes.shape
        assert n == 128

        position = torch.arange(
            n, device=codes.device
        )[None, :]
        valid = position < keep[:, None]

        if half_gradient and self.training:
            early = (
                position < (keep[:, None] // 2)
            )
            codes = torch.where(
                early[..., None],
                codes.detach(),
                codes,
            )

        embeddings = (
            self.project(codes) + self.pos
        )
        return torch.where(
            valid[..., None],
            embeddings,
            self.mask_token + self.pos,
        )


def time_embedding(t, dim):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(
            half,
            device=t.device,
            dtype=torch.float32,
        ) / half
    )
    angle = (
        t.float()[:, None]
        * 1000
        * frequencies[None]
    )
    return torch.cat(
        [angle.sin(), angle.cos()], dim=-1
    )


class FlowTransformer(nn.Module):
    def __init__(
        self,
        length=3000,
        patch=25,
        dim=256,
        depth=3,
        heads=4,
    ):
        super().__init__()
        self.patch = patch
        self.n_patches = length // patch

        self.signal_in = nn.Linear(patch, dim)
        self.signal_pos = nn.Parameter(
            torch.randn(
                1, self.n_patches, dim
            ) * 0.02
        )
        self.tokens = NestedTokens(dim)

        self.time_in = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.level = nn.Embedding(3, dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.blocks = nn.ModuleList([
            AdaTransformerBlock(dim, heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noised, t, codes, keep):
        b = noised.shape[0]
        p = self.signal_in(
            noised.reshape(
                b, self.n_patches, self.patch
            )
        ) + self.signal_pos

        tok = self.tokens(codes, keep)

        idx = torch.searchsorted(
            torch.tensor(
                LEVELS, device=keep.device
            ),
            keep,
        )
        cond = self.fuse(torch.cat([
            self.time_in(
                time_embedding(t, p.shape[-1])
            ),
            self.level(idx),
        ], dim=-1))

        h = torch.cat([p, tok], dim=1)
        for block in self.blocks:
            h = block(h, cond)

        pred = self.out(
            self.norm(h[:, :self.n_patches])
        )
        return pred.reshape(b, 1, -1)


class TokenOnlyDecoder(nn.Module):
    def __init__(
        self,
        length=3000,
        patch=25,
        dim=256,
        depth=2,
        heads=4,
    ):
        super().__init__()
        self.n_patches = length // patch
        self.tokens = NestedTokens(dim)
        self.queries = nn.Parameter(
            torch.randn(
                1, self.n_patches, dim
            ) * 0.02
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch)

    def forward(self, codes, keep):
        b = codes.size(0)
        h = torch.cat([
            self.queries.expand(b, -1, -1),
            self.tokens(codes, keep),
        ], dim=1)

        for block in self.blocks:
            h = block(h)

        return self.out(
            self.norm(h[:, :self.n_patches])
        ).reshape(b, 1, -1)


@torch.autocast(device_type="cuda", enabled=False)
def gaussian_target(x, keep, max_tokens=128):
    length = x.shape[-1]
    out = torch.empty_like(x)

    for k in LEVELS:
        ids = torch.where(keep == k)[0]
        if ids.numel() == 0:
            continue
        if k == max_tokens:
            out[ids] = x[ids]
            continue

        s = (
            math.log2(max_tokens) - math.log2(k)
        ) / math.log2(max_tokens)

        width = (
            1.0 + (0.5 * length - 1.0) * s
        )
        sigma = (width - 1.0) / 6.0
        radius = min(
            int(3 * sigma + 0.5),
            (length - 1) // 2,
        )

        coords = torch.arange(
            -radius,
            radius + 1,
            device=x.device,
            dtype=x.dtype,
        )
        weights = (
            -0.5 * (coords / sigma).square()
        ).exp()
        weights = (
            weights / weights.sum()
        ).reshape(1, 1, -1)

        section = x[ids]
        out[ids] = F.conv1d(
            F.pad(
                section,
                (radius, radius),
                mode="reflect",
            ),
            weights,
        )

    return out


class TimeTokPhysio(nn.Module):
    def __init__(
        self,
        length=3000,
        patch=25,
        dim=256,
    ):
        super().__init__()
        self.encoder = RegisterEncoder(
            length=length, patch=patch, dim=dim
        )
        self.fsq = FSQ()
        self.decoder = FlowTransformer(
            length=length, patch=patch, dim=dim
        )
        self.aux_decoder = TokenOnlyDecoder(
            length=length, patch=patch, dim=dim
        )

    def forward(self, x, keep, aux_weight=0.5):
        z = self.encoder(x)
        codes, ids = self.fsq(z)

        with torch.no_grad():
            clean = gaussian_target(x, keep)

        noise = torch.randn_like(clean)

        # TimeTok-style nonuniform noise level.
        u = torch.rand(
            x.shape[0], device=x.device
        )
        noise_level = (
            1 - u
            - 0.25 * (
                torch.cos(math.pi * u / 2).square()
                - 1 + u
            )
        )
        t = noise_level

        mixed = (
            t[:, None, None] * noise
            + (1 - t[:, None, None]) * clean
        )

        pred_velocity = self.decoder(
            mixed, t, codes, keep
        )
        fm_loss = F.mse_loss(
            pred_velocity.float(),
            (noise - clean).float(),
        )

        aux = self.aux_decoder(codes, keep)
        aux_loss = F.mse_loss(
            aux.float(),
            clean.float(),
        )

        return fm_loss + aux_weight * aux_loss, {
            "fm": fm_loss.detach(),
            "aux": aux_loss.detach(),
            "unique": torch.unique(
                ids.detach()
            ).numel(),
        }
