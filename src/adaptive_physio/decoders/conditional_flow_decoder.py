
import math

import torch
import torch.nn as nn


def sinusoidal_positions(length, dim, device, dtype):
    """Fixed temporal position embeddings [1, length, dim]."""
    if dim % 2:
        raise ValueError("hidden_dim must be even")

    pos = torch.arange(
        length, device=device, dtype=torch.float32
    ).unsqueeze(1)

    scale = torch.exp(
        torch.arange(
            0, dim, 2, device=device, dtype=torch.float32
        ) * (-math.log(10000.0) / dim)
    )

    emb = torch.zeros(
        1, length, dim, device=device, dtype=torch.float32
    )
    emb[..., 0::2] = torch.sin(pos * scale)
    emb[..., 1::2] = torch.cos(pos * scale)

    return emb.to(dtype=dtype)


class ConditionalFlowDecoder(nn.Module):
    """
    Predict v_theta(x_tau, tau | discrete token prefix).

    x_tau: [B, 1, T]
    tau:   [B]
    codes: [B, N, 6]

    Output: predicted velocity [B, 1, T]
    """

    def __init__(
        self,
        code_dim=6,
        hidden_dim=64,
        num_heads=4,
        max_tokens=128,
        downsample=8,
    ):
        super().__init__()

        if hidden_dim % 2 or hidden_dim % num_heads:
            raise ValueError(
                "hidden_dim must be even and divisible by num_heads"
            )

        self.max_tokens = max_tokens
        self.downsample = downsample

        # Noisy signal -> temporal features
        self.signal_in = nn.Conv1d(
            1, hidden_dim,
            kernel_size=15,
            stride=downsample,
            padding=7,
        )

        # Quantized tokens -> conditioning embeddings
        self.token_in = nn.Linear(code_dim, hidden_dim)

        self.token_pos = nn.Parameter(
            torch.randn(1, max_tokens, hidden_dim) * 0.02
        )

        # Flow time embedding
        self.time_in = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temporal features attend to discrete tokens
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )

        # Velocity field at original signal length
        self.signal_out = nn.ConvTranspose1d(
            hidden_dim, 1,
            kernel_size=downsample,
            stride=downsample,
        )

    def forward(self, x_tau, tau, codes):
        if x_tau.ndim != 3 or x_tau.size(1) != 1:
            raise ValueError("x_tau must have shape [B, 1, T]")

        batch, _, samples = x_tau.shape

        if samples % self.downsample:
            raise ValueError(
                "T must be divisible by downsample"
            )

        if tau.shape != (batch,):
            raise ValueError("tau must have shape [B]")

        if codes.ndim != 3 or codes.shape[0] != batch:
            raise ValueError(
                "codes must have shape [B, N, code_dim]"
            )

        n = codes.shape[1]
        if not 1 <= n <= self.max_tokens:
            raise ValueError("Invalid token prefix length")

        # Encode noisy signal
        h = self.signal_in(x_tau).transpose(1, 2)

        # Add flow-time conditioning
        h = h + self.time_in(tau[:, None])[:, None, :]

        # Add temporal positions
        h = h + sinusoidal_positions(
            h.size(1), h.size(2), h.device, h.dtype
        )

        # Encode the available token prefix
        context = (
            self.token_in(codes)
            + self.token_pos[:, :n]
        )

        # Cross-attention
        attn, _ = self.cross_attention(
            h, context, context,
            need_weights=False,
        )

        h = self.norm(h + attn).transpose(1, 2)

        # Temporal processing
        h = h + self.temporal(h)

        return self.signal_out(h)

