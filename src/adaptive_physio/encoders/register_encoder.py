
import torch
import torch.nn as nn


class RegisterEncoder(nn.Module):
    """
    Physiological signal -> continuous register tokens.

    Input:  [B, 1, T]
    Output: [B, M, D]

    B = batch size
    M = number of register tokens
    D = embedding dimension
    """

    def __init__(
        self,
        sample_rate: int,
        patch_seconds: float = 0.25,
        max_window_seconds: float = 30.0,
        embedding_dim: int = 128,
        num_registers: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        self.patch_size = round(sample_rate * patch_seconds)
        max_samples = round(sample_rate * max_window_seconds)

        if self.patch_size < 1:
            raise ValueError("Patch size must be positive.")

        if max_samples % self.patch_size != 0:
            raise ValueError("Maximum window must fit whole patches.")

        self.max_patches = max_samples // self.patch_size
        self.num_registers = num_registers

        # Raw signal patches -> embeddings
        self.patch_projection = nn.Linear(
            self.patch_size,
            embedding_dim,
        )

        # Learnable register tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_registers, embedding_dim) * 0.02
        )

        # Separate positional embeddings
        self.patch_pos = nn.Parameter(
            torch.randn(1, self.max_patches, embedding_dim) * 0.02
        )

        self.register_pos = nn.Parameter(
            torch.randn(1, num_registers, embedding_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected signal shape [B, 1, T].")

        B, _, T = x.shape

        if T % self.patch_size != 0:
            raise ValueError("Signal does not fit whole patches.")

        # [B, 1, T] -> [B, W, patch_size]
        patches = x.unfold(
            dimension=-1,
            size=self.patch_size,
            step=self.patch_size,
        ).squeeze(1)

        W = patches.shape[1]

        if W > self.max_patches:
            raise ValueError("Window exceeds configured maximum.")

        # [B, W, D]
        patch_tokens = self.patch_projection(patches)
        patch_tokens = patch_tokens + self.patch_pos[:, :W]

        # [B, M, D]
        registers = self.register_tokens.expand(B, -1, -1)
        registers = registers + self.register_pos

        # [B, W + M, D]
        tokens = torch.cat(
            [patch_tokens, registers],
            dim=1,
        )

        encoded = self.transformer(tokens)

        # Retain only the updated register tokens
        registers = encoded[:, W:, :]

        return self.norm(registers)

