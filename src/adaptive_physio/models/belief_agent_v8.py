"""V8: belief over unseen V3 FSQ details + token-budget actor-critic.

Frozen V3 layout:
  16 coarse tokens
  48 mid tokens: 3 per region
  64 fine tokens: 4 per region

The agent NEVER reads unrevealed code values.
Only acquisition and supervised belief targets access them.
"""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


REGIONS = 16
TOKENS = 128
DIM = 6
BUDGET = 80


def token_mask(levels):
    if levels.ndim != 2 or levels.shape[1] != REGIONS:
        raise ValueError("levels must have shape [B,16]")

    if not bool(((levels >= 0) & (levels <= 2)).all()):
        raise ValueError("levels must be 0/1/2")

    mid = (levels >= 1).repeat_interleave(3, 1)
    fine = (levels >= 2).repeat_interleave(4, 1)

    return torch.cat(
        (
            torch.ones_like(levels, dtype=torch.bool),
            mid,
            fine,
        ),
        dim=1,
    )


def count_tokens(levels):
    return (
        16
        + 3 * (levels >= 1).sum(1)
        + 4 * (levels >= 2).sum(1)
    )


@lru_cache(maxsize=None)
def _reachable(n0, n1, remaining):
    if remaining == 0:
        return True

    if remaining < 0:
        return False

    return (
        (
            n0 > 0
            and _reachable(n0 - 1, n1 + 1, remaining - 3)
        )
        or (
            n1 > 0
            and _reachable(n0, n1 - 1, remaining - 4)
        )
    )


def build_feasible_table(budget=BUDGET):
    table = torch.zeros(
        17, 17, budget + 1,
        dtype=torch.bool,
    )

    for n0 in range(17):
        for n1 in range(17 - n0):
            for rem in range(budget + 1):
                table[n0, n1, rem] = _reachable(
                    n0, n1, rem
                )

    return table


def feasible_actions(levels, budget=BUDGET, table=None):
    """Only actions from which exactly 80 tokens remain reachable."""

    if table is None:
        table = build_feasible_table(budget).to(levels.device)

    used = count_tokens(levels)

    cost = torch.where(levels == 0, 3, 4)
    remaining = budget - used[:, None] - cost

    n0 = (levels == 0).sum(1, keepdim=True)
    n1 = (levels == 1).sum(1, keepdim=True)

    next_n0 = n0 - (levels == 0).long()
    next_n1 = (
        n1
        + (levels == 0).long()
        - (levels == 1).long()
    )

    good = (
        (levels < 2)
        & (remaining >= 0)
        & (next_n0 >= 0)
        & (next_n1 >= 0)
        & (next_n0 + next_n1 <= 16)
    )

    lookup = table[
        next_n0.clamp(0, 16),
        next_n1.clamp(0, 16),
        remaining.clamp(0, budget),
    ]

    return good & lookup


def fsq_digits(codes):
    """FSQ values {-1,-0.5,0,0.5} -> digits {0,1,2,3}."""
    return (
        codes.float()
        .mul(2)
        .add(2)
        .round()
        .long()
    ).clamp(0, 3)


class BeliefAgentV8(nn.Module):

    def __init__(self, width=128, heads=4, depth=2):
        super().__init__()

        self.config = dict(
            width=width,
            heads=heads,
            depth=depth,
        )

        self.code_proj = nn.Linear(DIM, width)
        self.pos_emb = nn.Embedding(TOKENS, width)
        self.region_emb = nn.Embedding(REGIONS, width)
        self.tier_emb = nn.Embedding(3, width)
        self.mask_query_emb = nn.Embedding(7, width)

        token_region = (
            list(range(16))
            + [r for r in range(16) for _ in range(3)]
            + [r for r in range(16) for _ in range(4)]
        )

        token_tier = [0] * 16 + [1] * 48 + [2] * 64

        self.register_buffer(
            "token_region",
            torch.tensor(token_region, dtype=torch.long),
            persistent=False,
        )

        self.register_buffer(
            "token_tier",
            torch.tensor(token_tier, dtype=torch.long),
            persistent=False,
        )

        detail_positions = [
            list(range(16 + r * 3, 16 + (r + 1) * 3))
            + list(range(64 + r * 4, 64 + (r + 1) * 4))
            for r in range(16)
        ]

        self.register_buffer(
            "detail_positions",
            torch.tensor(detail_positions, dtype=torch.long),
            persistent=False,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            depth,
            enable_nested_tensor=False,
        )

        self.cross = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )

        # Predict categorical distribution over four FSQ values.
        self.predict = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, DIM * 4),
        )

        actor_size = width * 3 + 6 + 6 + 2 + 3

        self.actor = nn.Sequential(
            nn.Linear(actor_size, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(width * 2 + 3, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

        self.register_buffer(
            "feasible80",
            build_feasible_table(80),
            persistent=False,
        )

    def set_policy_phase(self):
        """Freeze belief model. Train only actor and critic."""

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith("actor.")
                or name.startswith("critic.")
            )

    def forward(self, codes, levels, budget=BUDGET):

        if codes.ndim != 3 or codes.shape[1:] != (128, 6):
            raise ValueError("codes must have shape [B,128,6]")

        if not (16 <= budget <= 128):
            raise ValueError("Budget must be between 16 and 128")

        b = codes.shape[0]
        device = codes.device

        # ----------------------------------------------------
        # 1. Visible token encoder
        # ----------------------------------------------------

        visible = token_mask(levels)

        positions = torch.arange(
            TOKENS, device=device
        )

        # Critical: hidden codes are removed before embedding.
        safe_codes = (
            codes.float()
            * visible.unsqueeze(-1).float()
        )

        memory = (
            self.code_proj(safe_codes)
            + self.pos_emb(positions)[None]
            + self.region_emb(self.token_region)[None]
            + self.tier_emb(self.token_tier)[None]
        )

        memory = self.encoder(
            memory,
            src_key_padding_mask=~visible,
        )

        summary = (
            memory * visible.unsqueeze(-1)
        ).sum(1) / visible.sum(1, keepdim=True)

        # ----------------------------------------------------
        # 2. Hidden detail queries
        # ----------------------------------------------------

        indexes = self.detail_positions.reshape(-1)

        queries = (
            self.pos_emb(indexes)[None]
            + self.region_emb(
                self.token_region[indexes]
            )[None]
            + self.tier_emb(
                self.token_tier[indexes]
            )[None]
            + self.mask_query_emb(
                torch.arange(
                    7, device=device
                ).repeat(16)
            )[None]
        )

        queries = queries.expand(b, -1, -1)

        belief, _ = self.cross(
            queries,
            memory,
            memory,
            key_padding_mask=~visible,
            need_weights=False,
        )

        belief = belief.reshape(b, 16, 7, -1)

        # Six FSQ dimensions, four categories per dimension.
        logits = self.predict(belief).reshape(
            b, 16, 7, 6, 4
        )

        prob = logits.softmax(-1)

        fsq_values = torch.tensor(
            [-1.0, -0.5, 0.0, 0.5],
            device=device,
        )

        mean = (prob * fsq_values).sum(-1)

        var = (
            prob
            * (
                fsq_values
                - mean.unsqueeze(-1)
            ).square()
        ).sum(-1)

        # ----------------------------------------------------
        # 3. Belief summaries for actor
        # ----------------------------------------------------

        pool_mid = belief[:, :, :3].mean(2)
        pool_fine = belief[:, :, 3:].mean(2)

        mean_mid = mean[:, :, :3].mean(2)
        mean_fine = mean[:, :, 3:].mean(2)

        var_mid = var[:, :, :3].mean(2)
        var_fine = var[:, :, 3:].mean(2)

        pool = torch.stack(
            (pool_mid, pool_fine),
            dim=2,
        )

        mean_pool = torch.stack(
            (mean_mid, mean_fine),
            dim=2,
        )

        var_pool = torch.stack(
            (var_mid, var_fine),
            dim=2,
        )

        remaining = (
            budget - count_tokens(levels)
        ).float() / budget

        action_cost = (
            torch.where(
                levels == 0, 3.0, 4.0
            ) / budget
        )

        extras = torch.stack(
            (
                remaining[:, None].expand(-1, 16),
                action_cost,
                levels.float() / 2,
            ),
            dim=-1,
        )

        region_features = self.region_emb(
            torch.arange(16, device=device)
        )[None, :, None].expand(
            b, -1, 2, -1
        )

        level_features = F.one_hot(
            levels.clamp(0, 2), 3
        ).float()[:, :, None, :2].expand(
            -1, -1, 2, -1
        )

        actor_input = torch.cat(
            (
                pool,
                summary[:, None, None].expand(
                    -1, 16, 2, -1
                ),
                region_features,
                mean_pool,
                var_pool,
                extras[:, :, None].expand(
                    -1, -1, 2, -1
                ),
                level_features,
            ),
            dim=-1,
        )

        action_logits = self.actor(
            actor_input
        ).squeeze(-1)

        # Select the current legal refinement type.
        chosen_logits = action_logits.gather(
            2,
            levels.clamp(max=1).unsqueeze(-1),
        ).squeeze(-1)

        # ----------------------------------------------------
        # 4. Critic
        # ----------------------------------------------------

        critic_input = torch.cat(
            (
                summary,
                belief.mean((1, 2)),
                remaining[:, None],
                (levels >= 1).float().mean(
                    1, keepdim=True
                ),
                (levels >= 2).float().mean(
                    1, keepdim=True
                ),
            ),
            dim=-1,
        )

        value = self.critic(
            critic_input
        ).squeeze(-1)

        return {
            "action_logits": chosen_logits,
            "value": value,
            "belief_logits": logits,
            "visible": visible,
        }

    @torch.no_grad()
    def sample_hidden(self, belief_logits, n=4):
        """Sample plausible discrete FSQ hypotheses."""

        prob = belief_logits.softmax(-1)
        flat = prob.reshape(-1, 4)

        samples = torch.multinomial(
            flat,
            n,
            replacement=True,
        ).T

        return (
            samples.reshape(
                n, *prob.shape[:-1]
            ).float() - 2
        ) / 2


def masked_belief_loss(pred_logits, codes, levels):

    truth = torch.cat(
        (
            fsq_digits(
                codes[:, 16:64]
            ).reshape(-1, 16, 3, 6),
            fsq_digits(
                codes[:, 64:128]
            ).reshape(-1, 16, 4, 6),
        ),
        dim=2,
    )

    hidden = torch.cat(
        (
            (levels == 0)[:, :, None].expand(
                -1, -1, 3
            ),
            (levels < 2)[:, :, None].expand(
                -1, -1, 4
            ),
        ),
        dim=2,
    )

    if not bool(hidden.any()):
        raise ValueError("No hidden tokens")

    return F.cross_entropy(
        pred_logits[hidden].reshape(-1, 4),
        truth[hidden].reshape(-1),
    )


@torch.no_grad()
def choose_levels(model, codes, budget=80):

    b = codes.shape[0]

    levels = torch.zeros(
        b, 16,
        dtype=torch.long,
        device=codes.device,
    )

    for _ in range(32):

        if bool((count_tokens(levels) == budget).all()):
            break

        output = model(
            codes, levels, budget
        )

        valid = feasible_actions(
            levels,
            budget,
            model.feasible80,
        )

        active = count_tokens(levels) < budget

        if not bool(valid[active].any(dim=1).all()):
            raise RuntimeError("No feasible action")

        action = output[
            "action_logits"
        ].masked_fill(
            ~valid, -1e9
        ).argmax(1)

        rows = torch.where(active)[0]

        levels[
            rows, action[rows]
        ] += 1

    used = count_tokens(levels)

    if not bool((used == budget).all()):
        raise RuntimeError(
            f"Exact budget failed: {used.tolist()}"
        )

    return levels, used
