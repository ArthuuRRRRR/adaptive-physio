"""V8 unit tests: FSQ, leakage, budget and actor-critic."""

import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))

from adaptive_physio.models.belief_agent_v8 import (
    BeliefAgentV8,
    token_mask,
    count_tokens,
    feasible_actions,
    masked_belief_loss,
    choose_levels,
    fsq_digits,
)


def main():

    torch.manual_seed(13)

    model = BeliefAgentV8(
        width=64,
        heads=4,
        depth=1,
    ).eval()

    digits = torch.randint(
        0, 4, (2, 128, 6)
    )

    codes = (
        digits.float() - 2
    ) / 2

    levels = torch.tensor(
        [
            [0] * 16,
            [2, 1, 0, 0] * 4,
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        fsq_digits(codes), digits
    )

    visible = token_mask(levels)

    assert visible[0].sum().item() == 16

    assert (
        visible[1].sum().item()
        == int(count_tokens(levels)[1])
    )

    output = model(codes, levels)

    assert output["belief_logits"].shape == (
        2, 16, 7, 6, 4
    )

    assert output["action_logits"].shape == (
        2, 16
    )

    assert output["value"].shape == (2,)

    loss = masked_belief_loss(
        output["belief_logits"],
        codes,
        levels,
    )

    loss.backward()

    assert any(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        for p in model.predict.parameters()
    )

    print("PASS: belief gradients")

    # Hidden-token leakage test.
    with torch.no_grad():

        changed = torch.where(
            visible[:, :, None],
            codes,
            codes.flip(0),
        )

        output2 = model(
            changed, levels
        )

        for key in (
            "action_logits",
            "value",
            "belief_logits",
        ):

            error = (
                output[key] - output2[key]
            ).abs().max().item()

            assert error < 1e-6, (
                key, error
            )

    print("PASS: no hidden-token leakage")

    # Full allocation must reach 80 exactly.
    with torch.no_grad():

        chosen, used = choose_levels(
            model, codes
        )

        assert (used == 80).all()

        assert (
            token_mask(chosen).sum(1) == 80
        ).all()

        assert not feasible_actions(
            chosen, 80
        ).any()

        samples = model.sample_hidden(
            output["belief_logits"],
            n=4,
        )

        assert samples.shape == (
            4, 2, 16, 7, 6
        )

        allowed = torch.tensor(
            [-1.0, -0.5, 0.0, 0.5]
        )

        assert torch.isin(
            samples, allowed
        ).all()

    print("PASS: exact 80-token budget")
    print("PASS: stochastic FSQ samples")

    # Only actor and critic should train in phase 2.
    model.set_policy_phase()
    model.zero_grad(set_to_none=True)

    output = model(codes, levels)

    mask = feasible_actions(
        levels,
        80,
        model.feasible80,
    )

    logits = output[
        "action_logits"
    ].masked_fill(
        ~mask, -1e9
    )

    distribution = torch.distributions.Categorical(
        logits=logits
    )

    target = torch.tensor(
        [0.1, -0.1]
    )

    train_loss = (
        -(
            distribution.log_prob(
                distribution.sample()
            ) * target
        ).mean()
        + output["value"].square().mean()
    )

    train_loss.backward()

    assert any(
        p.grad is not None
        for p in model.actor.parameters()
    )

    assert any(
        p.grad is not None
        for p in model.critic.parameters()
    )

    assert all(
        p.grad is None
        for p in model.predict.parameters()
    )

    print("PASS: actor-critic gradients")
    print("PASS: frozen belief during PPO")
    print("ALL V8 TESTS PASSED")


if __name__ == "__main__":
    main()
