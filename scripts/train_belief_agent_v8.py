"""V8 training.

Phase 1: masked FSQ belief prediction.
Phase 2: episodic PPO with terminal SingLEM reward.

Frozen: tokenizer V3 and pretrained SingLEM.
"""

import argparse
import math
from pathlib import Path

import torch
from torch.distributions import Categorical

from adaptive_physio.models.belief_agent_v8 import (
    BeliefAgentV8,
    count_tokens,
    feasible_actions,
    masked_belief_loss,
    choose_levels,
)

from adaptive_physio.models.timetok_physio_v3 import (
    reconstruction_stats,
)

from evaluate_timetok_v3_adaptive import decode_mixed

from train_temporal_agent_physio_v5 import (
    setup,
    data_loader,
    normalize,
    get_codes,
    physio_error,
)


def loader(split, n, args, epoch=0):
    dataloader, _ = data_loader(
        split,
        "eeg",
        n,
        args.batch_size,
        args.workers if split == "train" else 0,
        args.seed,
        epoch=epoch,
    )

    return dataloader


def make_state(codes, budget=80):
    """Random legal observation states for belief pretraining."""

    b = codes.size(0)
    device = codes.device

    levels = torch.zeros(
        b, 16,
        dtype=torch.long,
        device=device,
    )

    limits = torch.randint(
        0, 21,
        (b,),
        device=device,
    )

    for step in range(20):

        active = (
            (step < limits)
            & (count_tokens(levels) < budget)
        )

        if not bool(active.any()):
            break

        valid = feasible_actions(
            levels, budget
        )

        weights = valid.float()

        weights[~active] = 0
        weights[~active, 0] = 1

        picks = torch.multinomial(
            weights, 1
        ).squeeze(-1)

        rows = torch.where(active)[0]

        levels[
            rows, picks[rows]
        ] += 1

    return levels


@torch.no_grad()
def score_terminal(
    codes,
    x,
    levels,
    backbone,
    teacher,
    norm,
):
    """Terminal SingLEM reward relative to Fixed80."""

    fixed = torch.ones_like(levels)
    fixed[:, [0, 4, 8, 12]] = 2

    ours = decode_mixed(
        backbone.reconstructor,
        codes,
        levels,
    ).float()

    base = decode_mixed(
        backbone.reconstructor,
        codes,
        fixed,
    ).float()

    reference = teacher(x, norm)

    ours_dist = physio_error(
        teacher(ours, norm),
        reference,
    )

    fixed_dist = physio_error(
        teacher(base, norm),
        reference,
    )

    ours_mae, _, ours_corr = reconstruction_stats(
        ours, x
    )

    fixed_mae, _, fixed_corr = reconstruction_stats(
        base, x
    )

    return {
        "reward": fixed_dist - ours_dist,
        "singlem_v8": ours_dist,
        "singlem_fixed80": fixed_dist,
        "mae_v8": ours_mae,
        "mae_fixed80": fixed_mae,
        "corr_v8": ours_corr,
        "corr_fixed80": fixed_corr,
    }


@torch.no_grad()
def collect(
    model,
    codes,
    x,
    backbone,
    teacher,
    norm,
    temperature=1.0,
):
    """Collect on-policy complete trajectories."""

    model.eval()

    b = codes.shape[0]
    device = codes.device

    levels = torch.zeros(
        b, 16,
        dtype=torch.long,
        device=device,
    )

    transitions = []

    for step in range(32):

        active = count_tokens(levels) < 80

        if not bool(active.any()):
            break

        result = model(codes, levels)

        valid = feasible_actions(
            levels,
            80,
            model.feasible80,
        )

        if not bool(valid[active].any(1).all()):
            raise RuntimeError("No legal action")

        logits = (
            result["action_logits"] / temperature
        ).masked_fill(
            ~valid, -1e9
        )

        dist = Categorical(logits=logits)

        actions = dist.sample()
        logprob = dist.log_prob(actions)

        transitions.append({
            "levels": levels.clone(),
            "actions": actions,
            "old_logp": logprob,
            "value": result["value"],
            "active": active,
        })

        rows = torch.where(active)[0]

        levels[
            rows, actions[rows]
        ] += 1

    if not bool((count_tokens(levels) == 80).all()):
        raise RuntimeError(
            "Rollout failed exact budget"
        )

    scores = score_terminal(
        codes,
        x,
        levels,
        backbone,
        teacher,
        norm,
    )

    return transitions, levels, scores


def ppo_update(
    model,
    optimizer,
    codes,
    episodes,
    rewards,
    args,
):
    """Clipped PPO with episodic terminal returns."""

    levels = torch.stack([
        q["levels"] for q in episodes
    ])

    actions = torch.stack([
        q["actions"] for q in episodes
    ])

    old_lp = torch.stack([
        q["old_logp"] for q in episodes
    ])

    old_v = torch.stack([
        q["value"] for q in episodes
    ])

    active = torch.stack([
        q["active"] for q in episodes
    ])

    returns = (
        rewards.detach()[None, :]
        .expand_as(old_v)
        * args.reward_scale
    )

    advantage = (
        returns - old_v
    ).detach()

    advantage = (
        advantage - advantage[active].mean()
    ) / advantage[active].std(
        unbiased=False
    ).clamp_min(1e-5)

    t, b = levels.shape[:2]

    valid_ix = (
        active.reshape(-1)
        .nonzero()
        .squeeze(1)
    )

    all_levels = (
        levels.reshape(-1, 16)
        .index_select(0, valid_ix)
    )

    all_codes = (
        codes.detach()
        .unsqueeze(0)
        .expand(t, -1, -1, -1)
        .reshape(t * b, 128, 6)
        .index_select(0, valid_ix)
    )

    all_actions = (
        actions.flatten()
        .index_select(0, valid_ix)
    )

    old_logp = (
        old_lp.flatten()
        .index_select(0, valid_ix)
    )

    targets = (
        returns.flatten()
        .index_select(0, valid_ix)
    )

    adv = (
        advantage.flatten()
        .index_select(0, valid_ix)
    )

    n = len(valid_ix)

    metrics = []
    model.train()

    for _ in range(args.ppo_epochs):

        ordering = torch.randperm(
            n, device=codes.device
        )

        for indices in ordering.split(
            args.ppo_minibatch
        ):

            obs = all_codes[indices]
            state = all_levels[indices]

            output = model(obs, state)

            valid = feasible_actions(
                state,
                80,
                model.feasible80,
            )

            dist = Categorical(
                logits=(
                    output["action_logits"]
                    / args.temperature
                ).masked_fill(
                    ~valid, -1e9
                )
            )

            new_logp = dist.log_prob(
                all_actions[indices]
            )

            ratio = torch.exp(
                new_logp - old_logp[indices]
            )

            clipped = torch.clamp(
                ratio,
                1 - args.clip,
                1 + args.clip,
            )

            actor_loss = -torch.minimum(
                ratio * adv[indices],
                clipped * adv[indices],
            ).mean()

            value_loss = (
                output["value"]
                - targets[indices]
            ).square().mean()

            entropy = dist.entropy().mean()

            loss = (
                actor_loss
                + args.vf_weight * value_loss
                - args.entropy_weight * entropy
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                1.0,
            )

            optimizer.step()

            metrics.append((
                float(actor_loss.item()),
                float(value_loss.item()),
                float(entropy.item()),
            ))

    avg = torch.tensor(metrics).mean(0)

    return {
        "actor": float(avg[0]),
        "critic": float(avg[1]),
        "entropy": float(avg[2]),
        "transitions": n,
    }


def save(
    model, path, phase, bc, teacher,
    args, optimizer=None, epoch=None, val=None
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "phase": phase,
            "version": "v8-belief-ppo-exact80",
            "epoch": epoch,
            "val": val,
            "normalization": bc["normalization"],
            "backbone_step": bc["step"],
            "teacher_sha256": teacher.sha256,
            "seed": args.seed,
            "budget": 80,
            "optimizer": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
        },
        path,
    )


def load(path, device, bc=None, teacher=None):

    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        ck.get("version") != "v8-belief-ppo-exact80"
        or ck.get("budget") != 80
    ):
        raise ValueError("Invalid V8 checkpoint")

    if bc is not None:
        if (
            ck["backbone_step"] != bc["step"]
            or ck["normalization"] != bc["normalization"]
        ):
            raise ValueError("Frozen V3 mismatch")

    if teacher is not None:
        if ck["teacher_sha256"] != teacher.sha256:
            raise ValueError("Frozen SingLEM mismatch")

    model = BeliefAgentV8(
        **ck["config"]
    ).to(device)

    model.load_state_dict(
        ck["model"], strict=True
    )

    return model, ck


@torch.no_grad()
def eval_belief(
    model, backbone, norm, device, args
):

    model.eval()

    total = 0.0
    seen = 0

    for raw in loader(
        "val", args.val_windows, args
    ):

        x = normalize(raw, norm, device)
        codes = get_codes(backbone, x)
        states = make_state(codes)

        output = model(codes, states)

        loss = masked_belief_loss(
            output["belief_logits"],
            codes,
            states,
        )

        total += float(loss) * len(x)
        seen += len(x)

    return total / max(seen, 1)


@torch.no_grad()
def eval_policy(
    model,
    backbone,
    teacher,
    norm,
    device,
    args,
):

    model.eval()

    total = {}
    seen = 0

    for raw in loader(
        "val", args.val_windows, args
    ):

        x = normalize(raw, norm, device)
        codes = get_codes(backbone, x)

        levels, used = choose_levels(
            model, codes
        )

        assert bool((used == 80).all())

        score = score_terminal(
            codes,
            x,
            levels,
            backbone,
            teacher,
            norm,
        )

        for key, val in score.items():
            total[key] = (
                total.get(key, 0.0)
                + float(val.sum())
            )

        seen += len(x)

    return {
        key: val / max(seen, 1)
        for key, val in total.items()
    }


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--phase",
        choices=("belief", "ppo", "eval"),
        required=True,
    )

    p.add_argument(
        "--backbone",
        default="outputs/checkpoints/timetok_v3_eeg_ae/frozen_for_agent.pt",
    )

    p.add_argument(
        "--singlem",
        default="third_party/SingLEM",
    )

    p.add_argument(
        "--output-dir",
        default="outputs/checkpoints/timetok_v3_agent_belief_v8",
    )

    p.add_argument("--checkpoint", default=None)
    p.add_argument("--train-windows", type=int, default=4000)
    p.add_argument("--val-windows", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-minibatch", type=int, default=64)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--vf-weight", type=float, default=0.5)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=1.0)

    args = p.parse_args()

    if (
        args.batch_size < 1
        or args.train_windows < 1
        or args.val_windows < 1
    ):
        p.error("Positive sample counts required")

    torch.manual_seed(args.seed)

    device, backbone, bc, teacher = setup(args)

    out = Path(args.output_dir)
    norm = bc["normalization"]

    # --------------------------------------------------------
    # PHASE 1: BELIEF PRETRAINING
    # --------------------------------------------------------

    if args.phase == "belief":

        model = BeliefAgentV8().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )

        best = math.inf

        for epoch in range(args.epochs):

            model.train()
            running = 0.0
            step = 0

            for raw in loader(
                "train",
                args.train_windows,
                args,
                epoch,
            ):

                x = normalize(
                    raw, norm, device
                )

                codes = get_codes(
                    backbone, x
                )

                states = make_state(codes)

                output = model(
                    codes, states
                )

                loss = masked_belief_loss(
                    output["belief_logits"],
                    codes,
                    states,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optimizer.step()

                step += 1
                running += float(loss)

                if step % args.log_every == 0:
                    print(
                        f"BELIEF epoch={epoch+1} "
                        f"batch={step} "
                        f"CE={running/step:.5f}",
                        flush=True,
                    )

            val = eval_belief(
                model,
                backbone,
                norm,
                device,
                args,
            )

            print(
                f"BELIEF VAL epoch={epoch+1} "
                f"CE={val:.6f}",
                flush=True,
            )

            save(
                model,
                out / "belief_last.pt",
                "belief",
                bc,
                teacher,
                args,
                optimizer,
                epoch + 1,
                val,
            )

            if val < best:
                best = val

                save(
                    model,
                    out / "belief_best.pt",
                    "belief",
                    bc,
                    teacher,
                    args,
                    optimizer,
                    epoch + 1,
                    val,
                )

        print(
            "BELIEF TRAINING COMPLETE",
            flush=True,
        )

        return

    # --------------------------------------------------------
    # LOAD PRETRAINED V8
    # --------------------------------------------------------

    default_checkpoint = (
        "belief_best.pt"
        if args.phase == "ppo"
        else "best.pt"
    )

    ck_path = (
        args.checkpoint
        or str(out / default_checkpoint)
    )

    model, ck = load(
        ck_path,
        device,
        bc,
        teacher,
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    if args.phase == "eval":

        metrics = eval_policy(
            model,
            backbone,
            teacher,
            norm,
            device,
            args,
        )

        print(
            "V8 VS FIXED80 VAL:",
            metrics,
            flush=True,
        )

        return

    # --------------------------------------------------------
    # PHASE 2: PPO
    # --------------------------------------------------------

    if ck["phase"] not in ("belief", "ppo"):
        raise ValueError(
            "Run belief pretraining first"
        )

    model.set_policy_phase()

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.lr,
        weight_decay=0.01,
    )

    best = -math.inf

    for epoch in range(args.epochs):

        step = 0
        total_reward = 0.0

        for raw in loader(
            "train",
            args.train_windows,
            args,
            epoch,
        ):

            x = normalize(
                raw, norm, device
            )

            codes = get_codes(
                backbone, x
            )

            with torch.no_grad():

                episodes, levels, score = collect(
                    model,
                    codes,
                    x,
                    backbone,
                    teacher,
                    norm,
                    args.temperature,
                )

            metrics = ppo_update(
                model,
                optimizer,
                codes,
                episodes,
                score["reward"],
                args,
            )

            step += 1

            total_reward += float(
                score["reward"].mean()
            )

            if step % args.log_every == 0:

                print(
                    f"PPO epoch={epoch+1} "
                    f"batch={step} "
                    f"R={total_reward/step:.6f} "
                    f"actor={metrics['actor']:.4f} "
                    f"critic={metrics['critic']:.4f} "
                    f"entropy={metrics['entropy']:.4f}",
                    flush=True,
                )

        val = eval_policy(
            model,
            backbone,
            teacher,
            norm,
            device,
            args,
        )

        print(
            f"PPO VAL epoch={epoch+1}: {val}",
            flush=True,
        )

        save(
            model,
            out / "last.pt",
            "ppo",
            bc,
            teacher,
            args,
            optimizer,
            epoch + 1,
            val,
        )

        if val["reward"] > best:
            best = val["reward"]

            save(
                model,
                out / "best.pt",
                "ppo",
                bc,
                teacher,
                args,
                optimizer,
                epoch + 1,
                val,
            )

    print(
        "PPO TRAINING COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
