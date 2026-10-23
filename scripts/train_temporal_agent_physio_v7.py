"""V7: all-action SingLEM counterfactual gains and on-policy state training."""
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from adaptive_physio.models.temporal_agent_v3 import TemporalRefinementAgent
from adaptive_physio.models.timetok_physio_v3 import reconstruction_stats
from evaluate_timetok_v3_adaptive import decode_mixed
from analyze_v4_v5_regions import fixed_budget
from train_temporal_agent_v4 import load_agent
from train_temporal_agent_physio_v5 import (
    setup, data_loader, normalize, get_codes, physio_error,
)

V4_PATH = "outputs/checkpoints/timetok_v3_agent_cost_eeg/best.pt"


def action_costs(states):
    return torch.where(states == 0, 3, 4)


@torch.no_grad()
def sample_states(codes, actor):
    """Half policy-visited, half randomly explored legal hierarchical states."""
    b = len(codes)
    dev = codes.device
    levels = torch.zeros(b, 16, device=dev, dtype=torch.long)
    spent = torch.full((b,), 16, device=dev, dtype=torch.long)
    budgets = torch.tensor([32, 48, 64, 80, 96, 112], device=dev)
    max_budget = budgets[torch.randint(6, (b,), device=dev)]
    n_moves = torch.randint(0, 29, (b,), device=dev)
    random_policy = torch.rand(b, device=dev) < 0.5

    previously_training = actor.training
    actor.eval()

    for t in range(28):
        costs = action_costs(levels)
        valid = (levels < 2) & (
            spent[:, None] + costs <= max_budget[:, None]
        )
        active = (t < n_moves) & valid.any(dim=1)
        if not active.any().item():
            break

        outputs = actor(codes, levels)
        scores = outputs.gather(
            2, levels.clamp(max=1)[..., None]
        ).squeeze(-1)

        chosen_policy = scores.masked_fill(
            ~valid, -1e9
        ).argmax(dim=1)

        chosen_random = torch.rand(
            b, 16, device=dev
        ).masked_fill(~valid, -1).argmax(dim=1)

        selected = torch.where(
            random_policy, chosen_random, chosen_policy
        )

        rows = torch.nonzero(active).flatten()
        idx = selected[rows]
        spent[rows] += costs[rows, idx]
        levels[rows, idx] += 1

    actor.train(previously_training)
    return levels


@torch.no_grad()
def counterfactual_gains(
    backbone, teacher, codes, x, norm, states
):
    """Recompute every one-step gain against the same full EEG reference."""
    b = len(x)
    r = torch.arange(16, device=x.device)
    valid = states < 2
    costs = action_costs(states).float()

    candidate = states[:, None, :].expand(
        -1, 16, -1
    ).clone()

    candidate[:, r, r] = (
        candidate[:, r, r] + 1
    ).clamp(max=2)

    base = decode_mixed(
        backbone.reconstructor, codes, states
    ).float()

    cf = decode_mixed(
        backbone.reconstructor,
        codes.repeat_interleave(16, dim=0),
        candidate.reshape(b * 16, 16),
    ).float()

    reference = teacher(x, norm)
    d0 = physio_error(teacher(base, norm), reference)

    d1 = physio_error(
        teacher(cf, norm),
        reference.repeat_interleave(16, dim=0),
    ).reshape(b, 16)

    gain = (d0[:, None] - d1) / costs
    gain = gain.masked_fill(~valid, 0.0)

    return gain, valid


def predicted_action_scores(agent, codes, states):
    prediction = agent(codes, states)

    if prediction.shape != (len(codes), 16, 2):
        raise RuntimeError(
            f"Unexpected controller output: {tuple(prediction.shape)}"
        )

    return prediction.gather(
        2, states.clamp(max=1)[..., None]
    ).squeeze(-1)


def agent_loss(agent, codes, states, raw_gains, valid, scale):
    target = (raw_gains / scale).detach()
    predicted = predicted_action_scores(
        agent, codes, states
    )

    regression = F.smooth_l1_loss(
        predicted[valid], target[valid], beta=1.0
    )

    diff = target[:, :, None] - target[:, None, :]
    pred_diff = (
        predicted[:, :, None] - predicted[:, None, :]
    )

    relevant = (
        valid[:, :, None]
        & valid[:, None, :]
        & (diff.abs() > 0.1)
    )

    weights = diff.abs().clamp(max=5.0) * relevant.float()

    ranking = (
        F.softplus(-pred_diff * diff.sign()) * weights
    ).sum() / weights.sum().clamp_min(1.0)

    max_target = target.masked_fill(
        ~valid, -1e9
    ).max(dim=1).values

    max_predicted = predicted.masked_fill(
        ~valid, -1e9
    ).max(dim=1).values

    threshold_near = (
        max_target.detach()
        + 0.4 * torch.randn_like(max_target)
    ).clamp_min(0)

    threshold_wide = 2.5 * torch.rand_like(max_target)

    threshold = torch.where(
        torch.rand_like(max_target) < 0.5,
        threshold_near,
        threshold_wide,
    )

    go_label = (max_target > threshold).float()

    stop = F.binary_cross_entropy_with_logits(
        ((max_predicted - threshold) / 0.4).clamp(-30, 30),
        go_label,
    )

    objective = regression + 0.3 * ranking + 0.15 * stop

    chosen = predicted.masked_fill(
        ~valid, -1e9
    ).argmax(dim=1)

    oracle = target.masked_fill(
        ~valid, -1e9
    ).argmax(dim=1)

    chosen_gain = target.gather(
        1, chosen[:, None]
    ).squeeze(1)

    best_gain = target.gather(
        1, oracle[:, None]
    ).squeeze(1)

    diagnostics = {
        "reg": regression.item(),
        "rank": ranking.item(),
        "stop": stop.item(),
        "top1": (chosen == oracle).float().mean().item(),
        "regret": (best_gain - chosen_gain).mean().item(),
        "go_acc": (
            (max_predicted > threshold)
            == (go_label > 0.5)
        ).float().mean().item(),
    }

    return objective, diagnostics


@torch.no_grad()
def calibrate(backbone, teacher, actor, norm, device, args):
    loader, _ = data_loader(
        "train", "eeg", args.calib_windows,
        args.batch_size, 0, args.seed,
    )

    total, count = 0.0, 0

    for raw in loader:
        x = normalize(raw, norm, device)
        codes = get_codes(backbone, x)
        states = sample_states(codes, actor)

        gain, valid = counterfactual_gains(
            backbone, teacher, codes, x, norm, states
        )

        total += gain[valid].abs().sum().item()
        count += int(valid.sum().item())

    scale = total / max(count, 1)

    if scale < 1e-9:
        raise RuntimeError(
            f"Degenerate gain scale: {scale}"
        )

    print(
        f"Frozen V7 SingLEM gain scale: {scale:.9f}",
        flush=True,
    )

    return scale


@torch.no_grad()
def validate(
    agent, reference_agent, backbone, teacher,
    loader, norm, device, scale
):
    agent.eval()

    totals = {
        k: 0.0 for k in (
            "loss", "top1", "regret",
            "v4_regret", "stop_acc"
        )
    }
    n = 0

    cuda_devices = (
        [torch.cuda.current_device()]
        if device.type == "cuda" else []
    )

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(1729)

        for raw in loader:
            x = normalize(raw, norm, device)
            codes = get_codes(backbone, x)

            states = sample_states(
                codes, reference_agent
            )

            gains, valid = counterfactual_gains(
                backbone, teacher, codes,
                x, norm, states
            )

            loss, metrics = agent_loss(
                agent, codes, states,
                gains, valid, scale
            )

            target = gains / scale

            v4scores = predicted_action_scores(
                reference_agent, codes, states
            )

            best = target.masked_fill(
                ~valid, -1e9
            ).max(dim=1).values

            v4_idx = v4scores.masked_fill(
                ~valid, -1e9
            ).argmax(dim=1)

            v4_gain = target.gather(
                1, v4_idx[:, None]
            ).squeeze(1)

            b = len(x)

            for key, value in (
                ("loss", loss.item()),
                ("top1", metrics["top1"]),
                ("regret", metrics["regret"]),
                ("v4_regret", (
                    best - v4_gain
                ).mean().item()),
                ("stop_acc", metrics["go_acc"]),
            ):
                totals[key] += b * value

            n += b

    return {
        key: value / n
        for key, value in totals.items()
    }


@torch.no_grad()
def evaluate(
    agent, reference_agent, backbone,
    teacher, norm, device, args
):
    loader, _ = data_loader(
        "val", "eeg", args.eval_windows,
        args.batch_size, 0, args.seed
    )

    output = Path(args.output_dir) / "fixed_budget_val.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for raw in loader:
        x = normalize(raw, norm, device)
        codes = get_codes(backbone, x)
        f_ref = teacher(x, norm)

        for budget in (64, 80, 96):
            for name, current in (
                ("V4", reference_agent),
                ("V7", agent),
            ):
                states, used = fixed_budget(
                    current, codes, budget
                )

                pred = decode_mixed(
                    backbone.reconstructor,
                    codes, states
                ).float()

                mae, _, corr = reconstruction_stats(
                    pred, x
                )

                physiological = physio_error(
                    teacher(pred, norm), f_ref
                )

                for i in range(len(x)):
                    rows.append({
                        "budget": budget,
                        "model": name,
                        "tokens": float(used[i]),
                        "mae": float(mae[i]),
                        "corr": float(corr[i]),
                        "singlem": float(physiological[i]),
                    })

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    for budget in (64, 80, 96):
        print(f"\nBUDGET {budget}")

        for model in ("V4", "V7"):
            subset = [
                r for r in rows
                if r["budget"] == budget
                and r["model"] == model
            ]

            def mean(key):
                return sum(
                    r[key] for r in subset
                ) / len(subset)

            print(
                f"{model} | "
                f"tokens={mean('tokens'):.2f} "
                f"SingLEM={mean('singlem'):.6f} "
                f"MAE={mean('mae'):.5f} "
                f"Corr={mean('corr'):.5f}"
            )

    print("Saved:", output)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=("smoke", "train", "eval"),
        default="smoke",
    )
    p.add_argument(
        "--backbone",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
    )
    p.add_argument(
        "--singlem",
        default="third_party/SingLEM",
    )
    p.add_argument(
        "--physio-weight",
        type=float, default=1.0
    )
    p.add_argument(
        "--output-dir",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_agent_physio_v7"
        ),
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--calib-windows", type=int, default=128)
    p.add_argument("--train-windows", type=int, default=15000)
    p.add_argument("--val-windows", type=int, default=256)
    p.add_argument("--eval-windows", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)

    args = p.parse_args()
    torch.manual_seed(args.seed)

    device, backbone, bc, teacher = setup(args)

    reference_agent, _ = load_agent(
        V4_PATH, device
    )
    reference_agent.eval()

    agent = TemporalRefinementAgent().to(device)

    checkpoint_path = (
        args.checkpoint
        or str(Path(args.output_dir) / "best.pt")
    )

    if args.mode == "eval":
        ck = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            ck["teacher_sha256"] != teacher.sha256
            or ck["backbone_step"] != bc["step"]
        ):
            raise ValueError(
                "Checkpoint compatibility mismatch"
            )

        agent.load_state_dict(ck["model"])
        agent.eval()

        evaluate(
            agent, reference_agent,
            backbone, teacher,
            bc["normalization"], device, args
        )
        return

    scale = calibrate(
        backbone, teacher, reference_agent,
        bc["normalization"], device, args
    )

    if args.mode == "smoke":
        loader, _ = data_loader(
            "train", "eeg", 4, 4, 0, args.seed
        )

        raw = next(iter(loader))
        x = normalize(
            raw, bc["normalization"], device
        )
        codes = get_codes(backbone, x)
        states = sample_states(codes, reference_agent)

        gain, valid = counterfactual_gains(
            backbone, teacher, codes, x,
            bc["normalization"], states
        )

        loss, stats = agent_loss(
            agent, codes, states, gain, valid, scale
        )

        loss.backward()

        assert any(
            param.grad is not None
            for param in agent.parameters()
        )
        assert torch.isfinite(loss)

        print(
            "SMOKE V7 PASSED",
            "loss", round(loss.item(), 6),
            stats,
            flush=True,
        )
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(
        agent.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(
        args.train_windows / args.batch_size
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=steps_per_epoch * args.epochs,
        eta_min=args.lr * 0.05,
    )

    val_loader, _ = data_loader(
        "val", "eeg", args.val_windows,
        args.batch_size, 0, args.seed
    )

    best = float("inf")
    step = 0

    for epoch in range(args.epochs):
        loader, _ = data_loader(
            "train", "eeg",
            args.train_windows,
            args.batch_size,
            args.workers, args.seed,
            epoch=epoch,
        )

        agent.train()

        for raw in loader:
            x = normalize(
                raw, bc["normalization"], device
            )

            with torch.no_grad():
                codes = get_codes(backbone, x)
                states = sample_states(codes, agent)

                gains, valid = counterfactual_gains(
                    backbone, teacher, codes, x,
                    bc["normalization"], states
                )

            opt.zero_grad(set_to_none=True)

            loss, stats = agent_loss(
                agent, codes, states,
                gains, valid, scale
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite V7 loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), 1.0
            )

            opt.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch+1}/{args.epochs} "
                    f"step={step} "
                    f"loss={loss.item():.5f} "
                    f"top1={stats['top1']:.3f} "
                    f"regret={stats['regret']:.4f}",
                    flush=True,
                )

        metrics = validate(
            agent, reference_agent,
            backbone, teacher, val_loader,
            bc["normalization"], device, scale
        )

        improved = metrics["loss"] < best
        best = min(best, metrics["loss"])

        ck = {
            "model": agent.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": step,
            "best": best,
            "gain_scale": scale,
            "backbone_step": bc["step"],
            "teacher_sha256": teacher.sha256,
            "version": "v7-full-action-singlem",
        }

        torch.save(ck, out / "last.pt")

        if improved:
            torch.save(ck, out / "best.pt")

        print(
            f"VAL epoch={epoch+1} "
            + " ".join(
                f"{k}={v:.5f}"
                for k, v in metrics.items()
            )
            + f" best={best:.5f}",
            flush=True,
        )

    print("V7 FINISHED", flush=True)


if __name__ == "__main__":
    main()
