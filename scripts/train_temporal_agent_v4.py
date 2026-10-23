"""Cost-aware adaptive EEG agent trained on mixed-state marginal gains.

Backbone V3 is frozen.
The controller receives only revealed FSQ tokens.
"""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_physio.models.temporal_agent_v3 import (
    TemporalRefinementAgent,
)
from adaptive_physio.models.timetok_physio_v3 import (
    reconstruction_stats,
)
from train_temporal_agent_v3 import (
    load_backbone,
    data_loader,
    normalize,
)
from evaluate_timetok_v3_adaptive import decode_mixed


# Neural-network units per unit of reconstruction gain/token.
SCALE = 100.0


def quality_error(pred, truth):
    """Per-window L1 + derivative L1 + multi-resolution STFT."""
    p = pred.float()
    x = truth.float()

    mae = (p - x).abs().mean((1, 2))

    diff = (
        (p[..., 1:] - p[..., :-1])
        - (x[..., 1:] - x[..., :-1])
    ).abs().mean((1, 2))

    spectral = torch.zeros_like(mae)

    with torch.autocast(
        device_type=p.device.type,
        enabled=False,
    ):
        for fft in (64, 256, 512):
            window = torch.hann_window(
                fft, device=p.device
            )

            ps = torch.stft(
                p[:, 0],
                n_fft=fft,
                hop_length=fft // 4,
                window=window,
                return_complex=True,
            ).abs()

            xs = torch.stft(
                x[:, 0],
                n_fft=fft,
                hop_length=fft // 4,
                window=window,
                return_complex=True,
            ).abs()

            spectral += (
                torch.log1p(ps)
                - torch.log1p(xs)
            ).abs().mean((1, 2)) / 3.0

    return mae + 0.1 * diff + 0.25 * spectral


def random_states(batch, device):
    """Sample legal mixed observation states."""
    u = torch.rand(batch, 16, device=device)

    group = torch.randint(
        0, 4, (batch, 1), device=device
    )

    states = torch.where(
        group == 0,
        0,
        torch.where(
            group == 1,
            (u > 0.55).long(),
            torch.where(
                group == 2,
                torch.where(
                    u < 0.25,
                    0,
                    torch.where(u < 0.75, 1, 2),
                ),
                torch.where(
                    u < 0.12,
                    0,
                    torch.where(u < 0.46, 1, 2),
                ),
            ),
        ),
    ).long()

    # Ensure at least one valid refinement remains.
    fully_fine = (states == 2).all(dim=1)
    states[fully_fine, 0] = 1

    return states


@torch.no_grad()
def targets_mixed(backbone, codes, x, states):
    """
    Measure actual reconstruction-quality improvement
    for two sampled legal actions in the CURRENT mixed state.
    """
    b = len(x)

    eligible = (states < 2).float()

    actions = torch.multinomial(
        eligible,
        num_samples=2,
        replacement=True,
    )

    types = states.gather(1, actions)

    costs = torch.where(
        types == 0, 3, 4
    ).float()

    next_states = (
        states[:, None, :]
        .expand(-1, 2, -1)
        .clone()
    )

    next_states.scatter_(
        2,
        actions[..., None],
        types[..., None] + 1,
    )

    base = decode_mixed(
        backbone.reconstructor,
        codes,
        states,
    )

    changed = decode_mixed(
        backbone.reconstructor,
        codes.repeat_interleave(2, dim=0),
        next_states.reshape(b * 2, 16),
    )

    loss_before = quality_error(base, x)

    loss_after = quality_error(
        changed,
        x.repeat_interleave(2, dim=0),
    ).reshape(b, 2)

    gain_per_token = (
        loss_before[:, None] - loss_after
    ) / costs

    return actions, types, gain_per_token * SCALE


def predicted_action_gains(
    agent, codes, states, actions, types
):
    all_scores = agent(codes, states)

    chosen = all_scores.gather(
        1,
        actions[..., None].expand(-1, -1, 2),
    )

    return chosen.gather(
        2,
        types[..., None],
    ).squeeze(-1)


def supervised_step(agent, backbone, x):
    with torch.no_grad():
        with torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            codes, _ = backbone.tokenize(x)

        codes = codes.float()
        states = random_states(
            len(x), x.device
        )

        actions, types, targets = targets_mixed(
            backbone, codes, x, states
        )

    predictions = predicted_action_gains(
        agent,
        codes,
        states,
        actions,
        types,
    )

    regression = F.smooth_l1_loss(
        predictions, targets, beta=0.15
    )

    ranking = F.smooth_l1_loss(
        predictions[:, 0] - predictions[:, 1],
        targets[:, 0] - targets[:, 1],
        beta=0.15,
    )

    loss = regression + 0.2 * ranking

    error = (
        predictions - targets
    ).abs().mean()

    return loss, error


@torch.no_grad()
def rollout_lambdas(agent, codes, lambdas, maximum=128):
    """Sequential STOP policy for multiple token-cost penalties."""
    b = codes.shape[0]
    n = len(lambdas)

    repeated = codes.repeat_interleave(
        n, dim=0
    )

    cutoff = torch.tensor(
        lambdas,
        device=codes.device,
        dtype=torch.float32,
    ).repeat(b) * SCALE

    states = torch.zeros(
        b * n, 16,
        device=codes.device,
        dtype=torch.long,
    )

    used = torch.full(
        (b * n,), 16,
        device=codes.device,
        dtype=torch.long,
    )

    for _ in range(32):
        output = agent(repeated, states)

        scores = torch.where(
            states == 0,
            output[..., 0],
            output[..., 1],
        )

        costs = torch.where(
            states == 0, 3, 4
        )

        eligible = (
            (states < 2)
            & (used[:, None] + costs <= maximum)
        )

        scores = scores.masked_fill(
            ~eligible, -torch.inf
        )

        best, region = scores.max(dim=1)

        selected = (
            torch.isfinite(best)
            & (best > cutoff)
        )

        if not selected.any().item():
            break

        rows = torch.where(selected)[0]
        cols = region[rows]

        used[rows] += costs[rows, cols]
        states[rows, cols] += 1

    return repeated, states, used


def load_agent(path, device):
    agent = TemporalRefinementAgent().to(device)

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    agent.load_state_dict(ckpt["model"])

    return agent, ckpt


def atomic_save(path, obj):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


@torch.no_grad()
def validate(agent, backbone, loader, norm, device):
    agent.eval()

    total = 0.0
    absolute = 0.0
    n = 0

    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()]
    ):
        torch.manual_seed(1729)

        for raw in loader:
            x = normalize(
                raw, norm, device
            )

            with torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                codes, _ = backbone.tokenize(x)

            codes = codes.float()

            states = random_states(
                len(x), device
            )

            acts, types, target = targets_mixed(
                backbone, codes, x, states
            )

            pred = predicted_action_gains(
                agent, codes, states,
                acts, types
            )

            loss = F.smooth_l1_loss(
                pred, target, beta=0.15
            )

            total += float(loss) * len(x)

            absolute += float(
                (pred - target).abs().mean()
            ) * len(x)

            n += len(x)

    agent.train()

    return total / n, absolute / n


def run_train(args, device):
    backbone, bc = load_backbone(
        args.backbone, device
    )

    agent = TemporalRefinementAgent().to(device)

    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    n_batches = math.ceil(
        args.train_windows / args.batch_size
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_batches * args.epochs,
            eta_min=args.lr * 0.05,
        )
    )

    output = Path(
        "outputs/checkpoints/"
        "timetok_v3_agent_cost_eeg"
    )

    epoch_start = 0
    batch_start = 0
    step = 0
    best = float("inf")

    if args.resume:
        ck = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        for key, expected in [
            ("backbone_step", bc["step"]),
            ("epochs", args.epochs),
            ("train_windows", args.train_windows),
            ("batch_size", args.batch_size),
            ("seed", args.seed),
        ]:
            if ck[key] != expected:
                raise ValueError(
                    f"Resume mismatch: {key}"
                )

        agent.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])

        epoch_start = ck["epoch"]
        batch_start = ck["batch"]
        step = ck["step"]
        best = ck["best_val"]

        torch.set_rng_state(
            ck["rng_cpu"]
        )
        torch.cuda.set_rng_state(
            ck["rng_cuda"]
        )

        print(
            f"RESUME epoch={epoch_start+1}, "
            f"batch={batch_start+1}",
            flush=True,
        )

    val_loader, _ = data_loader(
        "val",
        "eeg",
        args.val_windows,
        args.batch_size,
        min(args.workers, 2),
        args.seed,
    )

    print(
        f"COST AGENT | V3 backbone step={bc['step']:,} "
        f"| epochs={args.epochs} "
        f"| steps/epoch={n_batches:,} "
        f"| gain scale={SCALE}",
        flush=True,
    )

    def checkpoint(epoch, batch):
        nonlocal best

        score, error = validate(
            agent, backbone, val_loader,
            bc["normalization"], device
        )

        improved = score < best
        best = min(best, score)

        ck = {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "step": step,
            "best_val": best,
            "backbone_step": bc["step"],
            "epochs": args.epochs,
            "train_windows": args.train_windows,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "gain_scale": SCALE,
            "quality": "MAE+0.1*dMAE+0.25*STFT",
        }

        atomic_save(
            output / "last.pt", ck
        )

        if improved:
            atomic_save(
                output / "best.pt", ck
            )

        print(
            f"VAL | step={step:,} "
            f"marginal_gain_loss={score:.6f} "
            f"gain_err={error:.5f} "
            f"best={best:.6f}",
            flush=True,
        )

    for epoch in range(
        epoch_start, args.epochs
    ):
        offset = (
            batch_start
            if epoch == epoch_start
            else 0
        )

        loader, _ = data_loader(
            "train",
            "eeg",
            args.train_windows,
            args.batch_size,
            args.workers,
            args.seed,
            epoch=epoch,
            start_batch=offset,
        )

        agent.train()

        for batch_i, raw in enumerate(
            loader, start=offset + 1
        ):
            x = normalize(
                raw,
                bc["normalization"],
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, error = supervised_step(
                agent, backbone, x
            )

            if not bool(
                torch.isfinite(loss).item()
            ):
                raise RuntimeError(
                    "Nonfinite loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), 1.0
            )

            optimizer.step()
            scheduler.step()
            step += 1

            if (
                step == 1
                or step % args.log_every == 0
            ):
                print(
                    f"epoch={epoch+1}/{args.epochs} "
                    f"step={step:,} "
                    f"loss={loss.item():.6f} "
                    f"gain_err={error.item():.5f}",
                    flush=True,
                )

            if (
                step % args.val_every == 0
                and batch_i < n_batches
            ):
                checkpoint(epoch, batch_i)

        checkpoint(epoch + 1, 0)
        batch_start = 0

    print(
        "COST AGENT TRAINING FINISHED",
        flush=True,
    )


@torch.inference_mode()
def run_eval(args, device):
    backbone, bc = load_backbone(
        args.backbone, device
    )

    agent, ac = load_agent(
        args.agent_checkpoint, device
    )

    if ac["backbone_step"] != bc["step"]:
        raise ValueError("Backbone mismatch")

    agent.eval()

    loader, n = data_loader(
        "val",
        "eeg",
        args.eval_windows,
        args.batch_size,
        min(args.workers, 2),
        args.seed,
    )

    k = len(args.lambdas)
    sums = torch.zeros(
        k, 4, dtype=torch.float64
    )
    counts = [[] for _ in range(k)]
    processed = 0

    for raw in loader:
        x = normalize(
            raw, bc["normalization"], device
        )

        with torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            codes, _ = backbone.tokenize(x)

        codes, states, used = rollout_lambdas(
            agent,
            codes.float(),
            args.lambdas,
            args.max_budget,
        )

        pred = decode_mixed(
            backbone.reconstructor,
            codes,
            states,
        )

        reference = x.repeat_interleave(
            k, dim=0
        )

        mae, _, corr = reconstruction_stats(
            pred, reference
        )

        b = len(x)

        mae = mae.view(b, k).cpu().double()
        corr = corr.view(b, k).cpu().double()
        tokens = used.view(b, k).cpu().double()

        sums[:, 0] += mae.sum(0)
        sums[:, 1] += corr.sum(0)
        sums[:, 2] += tokens.sum(0)
        sums[:, 3] += b

        for j in range(k):
            counts[j].extend(
                tokens[:, j].tolist()
            )

        processed += b

        if (
            processed % 128 == 0
            or processed == n
        ):
            print(
                f"Evaluated {processed}/{n}",
                flush=True,
            )

    output = Path(args.output_dir)
    output.mkdir(
        parents=True, exist_ok=True
    )

    rows = []

    print(
        "\nlambda  | Avg tokens | P10 | "
        "P90 | MAE | Corr"
    )

    for j, lam in enumerate(args.lambdas):
        row = {
            "lambda_cost": lam,
            "used": float(
                sums[j, 2] / sums[j, 3]
            ),
            "p10": float(
                np.percentile(counts[j], 10)
            ),
            "p90": float(
                np.percentile(counts[j], 90)
            ),
            "mae": float(
                sums[j, 0] / sums[j, 3]
            ),
            "corr": float(
                sums[j, 1] / sums[j, 3]
            ),
        }
        rows.append(row)

        print(
            f"{lam:7.5f} | "
            f"{row['used']:10.1f} | "
            f"{row['p10']:3.0f} | "
            f"{row['p90']:3.0f} | "
            f"{row['mae']:.4f} | "
            f"{row['corr']:.4f}"
        )

    csv_path = output / "cost_sweep.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8)
    )

    reference_path = Path(args.reference)

    if (
        reference_path.exists()
        and args.eval_windows == 2048
    ):
        with reference_path.open(newline="") as f:
            reference_rows = list(
                csv.DictReader(f)
            )

        for policy, color in (
            ("random", "#94a3b8"),
            ("agent", "#c77d23"),
            ("oracle", "#008577"),
        ):
            subset = sorted(
                [
                    r for r in reference_rows
                    if r["policy"] == policy
                ],
                key=lambda r: float(
                    r["avg_used"]
                ),
            )

            for ax, metric in zip(
                axes, ("corr", "mae")
            ):
                ax.plot(
                    [
                        float(r["avg_used"])
                        for r in subset
                    ],
                    [
                        float(r[metric])
                        for r in subset
                    ],
                    "--o",
                    markersize=3,
                    color=color,
                    label=policy,
                )

    for ax, metric in zip(
        axes, ("corr", "mae")
    ):
        ax.plot(
            [r["used"] for r in rows],
            [r[metric] for r in rows],
            "-o",
            linewidth=2.2,
            label="Cost-aware STOP",
        )

        ax.set_xlabel(
            "Mean selected tokens / 30 s window"
        )
        ax.set_ylabel(
            metric.upper() + " vs original EEG"
        )
        ax.grid(alpha=0.2)
        ax.legend()

    fig.suptitle(
        f"SHHS1 validation ({processed:,} windows) "
        "| learned mixed-state gains"
    )
    fig.tight_layout()

    fig.savefig(
        output / "cost_quality.png",
        dpi=170,
    )
    plt.close(fig)

    print(
        f"Saved: {csv_path}"
    )
    print(
        f"Saved: {output / 'cost_quality.png'}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=("train", "eval"),
        required=True,
    )
    p.add_argument(
        "--backbone",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
    )
    p.add_argument(
        "--agent-checkpoint",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_agent_cost_eeg/best.pt"
        ),
    )
    p.add_argument("--resume")
    p.add_argument(
        "--smoke-test",
        action="store_true",
    )

    p.add_argument(
        "--epochs", type=int, default=5
    )
    p.add_argument(
        "--train-windows",
        type=int, default=60000
    )
    p.add_argument(
        "--val-windows",
        type=int, default=512
    )
    p.add_argument(
        "--eval-windows",
        type=int, default=2048
    )
    p.add_argument(
        "--batch-size",
        type=int, default=16
    )
    p.add_argument(
        "--workers",
        type=int, default=4
    )
    p.add_argument(
        "--lr", type=float, default=3e-4
    )
    p.add_argument(
        "--val-every",
        type=int, default=1000
    )
    p.add_argument(
        "--log-every",
        type=int, default=200
    )
    p.add_argument(
        "--seed", type=int, default=42
    )
    p.add_argument(
        "--max-budget",
        type=int, default=128
    )
    p.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[
            0.0, 0.0001, 0.0002,
            0.0004, 0.0008, 0.0012,
            0.002, 0.004, 0.008, 0.016,
        ],
    )
    p.add_argument(
        "--reference",
        default=(
            "outputs/evaluation/timetok_v3/"
            "agent_comparison_2048.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        default=(
            "outputs/evaluation/timetok_v3/"
            "cost_agent"
        ),
    )

    args = p.parse_args()

    if any(
        value < 1 for value in (
            args.epochs,
            args.train_windows,
            args.batch_size,
            args.val_windows,
            args.eval_windows,
            args.val_every,
        )
    ):
        p.error("All counts must be positive")

    if not 16 <= args.max_budget <= 128:
        p.error(
            "max-budget must be in [16,128]"
        )

    if any(value < 0 for value in args.lambdas):
        p.error(
            "lambdas must be nonnegative"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.set_float32_matmul_precision(
        "high"
    )

    device = torch.device("cuda:0")

    if args.smoke_test:
        backbone, bc = load_backbone(
            args.backbone, device
        )

        loader, _ = data_loader(
            "train", "eeg",
            8, 4, 0, args.seed
        )

        x = normalize(
            next(iter(loader)),
            bc["normalization"],
            device,
        )

        agent = TemporalRefinementAgent().to(
            device
        )

        loss, error = supervised_step(
            agent, backbone, x
        )

        loss.backward()

        grad = next(
            agent.parameters()
        ).grad

        assert grad is not None
        assert torch.isfinite(grad).all()

        print(
            "SMOKE TEST PASSED | "
            f"loss={loss.item():.5f} "
            f"| gradient="
            f"{grad.abs().mean().item():.2e}"
        )
        return

    if args.mode == "train":
        run_train(args, device)
    else:
        run_eval(args, device)


if __name__ == "__main__":
    main()
