"""Supervised sequential EEG refinement with frozen V3."""

import argparse
import csv
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio_v3 import (
    PhysioTokenizerV3,
    reconstruction_stats,
)
from adaptive_physio.models.temporal_agent_v3 import (
    TemporalRefinementAgent,
)
from train_timetok_physio_v2 import (
    ChannelDataset,
    EpochPermutation,
)
from evaluate_timetok_v3_adaptive import decode_mixed


SCALE = 10.0
BUDGETS = (16, 32, 48, 64, 80, 96, 112, 128)


def regional_mae(pred, real):
    p = pred.float()
    r = real.float()
    size = r.shape[-1]

    return torch.stack([
        (
            p[
                ...,
                round(i * size / 16):
                round((i + 1) * size / 16)
            ]
            -
            r[
                ...,
                round(i * size / 16):
                round((i + 1) * size / 16)
            ]
        ).abs().mean(dim=(1, 2))
        for i in range(16)
    ], dim=1)


@torch.no_grad()
def observe_and_teach(backbone, x):
    """
    Build target-informed local gain estimates.

    Target gain:
        improvement in regional MAE
        divided by additional token cost.
    """

    with torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        codes, _ = backbone.tokenize(x)

        y16 = backbone.reconstructor(
            codes, 16
        )
        y64 = backbone.reconstructor(
            codes, 64
        )
        y128 = backbone.reconstructor(
            codes, 128
        )

    e16 = regional_mae(y16, x)
    e64 = regional_mae(y64, x)
    e128 = regional_mae(y128, x)

    gain_mid = (e16 - e64) / 3
    gain_fine = (e64 - e128) / 4

    return (
        codes.float().detach(),
        gain_mid,
        gain_fine,
        (y16, y64, y128),
    )


def sample_states(batch, device):
    """
    Sample different legal observation states
    so the controller learns to act sequentially.
    """
    u = torch.rand(
        batch, 16, device=device
    )

    return torch.where(
        u < 0.45,
        0,
        torch.where(u < 0.85, 1, 2),
    ).long()


def supervised_loss(
    agent, codes, gain_mid, gain_fine, levels
):
    output = agent(codes, levels)

    pred = torch.where(
        levels == 0,
        output[..., 0],
        output[..., 1],
    )

    target = torch.where(
        levels == 0,
        gain_mid,
        gain_fine,
    ) * SCALE

    active = levels < 2

    loss = F.smooth_l1_loss(
        pred[active],
        target[active],
        beta=0.2,
    )

    with torch.no_grad():
        error = (
            pred[active] - target[active]
        ).abs().mean()

    return loss, error


@torch.no_grad()
def rollout(
    agent,
    codes,
    budget,
    policy,
    gain_mid=None,
    gain_fine=None,
    rng=None,
    allow_stop=False,
):
    """
    Sequentially select local refinement actions.

    Only the oracle reads ground-truth-derived gains.
    The learned agent reads revealed tokens only.
    """

    b = codes.shape[0]
    device = codes.device

    levels = torch.zeros(
        b, 16,
        dtype=torch.long,
        device=device,
    )

    used = torch.full(
        (b,), 16,
        dtype=torch.long,
        device=device,
    )

    for _ in range(32):

        if policy == "agent":
            output = agent(codes, levels)

            scores = torch.where(
                levels == 0,
                output[..., 0],
                output[..., 1],
            )

        elif policy == "oracle":
            if gain_mid is None or gain_fine is None:
                raise ValueError(
                    "Oracle requires teacher gains"
                )

            scores = torch.where(
                levels == 0,
                gain_mid,
                gain_fine,
            )

        elif policy == "random":
            scores = torch.rand(
                (b, 16),
                generator=rng,
                device=device,
            )

        else:
            raise ValueError(policy)

        # Coarse -> mid costs 3 extra tokens.
        # Mid -> fine costs 4 extra tokens.
        costs = torch.where(
            levels == 0, 3, 4
        )

        eligible = (
            (levels < 2)
            &
            (
                used[:, None] + costs
                <= budget
            )
        )

        scores = scores.masked_fill(
            ~eligible,
            -torch.inf,
        )

        best_score, region = scores.max(
            dim=1
        )

        selected = torch.isfinite(best_score)

        # Optional learned STOP decision.
        if allow_stop and policy != "random":
            selected = (
                selected & (best_score > 0)
            )

        if not selected.any().item():
            break

        rows = torch.where(selected)[0]
        cols = region[selected]

        added = costs[rows, cols]

        levels[rows, cols] += 1
        used[rows] += added

    return levels, used


def data_loader(
    split,
    modality,
    windows,
    batch_size,
    workers,
    seed,
    epoch=0,
    start_batch=0,
):
    source = PreprocessedPSGDataset(
        split=split
    )

    dataset = ChannelDataset(
        source, modality
    )

    windows = min(windows, len(dataset))

    if split == "train":
        generator = (
            torch.Generator()
            .manual_seed(seed + 999)
        )

        indices = torch.randperm(
            len(dataset),
            generator=generator,
        )[:windows].tolist()

        subset = Subset(dataset, indices)

        sampler = EpochPermutation(
            len(subset),
            seed,
            epoch,
            start_batch,
            batch_size,
        )

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
        )

    else:
        indices = torch.linspace(
            0,
            len(dataset) - 1,
            windows,
        ).long().tolist()

        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
        )

    return loader, windows


def load_backbone(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("modality") != "eeg":
        raise ValueError(
            "This experiment is EEG-only"
        )

    model = PhysioTokenizerV3().to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval().requires_grad_(False)

    return model, checkpoint


def normalize(raw, norm, device):
    return (
        (
            raw.to(
                device,
                non_blocking=True,
            )
            - norm["mean"]
        ) / norm["std"]
    ).clamp(-8, 8)


@torch.no_grad()
def validate_regression(
    backbone, agent, loader, norm, device
):
    agent.eval()

    total_loss = 0.0
    total_error = 0.0
    count = 0

    # Use reproducible validation states.
    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()]
    ):
        torch.manual_seed(12345)

        for raw in loader:
            x = normalize(raw, norm, device)

            codes, gm, gf, _ = (
                observe_and_teach(backbone, x)
            )

            states = sample_states(
                len(x), device
            )

            loss, error = supervised_loss(
                agent,
                codes,
                gm,
                gf,
                states,
            )

            total_loss += (
                loss.item() * len(x)
            )
            total_error += (
                error.item() * len(x)
            )
            count += len(x)

    agent.train()

    return (
        total_loss / count,
        total_error / count,
    )


def atomic_save(path, state):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(".tmp")

    torch.save(state, temp)
    os.replace(temp, path)


def train(args, device):
    backbone, back_ckpt = load_backbone(
        args.backbone, device
    )

    norm = back_ckpt["normalization"]

    agent = TemporalRefinementAgent().to(
        device
    )

    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    per_epoch = math.ceil(
        args.train_windows / args.batch_size
    )

    all_steps = per_epoch * args.epochs

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=all_steps,
            eta_min=args.lr * 0.05,
        )
    )

    output = Path(
        "outputs/checkpoints/"
        "timetok_v3_agent_eeg"
    )

    start_epoch = 0
    start_batch = 0
    step = 0
    best = float("inf")

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        expected = (
            ("epochs", args.epochs),
            ("train_windows", args.train_windows),
            ("batch_size", args.batch_size),
            ("seed", args.seed),
            ("backbone_step", back_ckpt["step"]),
        )

        for name, required in expected:
            if checkpoint[name] != required:
                raise ValueError(
                    f"Resume mismatch: {name}"
                )

        agent.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch"]

        step = checkpoint["step"]
        best = checkpoint["best_val"]

        print(
            f"RESUME epoch={start_epoch+1} "
            f"batch={start_batch+1}",
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
        "AGENT SUPERVISED TRAINING"
        f"\nBackbone V3 step: "
        f"{back_ckpt['step']:,}"
        f"\nTrain windows: "
        f"{args.train_windows:,}"
        f"\nEpochs: {args.epochs}"
        f"\nSteps per epoch: {per_epoch:,}",
        flush=True,
    )

    def validate_save(epoch, batch):
        nonlocal best

        val, error = validate_regression(
            backbone,
            agent,
            val_loader,
            norm,
            device,
        )

        print(
            f"VAL agent | step={step:,} "
            f"regression={val:.6f} "
            f"gain_error_scaled={error:.4f}",
            flush=True,
        )

        improved = val < best
        best = min(best, val)

        state = {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "step": step,
            "best_val": best,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_windows": args.train_windows,
            "seed": args.seed,
            "backbone_path": args.backbone,
            "backbone_step": back_ckpt["step"],
        }

        atomic_save(
            output / "last.pt", state
        )

        if improved:
            atomic_save(
                output / "best.pt", state
            )
            print(
                "Saved BEST | "
                f"regression={val:.6f}",
                flush=True,
            )

    for epoch in range(
        start_epoch, args.epochs
    ):
        consumed = (
            start_batch
            if epoch == start_epoch
            else 0
        )

        loader, _ = data_loader(
            "train",
            "eeg",
            args.train_windows,
            args.batch_size,
            args.workers,
            args.seed,
            epoch,
            consumed,
        )

        agent.train()

        for batch_idx, raw in enumerate(
            loader,
            start=consumed + 1,
        ):
            x = normalize(
                raw, norm, device
            )

            codes, gm, gf, _ = (
                observe_and_teach(backbone, x)
            )

            states = sample_states(
                len(x), device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, error = supervised_loss(
                agent,
                codes,
                gm,
                gf,
                states,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Nonfinite agent loss"
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
                    f"agent epoch="
                    f"{epoch+1}/{args.epochs} "
                    f"step={step:,} "
                    f"loss={loss.item():.6f} "
                    f"gain_err={error.item():.4f}",
                    flush=True,
                )

            if (
                step % args.val_every == 0
                and batch_idx < per_epoch
            ):
                validate_save(
                    epoch, batch_idx
                )

        validate_save(epoch + 1, 0)

    print(
        "AGENT TRAINING FINISHED",
        flush=True,
    )


@torch.inference_mode()
def evaluate(args, device):
    if not args.agent_checkpoint:
        raise ValueError(
            "Use --agent-checkpoint"
        )

    backbone, back_ckpt = load_backbone(
        args.backbone, device
    )

    checkpoint = torch.load(
        args.agent_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint["backbone_step"]
        != back_ckpt["step"]
    ):
        raise ValueError(
            "Agent and V3 backbone "
            "checkpoint steps differ"
        )

    agent = TemporalRefinementAgent().to(
        device
    )

    agent.load_state_dict(
        checkpoint["model"]
    )
    agent.eval()

    loader, n_windows = data_loader(
        "val",
        "eeg",
        args.eval_windows,
        args.batch_size,
        min(args.workers, 2),
        args.seed,
    )

    norm = back_ckpt["normalization"]

    rng = torch.Generator(
        device=device
    ).manual_seed(
        args.seed + 2027
    )

    result = {
        (policy, budget): [0., 0., 0., 0]
        for policy in (
            "random", "agent", "oracle"
        )
        for budget in BUDGETS
    }

    uniform = {
        k: [0., 0., 0]
        for k in (16, 64, 128)
    }

    processed = 0

    for raw in loader:
        x = normalize(
            raw, norm, device
        )

        codes, gm, gf, recons = (
            observe_and_teach(backbone, x)
        )

        for k, pred in zip(
            (16, 64, 128), recons
        ):
            mae, _, corr = (
                reconstruction_stats(pred, x)
            )

            uniform[k][0] += mae.sum().item()
            uniform[k][1] += corr.sum().item()
            uniform[k][2] += len(x)

        for budget in BUDGETS:
            for policy in (
                "random", "agent", "oracle"
            ):
                levels, used = rollout(
                    agent,
                    codes,
                    budget,
                    policy,
                    gm,
                    gf,
                    rng,
                    allow_stop=args.allow_stop,
                )

                reconstruction = decode_mixed(
                    backbone.reconstructor,
                    codes,
                    levels,
                )

                mae, _, corr = (
                    reconstruction_stats(
                        reconstruction, x
                    )
                )

                record = result[
                    policy, budget
                ]

                record[0] += mae.sum().item()
                record[1] += corr.sum().item()
                record[2] += used.sum().item()
                record[3] += len(x)

        processed += len(x)

        if (
            processed % 128 == 0
            or processed == n_windows
        ):
            print(
                f"Evaluated "
                f"{processed}/{n_windows}",
                flush=True,
            )

    print(
        "\n=== UNIFORM V3 BASELINES ==="
    )

    for k, (mae, corr, n) in uniform.items():
        print(
            f"{k:3d} tokens | "
            f"MAE={mae/n:.4f} "
            f"Corr={corr/n:.4f}"
        )

    print(
        "\n=== CONTROLLER COMPARISON ==="
    )
    print(
        "Policy  | Budget | Used | MAE | Corr"
    )

    rows = []

    for budget in BUDGETS:
        for policy in (
            "random", "agent", "oracle"
        ):
            mae, corr, used, n = result[
                policy, budget
            ]

            row = {
                "policy": policy,
                "budget": budget,
                "avg_used": used / n,
                "mae": mae / n,
                "corr": corr / n,
            }

            rows.append(row)

            print(
                f"{policy:7} | "
                f"{budget:3d} | "
                f"{used/n:5.1f} | "
                f"{mae/n:.4f} | "
                f"{corr/n:.4f}"
            )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output)
    print(
        "Oracle uses ground truth; "
        "agent uses revealed tokens only."
    )
    print(
        "Token reduction is not yet "
        "a measured FLOPs reduction."
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        required=True,
        choices=("train", "eval"),
    )
    p.add_argument(
        "--backbone",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/best.pt"
        ),
    )
    p.add_argument("--agent-checkpoint")
    p.add_argument(
        "--epochs", type=int, default=5
    )
    p.add_argument(
        "--train-windows",
        type=int, default=60000
    )
    p.add_argument(
        "--val-windows",
        type=int, default=1024
    )
    p.add_argument(
        "--eval-windows",
        type=int, default=1024
    )
    p.add_argument(
        "--batch-size", type=int, default=16
    )
    p.add_argument(
        "--workers", type=int, default=4
    )
    p.add_argument(
        "--lr", type=float, default=3e-4
    )
    p.add_argument(
        "--val-every", type=int, default=1000
    )
    p.add_argument(
        "--log-every", type=int, default=200
    )
    p.add_argument(
        "--seed", type=int, default=42
    )
    p.add_argument("--resume")
    p.add_argument(
        "--allow-stop", action="store_true"
    )
    p.add_argument(
        "--output",
        default=(
            "outputs/evaluation/timetok_v3/"
            "agent_comparison.csv"
        ),
    )

    args = p.parse_args()

    for count in (
        args.epochs,
        args.train_windows,
        args.val_windows,
        args.eval_windows,
        args.batch_size,
        args.val_every,
        args.log_every,
    ):
        if count < 1:
            p.error(
                "Counts must be positive"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required"
        )

    device = torch.device("cuda:0")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(
        "high"
    )

    if args.mode == "train":
        train(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
