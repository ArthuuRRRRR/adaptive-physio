"""Train TimeTok-inspired tokenizer on SHHS1."""

import argparse
import math
import os
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Subset,
)

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio import (
    LEVELS,
    TimeTokPhysio,
)


class OneModality(Dataset):
    def __init__(self, source, modality):
        self.source = source
        self.modality = modality

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        i = bisect_right(
            self.source.cumulative, index
        )
        prev = (
            self.source.cumulative[i - 1]
            if i else 0
        )
        record = self.source.records[i]
        position = (
            record["positions"][self.modality]
            + index - prev
        )

        x = np.array(
            self.source._get_mmap(
                self.modality
            )[position],
            copy=True,
        )
        if not np.isfinite(x).all():
            raise ValueError(
                f"Nonfinite signal at index {index}"
            )
        return torch.from_numpy(x).unsqueeze(0)


def training_stats(source, modality):
    info = source.metadata[modality]["sig_info"]
    count = 0
    total = 0.0
    total_squared = 0.0

    for record in source.records:
        stats = info[
            record["recording"]
        ]["stats"]

        count += record["windows"] * 3000
        total += float(stats["sum"])
        total_squared += float(stats["sumsq"])

    mean = total / count
    variance = (
        total_squared / count - mean ** 2
    )
    if (
        not math.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError(
            "Invalid normalization statistics"
        )
    return mean, math.sqrt(variance)


def save_checkpoint(
    path, model, optimizer, scheduler,
    step, args, mean, std, best_val, val_loss
):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    tmp = path.with_suffix(".tmp")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "modality": args.modality,
        "normalization": {
            "mean": mean,
            "std": std,
        },
        "levels": LEVELS,
        "model_config": {
            "length": 3000,
            "patch": 25,
            "dim": 256,
        },
        "aux_weight": args.aux_weight,
        "best_val": best_val,
        "val_loss": val_loss,
    }, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(
    model, loader, mean, std,
    device, aux_weight, amp
):
    model.eval()
    totals = torch.zeros(
        3, dtype=torch.float64
    )
    n = 0

    with torch.random.fork_rng(
        devices=[0]
    ):
        torch.manual_seed(12345)

        for batch in loader:
            x = (
                (
                    batch.to(
                        device, non_blocking=True
                    ) - mean
                ) / std
            ).clamp(-8, 8)

            for level in LEVELS:
                keep = torch.full(
                    (len(x),),
                    level,
                    dtype=torch.long,
                    device=device,
                )

                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=amp,
                ):
                    loss, metrics = model(
                        x,
                        keep,
                        aux_weight=aux_weight,
                    )

                totals += torch.tensor([
                    loss.item(),
                    metrics["fm"].item(),
                    metrics["aux"].item(),
                ], dtype=torch.float64) * len(x)

                n += len(x)

    model.train()
    return (totals / n).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modality",
        choices=("eeg", "ecg", "resp"),
        default="eeg",
    )
    p.add_argument(
        "--steps", type=int, default=200000
    )
    p.add_argument(
        "--batch-size", type=int, default=8
    )
    p.add_argument(
        "--workers", type=int, default=4
    )
    p.add_argument(
        "--lr", type=float, default=2e-4
    )
    p.add_argument(
        "--aux-weight",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--warmup", type=int, default=1000
    )
    p.add_argument(
        "--val-every", type=int, default=2000
    )
    p.add_argument(
        "--val-samples", type=int, default=128
    )
    p.add_argument(
        "--log-every", type=int, default=100
    )
    p.add_argument("--resume", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda:0")
    amp = not args.no_amp

    print(
        "Loading SHHS1 train + val...",
        flush=True,
    )

    train_source = PreprocessedPSGDataset(
        split="train"
    )
    val_source = PreprocessedPSGDataset(
        split="val"
    )

    train_data = OneModality(
        train_source, args.modality
    )
    val_data = OneModality(
        val_source, args.modality
    )

    mean, std = training_stats(
        train_source, args.modality
    )

    print(
        f"Modality: {args.modality} "
        f"| train={len(train_data):,} "
        f"| val={len(val_data):,}",
        flush=True,
    )
    print(
        f"Normalization: "
        f"mean={mean:.6f}, std={std:.6f}",
        flush=True,
    )

    model = TimeTokPhysio().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    def lr_factor(s):
        if s < args.warmup:
            return max(
                1e-3,
                (s + 1) / max(1, args.warmup),
            )
        progress = min(
            1.0,
            (s - args.warmup)
            / max(1, args.steps - args.warmup),
        )
        return (
            0.05 + 0.95 * 0.5
            * (1 + math.cos(math.pi * progress))
        )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_factor
        )
    )

    out = (
        Path("outputs/checkpoints")
        / f"timetok_physio_{args.modality}"
    )

    start = 0
    best = float("inf")

    if args.resume:
        saved = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        if (
            saved["modality"] != args.modality
            or tuple(saved["levels"]) != LEVELS
        ):
            raise ValueError(
                "Checkpoint modality mismatch"
            )

        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(
            saved["optimizer"]
        )
        scheduler.load_state_dict(
            saved["scheduler"]
        )

        mean = saved["normalization"]["mean"]
        std = saved["normalization"]["std"]
        start = int(saved["step"])
        best = float(saved["best_val"])

        print(
            f"Resumed step {start:,}",
            flush=True,
        )

    if start >= args.steps:
        print("Training already completed")
        return

    sampler = RandomSampler(
        train_data,
        replacement=True,
        num_samples=(
            (args.steps - start)
            * args.batch_size
        ),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    positions = torch.linspace(
        0,
        len(val_data) - 1,
        min(args.val_samples, len(val_data)),
    ).long().tolist()

    val_loader = DataLoader(
        Subset(val_data, positions),
        batch_size=args.batch_size,
        num_workers=min(args.workers, 2),
        pin_memory=True,
    )

    print(
        f"TRAINING START | "
        f"steps={args.steps} "
        f"| batch={args.batch_size} "
        f"| levels={LEVELS} "
        f"| aux={args.aux_weight}",
        flush=True,
    )

    model.train()

    for step, batch in enumerate(
        train_loader, start=start + 1
    ):
        x = (
            (
                batch.to(
                    device, non_blocking=True
                ) - mean
            ) / std
        ).clamp(-8, 8)

        keep = torch.tensor(
            LEVELS, device=device
        )[torch.randint(
            0,
            len(LEVELS),
            (len(x),),
            device=device,
        )]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            loss, metrics = model(
                x,
                keep,
                aux_weight=args.aux_weight,
            )

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        scheduler.step()

        if (
            step == start + 1
            or step % args.log_every == 0
        ):
            print(
                f"step {step:7d}/{args.steps} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"fm={metrics['fm'].item():.4f} "
                f"aux={metrics['aux'].item():.4f} "
                f"unique={metrics['unique']}",
                flush=True,
            )

        if (
            step % args.val_every == 0
            or step == args.steps
        ):
            total, fm, aux = validate(
                model,
                val_loader,
                mean,
                std,
                device,
                args.aux_weight,
                amp,
            )

            print(
                f"VAL step={step} "
                f"total={total:.4f} "
                f"fm={fm:.4f} "
                f"aux={aux:.4f}",
                flush=True,
            )

            new_best = total < best
            if new_best:
                best = total

            save_checkpoint(
                out / "last.pt",
                model, optimizer, scheduler,
                step, args, mean, std,
                best, total,
            )

            if new_best:
                save_checkpoint(
                    out / "best.pt",
                    model, optimizer, scheduler,
                    step, args, mean, std,
                    best, total,
                )
                print(
                    "Saved BEST checkpoint",
                    flush=True,
                )

    print(
        "TRAINING COMPLETED",
        flush=True,
    )


if __name__ == "__main__":
    main()
