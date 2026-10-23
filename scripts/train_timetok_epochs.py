"""Full-pass SHHS1 training with deterministic epoch shuffling."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from train_timetok_physio import (
    OneModality,
    training_stats,
    validate,
)
from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio import (
    LEVELS,
    TimeTokPhysio,
)


class EpochShuffle(Sampler):
    """One deterministic permutation per epoch, with resume."""

    def __init__(self, count, seed, epoch, start_sample=0):
        self.count = count
        self.seed = seed
        self.epoch = epoch
        self.start_sample = start_sample

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch
        )
        order = torch.randperm(
            self.count, generator=generator
        )
        for index in order[self.start_sample:]:
            yield int(index)

    def __len__(self):
        return self.count - self.start_sample


def save(path, model, optimizer, scheduler,
         epoch, batches_done, step, best_val,
         mean, std, args, train_count, val_loss):

    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "batches_done": batches_done,
        "step": step,
        "best_val": best_val,
        "val_loss": val_loss,
        "normalization": {"mean": mean, "std": std},
        "modality": args.modality,
        "levels": LEVELS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_count": train_count,
        "seed": args.seed,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }

    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modality",
        choices=("eeg", "ecg", "resp"),
        default="eeg",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--val-every", type=int, default=10000)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", default=None)

    args = p.parse_args()

    if args.init_from and args.resume:
        p.error("Choose --init-from OR --resume")

    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0")

    print("Loading SHHS1...", flush=True)

    train_source = PreprocessedPSGDataset(split="train")
    val_source = PreprocessedPSGDataset(split="val")

    train_ds = OneModality(train_source, args.modality)
    val_ds = OneModality(val_source, args.modality)

    mean, std = training_stats(train_source, args.modality)

    batches_per_epoch = math.ceil(
        len(train_ds) / args.batch_size
    )
    total_steps = batches_per_epoch * args.epochs

    print(
        f"Modality: {args.modality}\n"
        f"Train windows: {len(train_ds):,}\n"
        f"Val windows: {len(val_ds):,}\n"
        f"Steps per epoch: {batches_per_epoch:,}\n"
        f"Total epochs: {args.epochs}",
        flush=True,
    )

    model = TimeTokPhysio().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    def make_scheduler():
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                0.1 + 0.9 * 0.5 * (
                    1 + math.cos(
                        math.pi
                        * min(step, total_steps)
                        / total_steps
                    )
                )
            ),
        )

    scheduler = make_scheduler()

    epoch_start = 0
    batch_start = 0
    global_step = 0
    best = float("inf")

    # Start a new epoch-based phase from previous training.
    if args.init_from:
        old = torch.load(
            args.init_from,
            map_location="cpu",
            weights_only=False,
        )

        if old["modality"] != args.modality:
            raise ValueError("Modality mismatch")

        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])

        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["initial_lr"] = args.lr

        scheduler = make_scheduler()

        mean = old["normalization"]["mean"]
        std = old["normalization"]["std"]

        print(
            f"Warm start from previous step {old['step']}. "
            "Beginning epoch 1.",
            flush=True,
        )

    # Exact data-order resume from an epoch-based checkpoint.
    if args.resume:
        saved = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        expected = {
            "modality": args.modality,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_count": len(train_ds),
            "seed": args.seed,
        }

        for field, value in expected.items():
            if saved[field] != value:
                raise ValueError(
                    f"Checkpoint mismatch: {field}"
                )

        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])

        epoch_start = saved["epoch"]
        batch_start = saved["batches_done"]
        global_step = saved["step"]
        best = saved["best_val"]

        mean = saved["normalization"]["mean"]
        std = saved["normalization"]["std"]

        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])

        print(
            f"Resuming epoch {epoch_start + 1}, "
            f"batch {batch_start}",
            flush=True,
        )

    indices = torch.linspace(
        0,
        len(val_ds) - 1,
        min(args.val_samples, len(val_ds)),
    ).long().tolist()

    val_loader = DataLoader(
        Subset(val_ds, indices),
        batch_size=args.batch_size,
        num_workers=min(2, args.workers),
        pin_memory=True,
    )

    output = (
        Path("outputs/checkpoints")
        / f"timetok_epochs_{args.modality}"
    )

    def evaluate_and_save(next_epoch, next_batch, step):
        nonlocal best

        total, fm, aux = validate(
            model,
            val_loader,
            mean,
            std,
            device,
            args.aux_weight,
            True,
        )

        print(
            f"VAL | epoch={next_epoch + 1} "
            f"| step={step:,} "
            f"| total={total:.4f} "
            f"| fm={fm:.4f} "
            f"| aux={aux:.4f}",
            flush=True,
        )

        improved = total < best

        if improved:
            best = total

        save(
            output / "last.pt",
            model, optimizer, scheduler,
            next_epoch, next_batch, step,
            best, mean, std, args,
            len(train_ds), total,
        )

        if improved:
            save(
                output / "best.pt",
                model, optimizer, scheduler,
                next_epoch, next_batch, step,
                best, mean, std, args,
                len(train_ds), total,
            )
            print("Saved BEST checkpoint", flush=True)

    for epoch in range(epoch_start, args.epochs):

        consumed = (
            batch_start if epoch == epoch_start else 0
        )

        sampler = EpochShuffle(
            len(train_ds),
            args.seed,
            epoch,
            start_sample=consumed * args.batch_size,
        )

        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

        model.train()

        print(
            f"\nEPOCH {epoch + 1}/{args.epochs}",
            flush=True,
        )

        for batch_index, raw in enumerate(
            loader,
            start=consumed + 1,
        ):
            x = (
                (
                    raw.to(device, non_blocking=True)
                    - mean
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
            ):
                loss, metrics = model(
                    x,
                    keep,
                    aux_weight=args.aux_weight,
                )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Nonfinite loss at step {global_step + 1}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()
            scheduler.step()

            global_step += 1

            if (
                global_step % args.log_every == 0
                or global_step == 1
            ):
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"batch={batch_index:,}/{batches_per_epoch:,} "
                    f"step={global_step:,} "
                    f"fm={metrics['fm'].item():.4f} "
                    f"aux={metrics['aux'].item():.4f} "
                    f"unique={metrics['unique']}",
                    flush=True,
                )

            if (
                global_step % args.val_every == 0
                and batch_index < batches_per_epoch
            ):
                evaluate_and_save(
                    epoch, batch_index, global_step
                )

        # End of the full epoch.
        evaluate_and_save(epoch + 1, 0, global_step)
        batch_start = 0

        print(
            f"EPOCH {epoch + 1} COMPLETED",
            flush=True,
        )

    print("ALL EPOCHS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
