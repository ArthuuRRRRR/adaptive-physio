"""SHHS1 hierarchical temporal tokenizer V3 training."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio_v3 import (
    LEVELS,
    PhysioTokenizerV3,
    reconstruction_loss,
    reconstruction_stats,
)
from train_timetok_physio_v2 import (
    ChannelDataset,
    EpochPermutation,
    split_stats,
)


def prepare(raw, mean, std, device):
    return (
        (raw.to(device, non_blocking=True) - mean)
        / std
    ).clamp(-8, 8)


@torch.inference_mode()
def validate(model, loader, mean, std, device, amp):
    model.eval()

    metrics = {
        k: {
            "mae": 0.,
            "rmse": 0.,
            "corr": 0.,
            "stft": 0.,
            "ae": 0.,
        }
        for k in LEVELS
    }

    unique = {k: set() for k in LEVELS}
    n = 0
    baseline = 0.
    shuffled_mae = 0.

    for raw in loader:
        x = prepare(raw, mean, std, device)
        b = len(x)
        n += b

        baseline += x.abs().mean(
            dim=(1, 2)
        ).sum().item()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            codes, ids = model.tokenize(x)

            for keep in LEVELS:
                unique[keep].update(
                    ids[:, :keep].reshape(-1)
                    .cpu().tolist()
                )

                y = model.reconstructor(
                    codes, keep
                )

                loss, parts = reconstruction_loss(
                    y, x
                )

                mae, rmse, corr = reconstruction_stats(
                    y, x
                )

                metrics[keep]["mae"] += (
                    mae.sum().item()
                )
                metrics[keep]["rmse"] += (
                    rmse.sum().item()
                )
                metrics[keep]["corr"] += (
                    corr.sum().item()
                )
                metrics[keep]["stft"] += (
                    parts["spec"].item() * b
                )
                metrics[keep]["ae"] += (
                    loss.item() * b
                )

            if b > 1:
                wrong_codes = codes.roll(
                    1, dims=0
                )
                wrong_y = model.reconstructor(
                    wrong_codes, 128
                )
                wrong_mae, _, _ = reconstruction_stats(
                    wrong_y, x
                )
                shuffled_mae += wrong_mae.sum().item()

    model.train()

    result = {
        str(k): {
            name: value / n
            for name, value
            in metrics[k].items()
        }
        for k in LEVELS
    }

    result["unique"] = {
        str(k): len(unique[k])
        for k in LEVELS
    }
    result["zero_mae"] = baseline / n
    result["shuffled_mae_128"] = shuffled_mae / n

    return result


def save(
    path, model, optimizer, scheduler,
    args, epoch, next_batch,
    step, best, normalization, validation
):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    checkpoint = {
        "version": "v3-local-wavelet",
        "stage": "ae",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "modality": args.modality,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_train": args.n_train,
        "epoch": epoch,
        "next_batch": next_batch,
        "step": step,
        "best_val": best,
        "val": validation,
        "normalization": normalization,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "rng_numpy": np.random.get_state(),
    }

    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--modality",
        choices=("eeg", "ecg", "resp"),
        default="eeg",
    )
    p.add_argument(
        "--epochs", type=int, default=3
    )
    p.add_argument(
        "--batch-size", type=int, default=8
    )
    p.add_argument(
        "--workers", type=int, default=4
    )
    p.add_argument(
        "--lr", type=float, default=1e-4
    )
    p.add_argument(
        "--val-every", type=int, default=5000
    )
    p.add_argument(
        "--val-windows", type=int, default=256
    )
    p.add_argument(
        "--log-every", type=int, default=500
    )
    p.add_argument(
        "--seed", type=int, default=42
    )
    p.add_argument("--resume")
    p.add_argument(
        "--smoke-test", action="store_true"
    )
    p.add_argument(
        "--no-amp", action="store_true"
    )

    args = p.parse_args()

    for name in (
        "epochs", "batch_size",
        "val_every", "val_windows",
        "log_every",
    ):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required"
        )

    device = torch.device("cuda:0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    train_src = PreprocessedPSGDataset(
        split="train"
    )
    val_src = PreprocessedPSGDataset(
        split="val"
    )

    train_ds = ChannelDataset(
        train_src, args.modality
    )
    val_ds = ChannelDataset(
        val_src, args.modality
    )

    args.n_train = len(train_ds)

    mean, std = split_stats(
        train_src, args.modality
    )

    normalization = {
        "mean": mean,
        "std": std,
    }

    batches_per_epoch = math.ceil(
        len(train_ds) / args.batch_size
    )

    total_steps = (
        batches_per_epoch * args.epochs
    )

    val_indices = torch.linspace(
        0, len(val_ds) - 1,
        min(args.val_windows, len(val_ds))
    ).long().tolist()

    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=args.batch_size,
        num_workers=min(2, args.workers),
        pin_memory=True,
    )

    model = PhysioTokenizerV3().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    def lr_factor(step):
        warmup = min(
            500, max(1, total_steps // 10)
        )
        ramp = min(
            1.0, (step + 1) / warmup
        )

        cosine = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(
                math.pi
                * min(step, total_steps)
                / total_steps
            )
        )

        return ramp * cosine

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_factor
        )
    )

    start_epoch = 0
    start_batch = 0
    step = 0
    best = float("inf")

    output = (
        Path("outputs/checkpoints")
        / f"timetok_v3_{args.modality}_ae"
    )

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        expected = {
            "version": "v3-local-wavelet",
            "stage": "ae",
            "modality": args.modality,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_train": args.n_train,
        }

        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(
                    f"Resume mismatch: {key}"
                )

        model.load_state_dict(
            checkpoint["model"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["next_batch"]
        step = checkpoint["step"]
        best = checkpoint["best_val"]

        normalization = checkpoint[
            "normalization"
        ]
        mean = normalization["mean"]
        std = normalization["std"]

        torch.set_rng_state(
            checkpoint["rng_cpu"]
        )
        torch.cuda.set_rng_state(
            checkpoint["rng_cuda"]
        )
        np.random.set_state(
            checkpoint["rng_numpy"]
        )

        print(
            f"RESUME | epoch={start_epoch+1} "
            f"batch={start_batch+1} step={step}",
            flush=True,
        )

    amp = not args.no_amp

    print(
        f"V3 | modality={args.modality} "
        f"epochs={args.epochs} "
        f"train={len(train_ds):,} "
        f"steps/epoch={batches_per_epoch:,}",
        flush=True,
    )
    print(
        f"Normalization: mean={mean:.5f}, "
        f"std={std:.5f}",
        flush=True,
    )
    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    if args.smoke_test:
        x = next(iter(DataLoader(
            train_ds,
            batch_size=2,
            num_workers=0,
        )))
        x = prepare(x, mean, std, device)

        for keep in LEVELS:
            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                y, ids = model(x, keep)
                loss, _ = reconstruction_loss(
                    y, x
                )

            loss.backward()

            grad = model.encoder.net[
                0
            ].weight.grad

            assert y.shape == x.shape
            assert torch.isfinite(loss).all()
            assert grad is not None
            assert torch.isfinite(grad).all()

            print(
                f"SMOKE OK | tokens={keep} "
                f"loss={loss.item():.4f} "
                f"unique={ids.unique().numel()} "
                f"grad={grad.abs().mean().item():.2e}",
                flush=True,
            )

        print(
            "SMOKE TEST PASSED",
            flush=True,
        )
        return

    def evaluate(epoch, next_batch):
        nonlocal best

        val = validate(
            model, val_loader,
            mean, std, device, amp
        )

        shown_epoch = (
            epoch if next_batch == 0
            else epoch + 1
        )

        print(
            f"VAL | epoch={shown_epoch}/{args.epochs} "
            f"step={step:,} "
            f"zero_MAE={val['zero_mae']:.4f} "
            f"shuffled128_MAE="
            f"{val['shuffled_mae_128']:.4f}",
            flush=True,
        )

        for keep in LEVELS:
            m = val[str(keep)]

            print(
                f"  tokens={keep:3d} "
                f"| MAE={m['mae']:.4f} "
                f"RMSE={m['rmse']:.4f} "
                f"Corr={m['corr']:.4f} "
                f"STFT={m['stft']:.4f} "
                f"AE_loss={m['ae']:.4f} "
                f"unique={val['unique'][str(keep)]}",
                flush=True,
            )

        score = val["128"]["ae"]
        improved = score < best

        if improved:
            best = score

        save(
            output / "last.pt",
            model, optimizer, scheduler,
            args, epoch, next_batch,
            step, best, normalization, val
        )

        if improved:
            save(
                output / "best.pt",
                model, optimizer, scheduler,
                args, epoch, next_batch,
                step, best, normalization, val
            )
            print(
                f"Saved BEST | AE128={score:.4f}",
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

        sampler = EpochPermutation(
            len(train_ds),
            args.seed,
            epoch,
            consumed,
            args.batch_size,
        )

        loader = DataLoader(
            train_ds,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(
                args.workers > 0
            ),
        )

        print(
            f"\nEPOCH {epoch+1}/{args.epochs} "
            f"batch={consumed+1:,}/{batches_per_epoch:,}",
            flush=True,
        )

        model.train()

        for batch_idx, raw in enumerate(
            loader, start=consumed + 1
        ):
            x = prepare(
                raw, mean, std, device
            )

            keep = LEVELS[
                torch.randint(
                    len(LEVELS), (1,)
                ).item()
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                reconstruction, ids = model(
                    x, keep
                )

                loss, parts = reconstruction_loss(
                    reconstruction, x
                )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Nonfinite loss at step {step+1}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
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
                    f"batch={batch_idx:,}/{batches_per_epoch:,} "
                    f"step={step:,} "
                    f"keep={keep} "
                    f"MAE={parts['mae'].item():.4f} "
                    f"STFT={parts['spec'].item():.4f} "
                    f"unique={ids.unique().numel()}",
                    flush=True,
                )

            if (
                step % args.val_every == 0
                and batch_idx < batches_per_epoch
            ):
                evaluate(epoch, batch_idx)

        evaluate(epoch + 1, 0)
        start_batch = 0

    print(
        "TRAINING FINISHED",
        flush=True,
    )


if __name__ == "__main__":
    main()
