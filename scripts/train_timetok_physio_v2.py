"""Two-stage SHHS1 training: autoencoder then flow."""

import argparse
import math
import os
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    Dataset, DataLoader, Sampler, Subset
)

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset
)
from adaptive_physio.models.timetok_physio_v2 import (
    LEVELS,
    PhysioTokenizerV2,
    reconstruction_loss,
    reconstruction_stats,
    flow_loss,
)


class ChannelDataset(Dataset):
    def __init__(self, source, modality):
        self.source = source
        self.modality = modality

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        rid = bisect_right(
            self.source.cumulative, i
        )
        previous = (
            self.source.cumulative[rid - 1]
            if rid else 0
        )
        rec = self.source.records[rid]
        offset = (
            rec["positions"][self.modality]
            + i - previous
        )

        signal = np.array(
            self.source._get_mmap(
                self.modality
            )[offset],
            copy=True
        )

        if not np.isfinite(signal).all():
            raise ValueError(
                f"Nonfinite {self.modality} window {i}"
            )

        return torch.from_numpy(
            signal
        ).unsqueeze(0)


def split_stats(ds, modality):
    info = ds.metadata[modality]["sig_info"]
    count = 0
    total = 0.0
    squares = 0.0

    for rec in ds.records:
        stats = info[
            rec["recording"]
        ]["stats"]

        count += rec["windows"] * 3000
        total += float(stats["sum"])
        squares += float(stats["sumsq"])

    mean = total / count
    std = math.sqrt(
        max(
            squares / count - mean ** 2,
            1e-8
        )
    )

    if not math.isfinite(mean + std):
        raise ValueError(
            "Invalid training statistics"
        )

    return mean, std


class EpochPermutation(Sampler):
    def __init__(
        self, n, seed, epoch,
        start_batch, batch_size
    ):
        self.n = n
        self.seed = seed
        self.epoch = epoch
        self.offset = min(
            n, start_batch * batch_size
        )

    def __iter__(self):
        g = torch.Generator().manual_seed(
            self.seed + self.epoch
        )
        order = torch.randperm(
            self.n, generator=g
        )

        for idx in order[self.offset:].tolist():
            yield idx

    def __len__(self):
        return self.n - self.offset


def build_optimizer(model, args):
    if args.stage == "ae":
        groups = [
            {
                "params": list(
                    model.encoder.parameters()
                ),
                "lr": args.lr,
            },
            {
                "params": list(
                    model.reconstructor.parameters()
                ),
                "lr": args.lr,
            },
        ]
    else:
        groups = [
            {
                "params": list(
                    model.encoder.parameters()
                ),
                "lr": (
                    args.lr * args.encoder_lr_factor
                ),
            },
            {
                "params": list(
                    model.reconstructor.parameters()
                ),
                "lr": (
                    args.lr * args.encoder_lr_factor
                ),
            },
            {
                "params": list(
                    model.flow.parameters()
                ),
                "lr": args.lr,
            },
        ]

    return torch.optim.AdamW(
        groups, weight_decay=0.01
    )


def save_checkpoint(
    path, model, optimizer, scheduler,
    args, epoch, next_batch,
    global_step, best, mean, std, val
):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stage": args.stage,
        "modality": args.modality,
        "epoch": epoch,
        "next_batch": next_batch,
        "step": global_step,
        "best_val": best,
        "val": val,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_train": args.n_train,
        "normalization": {
            "mean": mean,
            "std": std,
        },
        "config": {
            "length": 3000,
            "patch": 25,
            "dim": 256,
            "levels": LEVELS,
        },
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "rng_numpy": np.random.get_state(),
    }

    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(
    model, loader, args,
    device, mean, std, use_amp
):
    model.eval()

    counts = {
        k: 0 for k in LEVELS
    }
    sums = {
        k: {
            "ae": 0.,
            "mae": 0.,
            "rmse": 0.,
            "corr": 0.,
            "spec": 0.,
            "fm": 0.,
        }
        for k in LEVELS
    }

    unique = set()
    baseline_sum = 0.
    n_all = 0

    # Fixed validation noise.
    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()]
    ):
        torch.manual_seed(817)

        for raw in loader:
            x = (
                (
                    raw.to(
                        device,
                        non_blocking=True
                    ) - mean
                ) / std
            ).clamp(-8, 8)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp
            ):
                codes, ids = model.tokenize(x)

                unique.update(
                    ids.detach().flatten()
                    .cpu().tolist()
                )

                baseline_sum += (
                    x.abs().sum().item()
                    / x.shape[-1]
                )
                n_all += len(x)

                for k in LEVELS:
                    recon = model.reconstructor(
                        codes, k
                    )

                    ae, terms = reconstruction_loss(
                        recon, x
                    )
                    mae, rmse, corr = reconstruction_stats(
                        recon, x
                    )

                    fm = (
                        flow_loss(
                            model.flow, codes, x, k
                        )
                        if args.stage == "flow"
                        else None
                    )

                    counts[k] += len(x)

                    sums[k]["ae"] += (
                        ae.item() * len(x)
                    )
                    sums[k]["mae"] += (
                        mae.sum().item()
                    )
                    sums[k]["rmse"] += (
                        rmse.sum().item()
                    )
                    sums[k]["corr"] += (
                        corr.sum().item()
                    )
                    sums[k]["spec"] += (
                        terms["spec"].item()
                        * len(x)
                    )

                    if fm is not None:
                        sums[k]["fm"] += (
                            fm.item() * len(x)
                        )

    model.train()

    result = {
        str(k): {
            metric: value / counts[k]
            for metric, value in sums[k].items()
        }
        for k in LEVELS
    }

    result["unique_ids"] = len(unique)
    result["zero_mae"] = (
        baseline_sum / n_all
    )

    return result


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage",
        choices=("ae", "flow"),
        required=True
    )
    p.add_argument(
        "--modality",
        choices=("eeg", "ecg", "resp"),
        default="eeg"
    )
    p.add_argument(
        "--epochs", type=int, default=1
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
        "--encoder-lr-factor",
        type=float, default=0.1
    )
    p.add_argument(
        "--ae-weight", type=float, default=0.5
    )
    p.add_argument(
        "--val-every", type=int, default=10000
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
    p.add_argument(
        "--init-from", default=None
    )
    p.add_argument(
        "--resume", default=None
    )
    p.add_argument(
        "--no-amp", action="store_true"
    )

    args = p.parse_args()

    if args.init_from and args.resume:
        p.error(
            "Select either --init-from or --resume"
        )

    if (
        args.stage == "flow"
        and not (args.init_from or args.resume)
    ):
        p.error(
            "Flow requires an AE checkpoint"
        )

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.val_every < 1
    ):
        p.error("Invalid training parameters")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda:0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    print("Loading SHHS1...", flush=True)

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

    n_batches = math.ceil(
        len(train_ds) / args.batch_size
    )
    total_steps = (
        n_batches * args.epochs
    )

    print(
        f"{args.stage.upper()} | "
        f"modality={args.modality} | "
        f"train={len(train_ds):,} | "
        f"val={len(val_ds):,}",
        flush=True
    )
    print(
        f"epochs={args.epochs} | "
        f"steps/epoch={n_batches:,} | "
        f"batch={args.batch_size} | "
        f"mean={mean:.5f} | std={std:.5f}",
        flush=True
    )

    model = PhysioTokenizerV2().to(device)
    optimizer = build_optimizer(model, args)

    def schedule(s):
        warmup = min(
            500, total_steps // 10
        )
        ramp = min(
            1., (s + 1) / max(1, warmup)
        )
        cosine = (
            0.05 + 0.95 * 0.5
            * (
                1. + math.cos(
                    math.pi
                    * min(s, total_steps)
                    / total_steps
                )
            )
        )
        return ramp * cosine

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=schedule
        )
    )

    start_epoch = 0
    start_batch = 0
    global_step = 0
    best = float("inf")

    if args.init_from:
        checkpoint = torch.load(
            args.init_from,
            map_location="cpu",
            weights_only=False
        )

        if (
            checkpoint["stage"] != "ae"
            or checkpoint["modality"]
            != args.modality
        ):
            raise ValueError(
                "Expected matching V2 AE checkpoint"
            )

        model.load_state_dict(
            checkpoint["model"]
        )
        mean = checkpoint[
            "normalization"
        ]["mean"]
        std = checkpoint[
            "normalization"
        ]["std"]

        print(
            "FLOW initialized from "
            f"AE step {checkpoint['step']:,}",
            flush=True
        )

    elif args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False
        )

        expected = [
            ("stage", args.stage),
            ("modality", args.modality),
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("seed", args.seed),
            ("n_train", args.n_train),
        ]

        for key, value in expected:
            if checkpoint[key] != value:
                raise ValueError(
                    f"Checkpoint mismatch: {key}"
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
        global_step = checkpoint["step"]
        best = checkpoint["best_val"]

        mean = checkpoint[
            "normalization"
        ]["mean"]
        std = checkpoint[
            "normalization"
        ]["std"]

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
            f"Resuming epoch {start_epoch + 1}, "
            f"next batch {start_batch + 1}",
            flush=True
        )

    if start_epoch >= args.epochs:
        print("Training already completed")
        return

    positions = torch.linspace(
        0,
        len(val_ds) - 1,
        min(args.val_windows, len(val_ds))
    ).long().tolist()

    val_loader = DataLoader(
        Subset(val_ds, positions),
        batch_size=args.batch_size,
        num_workers=min(args.workers, 2),
        pin_memory=True
    )

    output = (
        Path("outputs/checkpoints")
        / f"timetok_v2_{args.modality}_{args.stage}"
    )

    use_amp = not args.no_amp

    def evaluation(epoch, next_batch):
        nonlocal best

        val = validate(
            model, val_loader, args,
            device, mean, std, use_amp
        )

        display_epoch = (
            epoch if next_batch == 0
            else epoch + 1
        )

        print(
            f"VAL | stage={args.stage} "
            f"epoch={display_epoch}/{args.epochs} "
            f"step={global_step:,} "
            f"unique={val['unique_ids']} "
            f"zero_MAE={val['zero_mae']:.4f}",
            flush=True
        )

        for k in LEVELS:
            v = val[str(k)]

            message = (
                f"  tokens={k:3d} "
                f"| MAE={v['mae']:.4f} "
                f"| RMSE={v['rmse']:.4f} "
                f"| Corr={v['corr']:.4f} "
                f"| STFT={v['spec']:.4f} "
                f"| AE_loss={v['ae']:.4f}"
            )

            if args.stage == "flow":
                message += (
                    f" | FM_loss={v['fm']:.4f}"
                )

            print(message, flush=True)

        if args.stage == "ae":
            score = val["128"]["ae"]
        else:
            score = sum(
                val[str(k)]["fm"]
                for k in LEVELS
            ) / len(LEVELS)

        improved = score < best

        if improved:
            best = score

        save_checkpoint(
            output / "last.pt",
            model, optimizer, scheduler,
            args, epoch, next_batch,
            global_step, best, mean, std, val
        )

        if improved:
            save_checkpoint(
                output / "best.pt",
                model, optimizer, scheduler,
                args, epoch, next_batch,
                global_step, best, mean, std, val
            )
            print(
                f"Saved BEST | criterion={score:.4f}",
                flush=True
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
            args.batch_size
        )

        loader = DataLoader(
            train_ds,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0
        )

        model.train()

        print(
            f"\nEPOCH {epoch + 1}/{args.epochs} "
            f"starting batch "
            f"{consumed + 1:,}/{n_batches:,}",
            flush=True
        )

        for batch_idx, raw in enumerate(
            loader, start=consumed + 1
        ):
            x = (
                (
                    raw.to(
                        device,
                        non_blocking=True
                    ) - mean
                ) / std
            ).clamp(-8, 8)

            # One level per batch.
            keep = LEVELS[
                torch.randint(
                    0, len(LEVELS), (1,)
                ).item()
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp
            ):
                codes, ids = model.tokenize(x)

                recon = model.reconstructor(
                    codes, keep
                )

                ae, terms = reconstruction_loss(
                    recon, x
                )

                if args.stage == "ae":
                    loss = ae
                    fm = None
                else:
                    fm = flow_loss(
                        model.flow,
                        codes, x, keep
                    )
                    loss = (
                        fm + args.ae_weight * ae
                    )

            if not bool(
                torch.isfinite(loss).item()
            ):
                raise RuntimeError(
                    "Nonfinite loss at step "
                    f"{global_step + 1}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()
            scheduler.step()

            global_step += 1

            if (
                global_step == 1
                or global_step % args.log_every == 0
            ):
                message = (
                    f"stage={args.stage} "
                    f"epoch={epoch+1}/{args.epochs} "
                    f"batch={batch_idx:,}/{n_batches:,} "
                    f"step={global_step:,} "
                    f"keep={keep} "
                    f"MAE={terms['mae'].item():.4f} "
                    f"STFT={terms['spec'].item():.4f} "
                    f"unique={torch.unique(ids).numel()}"
                )

                if fm is not None:
                    message += (
                        f" FM={fm.item():.4f}"
                    )

                print(message, flush=True)

            if (
                global_step % args.val_every == 0
                and batch_idx < n_batches
            ):
                evaluation(epoch, batch_idx)

        evaluation(epoch + 1, 0)
        start_batch = 0

    print(
        "TRAINING FINISHED",
        flush=True
    )


if __name__ == "__main__":
    main()
