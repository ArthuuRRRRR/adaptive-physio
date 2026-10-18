import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio_v3 import (
    PhysioTokenizerV3, LEVELS,
)
from train_timetok_physio_v2 import ChannelDataset
from train_timetok_physio_v3 import validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/timetok_v3_eeg_ae/best.pt",
    )
    p.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
    )
    p.add_argument(
        "--windows", type=int, default=2048
    )
    p.add_argument(
        "--batch-size", type=int, default=8
    )
    args = p.parse_args()

    if args.windows < 2 or args.batch_size < 2:
        p.error("Use at least two windows and batch size >= 2")

    device = torch.device("cuda:0")

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = PhysioTokenizerV3().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    source = PreprocessedPSGDataset(
        split=args.split
    )
    dataset = ChannelDataset(
        source, ckpt["modality"]
    )

    n = min(args.windows, len(dataset))
    indices = torch.linspace(
        0, len(dataset) - 1, n
    ).long().tolist()

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    norm = ckpt["normalization"]

    result = validate(
        model,
        loader,
        norm["mean"],
        norm["std"],
        device,
        amp=True,
    )

    print("\n=== TIMETOK V3 EVALUATION ===")
    print(f"Split: {args.split}")
    print(f"Checkpoint step: {ckpt['step']:,}")
    print(f"Windows: {n:,}")
    print(f"Zero baseline MAE: {result['zero_mae']:.4f}")
    print(
        f"Shuffled 128 MAE: "
        f"{result['shuffled_mae_128']:.4f}"
    )

    print(
        "\nTokens | MAE | RMSE | Corr | STFT | Unique"
    )
    for k in LEVELS:
        r = result[str(k)]
        print(
            f"{k:6d} | "
            f"{r['mae']:.4f} | "
            f"{r['rmse']:.4f} | "
            f"{r['corr']:.4f} | "
            f"{r['stft']:.4f} | "
            f"{result['unique'][str(k)]}"
        )

    output = Path(
        f"outputs/evaluation/timetok_v3/"
        f"metrics_{args.split}_{n}.json"
    )
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output.open("w") as f:
        json.dump({
            "split": args.split,
            "checkpoint_step": ckpt["step"],
            "windows": n,
            "metrics": result,
        }, f, indent=2)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
