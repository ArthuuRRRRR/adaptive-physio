"""Plot V3 reconstruction at each granularity."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio_v3 import (
    LEVELS, PhysioTokenizerV3,
)
from train_timetok_physio_v2 import ChannelDataset


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/timetok_v3_eeg_ae/best.pt",
    )
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument(
        "--output",
        default="outputs/evaluation/timetok_v3/conditioning.png",
    )

    args = p.parse_args()

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = PhysioTokenizerV3().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source = PreprocessedPSGDataset(
        split="val"
    )
    dataset = ChannelDataset(
        source,
        checkpoint["modality"],
    )

    if not 0 <= args.index < len(dataset):
        raise ValueError(
            "Invalid validation window index"
        )

    # Use a distant window for mismatched conditioning.
    wrong_index = (
        args.index + len(dataset) // 2
    ) % len(dataset)

    x = torch.stack([
        dataset[args.index],
        dataset[wrong_index],
    ]).to(device)

    norm = checkpoint["normalization"]
    x = (
        (x - norm["mean"])
        / norm["std"]
    ).clamp(-8, 8)

    predictions = {}

    with torch.inference_mode():
        codes, _ = model.tokenize(x)

        for keep in LEVELS:
            correct = model.reconstructor(
                codes[:1], keep
            )
            shuffled = model.reconstructor(
                codes[1:], keep
            )

            predictions[keep] = (
                correct[0, 0].float().cpu().numpy(),
                shuffled[0, 0].float().cpu().numpy(),
            )

    original = x[0, 0].float().cpu().numpy()

    n = min(
        len(original),
        max(1, int(args.seconds * 100)),
    )

    times = np.arange(n) / 100.0

    fig, axes = plt.subplots(
        3, 2,
        figsize=(14, 9),
        sharex=True,
        sharey=True,
    )

    for row, keep in enumerate(LEVELS):
        for col in (0, 1):
            ax = axes[row, col]

            ax.plot(
                times, original[:n],
                label="Original",
                linewidth=0.8,
            )

            ax.plot(
                times,
                predictions[keep][col][:n],
                label=(
                    "Correct"
                    if col == 0
                    else "Wrong tokens"
                ),
                linewidth=0.8,
            )

            ax.set_title(
                f"{keep} tokens - "
                + (
                    "Correct"
                    if col == 0
                    else "Shuffled"
                )
            )

            ax.grid(alpha=0.25)

            if row == 0:
                ax.legend()

            if col == 0:
                ax.set_ylabel(
                    "EEG (normalized)"
                )

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    fig.suptitle(
        "TimeTok-Physio V3 | "
        f"Checkpoint step {checkpoint['step']}"
    )

    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(output, dpi=160)
    plt.close(fig)

    print("Saved:", output)


if __name__ == "__main__":
    main()
