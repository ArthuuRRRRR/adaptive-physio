import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio_v3 import (
    PhysioTokenizerV3,
    LEVELS,
    reconstruction_stats,
)
from train_timetok_physio_v2 import ChannelDataset


def gaussian_smooth(x, sigma_seconds, fs=100):
    x = x.float()
    if sigma_seconds <= 0:
        return x

    sigma = sigma_seconds * fs
    radius = max(1, int(round(3 * sigma)))

    t = torch.arange(
        -radius, radius + 1,
        device=x.device,
        dtype=torch.float32,
    )

    kernel = torch.exp(
        -0.5 * (t / sigma).square()
    )
    kernel = (kernel / kernel.sum()).view(1, 1, -1)

    return F.conv1d(
        F.pad(x, (radius, radius), mode="reflect"),
        kernel,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/timetok_v3_eeg_ae/best.pt",
    )
    p.add_argument("--windows", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--coarse-sigma", type=float, default=0.20)
    p.add_argument("--mid-sigma", type=float, default=0.05)
    args = p.parse_args()

    if args.windows < 1 or args.batch_size < 1:
        p.error("Invalid evaluation size")
    if args.coarse_sigma < 0 or args.mid_sigma < 0:
        p.error("Smoothing sigmas must be nonnegative")

    device = torch.device("cuda:0")

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = PhysioTokenizerV3().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    source = PreprocessedPSGDataset(split="val")
    dataset = ChannelDataset(source, ckpt["modality"])

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

    sigmas = {
        16: args.coarse_sigma,
        64: args.mid_sigma,
        128: 0.0,
    }

    fields = (
        "mae_raw",
        "corr_raw",
        "mae_target",
        "corr_target",
        "mae_filtered",
        "corr_filtered",
        "zero_mae_target",
        "target_rms",
    )

    results = {
        k: {field: 0.0 for field in fields}
        for k in LEVELS
    }

    plot_data = None
    total = 0

    with torch.inference_mode():
        for raw in loader:
            x = (
                raw.to(device, non_blocking=True)
                - norm["mean"]
            ) / norm["std"]
            x = x.clamp(-8, 8).float()

            codes, _ = model.tokenize(x)
            total += len(x)

            first = plot_data is None

            if first:
                plot_data = {
                    "original": x[0, 0].cpu().numpy()
                }

            for k in LEVELS:
                pred = model.reconstructor(
                    codes, k
                ).float()

                target = gaussian_smooth(
                    x, sigmas[k]
                )

                pred_filtered = gaussian_smooth(
                    pred, sigmas[k]
                )

                mae_raw, _, corr_raw = (
                    reconstruction_stats(pred, x)
                )

                mae_target, _, corr_target = (
                    reconstruction_stats(pred, target)
                )

                mae_filtered, _, corr_filtered = (
                    reconstruction_stats(
                        pred_filtered, target
                    )
                )

                values = {
                    "mae_raw": mae_raw,
                    "corr_raw": corr_raw,
                    "mae_target": mae_target,
                    "corr_target": corr_target,
                    "mae_filtered": mae_filtered,
                    "corr_filtered": corr_filtered,
                    "zero_mae_target": (
                        target.abs().mean((1, 2))
                    ),
                    "target_rms": (
                        target.square()
                        .mean((1, 2)).sqrt()
                    ),
                }

                for name, tensor in values.items():
                    results[k][name] += tensor.sum().item()

                if first:
                    plot_data[f"target_{k}"] = (
                        target[0, 0].cpu().numpy()
                    )
                    plot_data[f"pred_{k}"] = (
                        pred[0, 0].cpu().numpy()
                    )

    output = Path(
        "outputs/evaluation/timetok_v3"
    )
    output.mkdir(parents=True, exist_ok=True)

    rows = []

    print(
        f"\nCheckpoint: {ckpt['step']:,} steps"
        f"\nWindows: {total:,}"
    )

    print("\n=== ORIGINAL EEG ===")
    print("Tokens | MAE | Corr")

    for k in LEVELS:
        r = {
            name: value / total
            for name, value in results[k].items()
        }

        skill = 100 * (
            1 - r["mae_target"]
            / max(r["zero_mae_target"], 1e-8)
        )

        rows.append({
            "tokens": k,
            "sigma_seconds": sigmas[k],
            **r,
            "target_skill_percent": skill,
        })

        print(
            f"{k:6d} | "
            f"{r['mae_raw']:.4f} | "
            f"{r['corr_raw']:.4f}"
        )

    print("\n=== GRANULARITY-ADAPTED TARGETS ===")
    print(
        "Tokens | Target MAE | Target Corr | "
        "Filtered Corr | Zero MAE | Skill"
    )

    for r in rows:
        print(
            f"{r['tokens']:6d} | "
            f"{r['mae_target']:.4f} | "
            f"{r['corr_target']:.4f} | "
            f"{r['corr_filtered']:.4f} | "
            f"{r['zero_mae_target']:.4f} | "
            f"{r['target_skill_percent']:+.1f}%"
        )

    path = output / "multires_metrics.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9),
        sharex=True,
    )

    nplot = 1000
    t = torch.arange(nplot).numpy() / 100

    for ax, k in zip(axes, LEVELS):
        ax.plot(
            t, plot_data["original"][:nplot],
            label="Original EEG",
            alpha=0.3, linewidth=0.7,
        )
        ax.plot(
            t, plot_data[f"target_{k}"][:nplot],
            label=f"Target {k}",
            linewidth=1.2,
        )
        ax.plot(
            t, plot_data[f"pred_{k}"][:nplot],
            label=f"Reconstruction {k}",
            linewidth=1,
        )
        ax.set_title(
            f"{k} tokens | sigma={sigmas[k]} s"
        )
        ax.set_ylabel("Normalized EEG")
        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    png = output / "multires_comparison.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    print("\nSaved:", path)
    print("Saved:", png)


if __name__ == "__main__":
    main()
