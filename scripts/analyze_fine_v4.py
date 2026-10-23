"""Post-hoc EEG analysis of V4 granularity decisions."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from train_timetok_physio_v2 import ChannelDataset
from train_temporal_agent_v3 import load_backbone, normalize
from train_temporal_agent_v4 import load_agent, rollout_lambdas


def features_16(x):
    """EEG features for the 16 temporal regions."""
    b, _, length = x.shape

    bands = [
        (0.5, 4),    # Delta
        (4, 8),      # Theta
        (8, 12),     # Alpha
        (12, 16),    # Sigma
        (16, 30),    # Beta
    ]

    results = []

    for r in range(16):
        start = round(r * length / 16)
        end = round((r + 1) * length / 16)

        signal = x[:, 0, start:end].float()

        rms = signal.square().mean(-1).sqrt()
        line = signal.diff(dim=-1).abs().mean(-1)

        window = torch.hann_window(
            end - start, device=x.device
        )

        spectrum = torch.fft.rfft(
            signal * window, dim=-1
        ).abs().square()

        freqs = torch.fft.rfftfreq(
            end - start, d=0.01, device=x.device
        )

        power = torch.stack([
            spectrum[
                :, (freqs >= low) & (freqs < high)
            ].sum(-1)
            for low, high in bands
        ], dim=-1)

        fractions = power / power.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)

        results.append(torch.cat(
            [rms[:, None], line[:, None], fractions],
            dim=-1
        ))

    return torch.stack(results, dim=1)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--split", default="val")
    p.add_argument(
        "--recording", default="shhs1-202318"
    )
    p.add_argument(
        "--lambda-cost", type=float, default=0.0008
    )
    p.add_argument(
        "--batch-size", type=int, default=32
    )
    p.add_argument(
        "--backbone",
        default="outputs/checkpoints/timetok_v3_eeg_ae/frozen_for_agent.pt",
    )
    p.add_argument(
        "--agent",
        default="outputs/checkpoints/timetok_v3_agent_cost_eeg/best.pt",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/evaluation/timetok_v3/fine_diagnostics",
    )

    args = p.parse_args()

    if args.lambda_cost < 0:
        p.error("lambda-cost must be nonnegative")

    device = torch.device("cuda:0")

    source = PreprocessedPSGDataset(split=args.split)

    rec_index = next(
        (
            i for i, rec in enumerate(source.records)
            if str(rec["recording"]) == args.recording
        ),
        None
    )

    if rec_index is None:
        p.error(
            f"{args.recording} not found in {args.split}"
        )

    start = (
        int(source.cumulative[rec_index - 1])
        if rec_index else 0
    )

    count = int(
        source.records[rec_index]["windows"]
    )

    dataset = Subset(
        ChannelDataset(source, "eeg"),
        range(start, start + count)
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )

    backbone, bc = load_backbone(
        args.backbone, device
    )

    agent, ac = load_agent(
        args.agent, device
    )

    if ac["backbone_step"] != bc["step"]:
        raise ValueError("Backbone checkpoint mismatch")

    agent.eval()

    rows = []
    processed = 0

    with torch.inference_mode():
        for raw in loader:
            x = normalize(
                raw, bc["normalization"], device
            )

            with torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                codes, _ = backbone.tokenize(x)

            codes = codes.float()

            _, levels, used = rollout_lambdas(
                agent,
                codes,
                [args.lambda_cost],
                maximum=128
            )

            # Ground-truth features are used
            # ONLY for this post-hoc analysis.
            feats = features_16(x).cpu().numpy()

            lv = levels.cpu().numpy()
            budget = used.cpu().numpy()

            coarse_scores = agent(
                codes,
                torch.zeros_like(levels)
            )[..., 0].cpu().numpy()

            for j in range(len(x)):
                for region in range(16):
                    f = feats[j, region]

                    rows.append({
                        "window": processed + j,
                        "hour": (processed + j) / 120.0,
                        "region": region,
                        "level": int(lv[j, region]),
                        "used_window": int(budget[j]),
                        "rms": float(f[0]),
                        "line_length": float(f[1]),
                        "delta_fraction": float(f[2]),
                        "theta_fraction": float(f[3]),
                        "alpha_fraction": float(f[4]),
                        "sigma_fraction": float(f[5]),
                        "beta_fraction": float(f[6]),
                        "coarse_predicted_gain": float(
                            coarse_scores[j, region]
                        ),
                    })

            processed += len(x)

            if (
                processed % 256 == 0
                or processed == count
            ):
                print(
                    f"Analyzed {processed}/{count} windows",
                    flush=True
                )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    prefix = (
        f"{args.recording}_lambda_"
        f"{args.lambda_cost:g}"
    )

    csv_path = output / f"{prefix}_regions.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== EEG FEATURES BY GRANULARITY ===")

    feature_names = [
        "rms",
        "line_length",
        "delta_fraction",
        "sigma_fraction",
        "beta_fraction",
        "coarse_predicted_gain",
    ]

    for level, name in [
        (0, "COARSE"),
        (1, "MID"),
        (2, "FINE"),
    ]:
        selected = [
            r for r in rows if r["level"] == level
        ]

        print(
            f"\n{name}: {len(selected)} regions "
            f"({len(selected)/len(rows):.1%})"
        )

        if selected:
            for key in feature_names:
                value = np.mean([
                    r[key] for r in selected
                ])
                print(f"  {key}: {value:.5f}")

    print("\n=== ASSOCIATION WITH FINE DECISIONS ===")

    fine_label = np.array(
        [r["level"] == 2 for r in rows],
        dtype=float
    )

    for key in feature_names:
        values = np.array(
            [r[key] for r in rows]
        )

        correlation = np.corrcoef(
            values, fine_label
        )[0, 1]

        q10, q90 = np.quantile(
            values, [0.1, 0.9]
        )

        low = fine_label[
            values <= q10
        ].mean() * 100

        high = fine_label[
            values >= q90
        ].mean() * 100

        print(
            f"{key:22s} | "
            f"r={correlation:+.3f} | "
            f"fine low={low:.1f}% | "
            f"fine high={high:.1f}%"
        )

    # Night-level aggregation in 30-minute bins.
    groups = np.arange(count) // 60

    used = np.array([
        rows[i * 16]["used_window"]
        for i in range(count)
    ])

    fine = fine_label.reshape(
        count, 16
    ).mean(1)

    rms = np.array([
        r["rms"] for r in rows
    ]).reshape(count, 16).mean(1)

    beta = np.array([
        r["beta_fraction"] for r in rows
    ]).reshape(count, 16).mean(1)

    line = np.array([
        r["line_length"] for r in rows
    ]).reshape(count, 16).mean(1)

    centers = []
    avg_used = []
    avg_fine = []
    avg_rms = []
    avg_beta = []
    avg_line = []

    for group in np.unique(groups):
        mask = groups == group

        centers.append(
            (np.flatnonzero(mask).mean() + 0.5) / 120
        )
        avg_used.append(used[mask].mean())
        avg_fine.append(fine[mask].mean())
        avg_rms.append(rms[mask].mean())
        avg_beta.append(beta[mask].mean())
        avg_line.append(line[mask].mean())

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True
    )

    hours = np.arange(count) / 120.0

    axes[0].plot(
        hours, used,
        color="#7057a3",
        linewidth=0.65,
        alpha=0.7,
        label="Tokens per window"
    )

    axes[0].plot(
        centers, avg_used,
        color="#30204f",
        linewidth=2,
        label="30-minute mean"
    )

    axes[0].set_ylabel("Selected tokens")
    axes[0].legend()

    axes[1].plot(
        centers, avg_fine,
        color="#df5b54",
        linewidth=2,
        label="Fine fraction"
    )

    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fine region share")
    axes[1].legend()

    for values, name, color in [
        (avg_rms, "RMS", "#346d9c"),
        (avg_line, "Line length", "#269382"),
        (avg_beta, "Beta power fraction", "#b98d32"),
    ]:
        a = np.asarray(values)
        z = (a - a.mean()) / (a.std() + 1e-8)

        axes[2].plot(
            centers, z,
            linewidth=1.8,
            color=color,
            label=name
        )

    axes[2].set_ylabel(
        "EEG feature (night-standardized)"
    )
    axes[2].set_xlabel(
        "Hours since recording start"
    )
    axes[2].legend(ncol=3)

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{args.recording} — Why does the agent refine?"
    )
    fig.tight_layout()

    png_path = output / f"{prefix}_why_fine.png"

    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print("\nSaved:", csv_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
