"""Evaluate learned STOP thresholds and the quality/token tradeoff."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


@torch.inference_mode()
def rollout_thresholds(agent, codes, thresholds, max_budget):
    """Run all STOP thresholds in parallel for each window."""
    batch = codes.shape[0]
    nt = len(thresholds)
    device = codes.device

    expanded = codes.repeat_interleave(nt, dim=0)

    stop_values = torch.tensor(
        thresholds, device=device, dtype=torch.float32
    ).repeat(batch)

    total = batch * nt
    levels = torch.zeros(
        total, 16, device=device, dtype=torch.long
    )
    used = torch.full(
        (total,), 16, device=device, dtype=torch.long
    )

    for _ in range(32):
        output = agent(expanded, levels)

        scores = torch.where(
            levels == 0,
            output[..., 0],
            output[..., 1],
        )

        costs = torch.where(levels == 0, 3, 4)

        eligible = (
            (levels < 2)
            & (used[:, None] + costs <= max_budget)
        )

        scores = scores.masked_fill(~eligible, -torch.inf)

        best, region = scores.max(dim=1)

        selected = (
            torch.isfinite(best)
            & (best > stop_values)
        )

        if not selected.any().item():
            break

        rows = torch.where(selected)[0]
        cols = region[rows]

        used[rows] += costs[rows, cols]
        levels[rows, cols] += 1

    return expanded, levels, used


def main():
    p = argparse.ArgumentParser()

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
            "timetok_v3_agent_eeg/best.pt"
        ),
    )
    p.add_argument(
        "--reference",
        default=(
            "outputs/evaluation/timetok_v3/"
            "agent_comparison_2048.csv"
        ),
    )
    p.add_argument("--windows", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-budget", type=int, default=128)
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[
            0.0, 0.05, 0.10, 0.20, 0.40,
            0.80, 1.20, 2.0, 3.0, 5.0,
        ],
    )
    p.add_argument(
        "--output-dir",
        default="outputs/evaluation/timetok_v3/budget_sweep",
    )
    args = p.parse_args()

    if args.windows < 1 or args.batch_size < 1:
        p.error("Invalid evaluation size")

    if not 16 <= args.max_budget <= 128:
        p.error("max-budget must be between 16 and 128")

    if (
        len(args.thresholds) != len(set(args.thresholds))
        or min(args.thresholds) < 0
    ):
        p.error("Thresholds must be unique and nonnegative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")

    backbone, backbone_ckpt = load_backbone(
        args.backbone, device
    )

    ckpt = torch.load(
        args.agent_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if ckpt["backbone_step"] != backbone_ckpt["step"]:
        raise ValueError("Backbone checkpoint mismatch")

    agent = TemporalRefinementAgent().to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    loader, n = data_loader(
        "val", "eeg", args.windows,
        args.batch_size, args.workers, seed=42
    )

    norm = backbone_ckpt["normalization"]
    nt = len(args.thresholds)

    mae_sum = torch.zeros(nt, dtype=torch.float64)
    corr_sum = torch.zeros(nt, dtype=torch.float64)
    used_sum = torch.zeros(nt, dtype=torch.float64)

    level_sums = {
        k: torch.zeros(nt, dtype=torch.float64)
        for k in range(3)
    }

    all_costs = [[] for _ in range(nt)]
    processed = 0

    with torch.inference_mode():
        for raw in loader:
            x = normalize(raw, norm, device)
            batch = len(x)

            with torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                codes, _ = backbone.tokenize(x)

            expanded, levels, used = rollout_thresholds(
                agent,
                codes.float(),
                args.thresholds,
                args.max_budget,
            )

            recon = decode_mixed(
                backbone.reconstructor,
                expanded,
                levels,
            )

            reference = x.repeat_interleave(nt, dim=0)

            mae, _, corr = reconstruction_stats(
                recon, reference
            )

            mae = mae.view(batch, nt).cpu().double()
            corr = corr.view(batch, nt).cpu().double()
            used = used.view(batch, nt).cpu()
            levels = levels.view(batch, nt, 16).cpu()

            mae_sum += mae.sum(0)
            corr_sum += corr.sum(0)
            used_sum += used.sum(0)

            for level in range(3):
                level_sums[level] += (
                    (levels == level).sum((0, 2))
                )

            for j in range(nt):
                all_costs[j].extend(
                    used[:, j].tolist()
                )

            processed += batch

            if processed % 128 == 0 or processed == n:
                print(
                    f"Evaluated {processed}/{n}",
                    flush=True,
                )

    rows = []

    print("\n=== ADAPTIVE STOP THRESHOLD SWEEP ===")
    print(
        "Threshold | Avg used | P10 | P90 | "
        "MAE | Corr | C% | M% | F%"
    )

    for j, threshold in enumerate(args.thresholds):
        costs = np.asarray(all_costs[j])

        row = {
            "threshold": threshold,
            "avg_used": float(used_sum[j] / processed),
            "p10_used": float(np.percentile(costs, 10)),
            "p90_used": float(np.percentile(costs, 90)),
            "mae": float(mae_sum[j] / processed),
            "corr": float(corr_sum[j] / processed),
            "coarse_pct": float(
                level_sums[0][j] / (processed * 16) * 100
            ),
            "mid_pct": float(
                level_sums[1][j] / (processed * 16) * 100
            ),
            "fine_pct": float(
                level_sums[2][j] / (processed * 16) * 100
            ),
        }

        rows.append(row)

        print(
            f"{threshold:9.3f} | "
            f"{row['avg_used']:8.1f} | "
            f"{row['p10_used']:3.0f} | "
            f"{row['p90_used']:3.0f} | "
            f"{row['mae']:.4f} | "
            f"{row['corr']:.4f} | "
            f"{row['coarse_pct']:4.1f} | "
            f"{row['mid_pct']:4.1f} | "
            f"{row['fine_pct']:4.1f}"
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "threshold_sweep.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    # Read previously measured fixed-budget baselines.
    references = []
    reference_path = Path(args.reference)

    if reference_path.exists():
        with reference_path.open(newline="") as f:
            references = list(csv.DictReader(f))

        if args.windows != 2048:
            print(
                "WARNING: reference curves use 2048 windows"
            )
    else:
        print(
            "Reference CSV not found:",
            reference_path,
        )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.2),
        sharex=True,
    )

    colors = {
        "random": "#94a3b8",
        "oracle": "#0d9488",
        "agent": "#b45309",
    }

    for policy, label in (
        ("random", "Random (fixed budget)"),
        ("oracle", "Oracle (GT-informed)"),
        ("agent", "Agent (fixed budget)"),
    ):
        subset = [
            r for r in references
            if r["policy"] == policy
        ]

        subset.sort(
            key=lambda r: float(r["avg_used"])
        )

        if not subset:
            continue

        xx = [
            float(r["avg_used"])
            for r in subset
        ]

        for ax, metric in zip(
            axes, ("corr", "mae")
        ):
            ax.plot(
                xx,
                [float(r[metric]) for r in subset],
                marker="o",
                markersize=3,
                linestyle="--",
                linewidth=1.3,
                color=colors[policy],
                label=label,
            )

    xx = [r["avg_used"] for r in rows]

    for ax, metric in zip(
        axes, ("corr", "mae")
    ):
        ax.plot(
            xx,
            [r[metric] for r in rows],
            color="#4f46e5",
            marker="o",
            linewidth=2.5,
            markersize=5,
            label="Agent (learned STOP)",
        )

        ax.set_xlabel(
            "Mean selected tokens / 30 s window"
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(
        "Correlation to original EEG (higher better)"
    )
    axes[1].set_ylabel(
        "MAE to original EEG (lower better)"
    )

    axes[0].set_title(
        "Temporal fidelity vs token budget"
    )
    axes[1].set_title(
        "Waveform error vs token budget"
    )

    axes[0].set_xlim(12, 132)

    fig.suptitle(
        f"SHHS1 validation | {processed:,} windows "
        f"| V3 frozen | max budget {args.max_budget}"
    )

    fig.tight_layout()

    png_path = outdir / "quality_vs_tokens.png"
    fig.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved:", csv_path)
    print("Saved:", png_path)
    print(
        "Thresholds apply to scaled predicted "
        "per-token gains."
    )


if __name__ == "__main__":
    main()
