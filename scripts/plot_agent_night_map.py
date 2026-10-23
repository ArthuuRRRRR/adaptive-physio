import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.models.timetok_physio_v3 import (
    PhysioTokenizerV3,
)
from adaptive_physio.models.temporal_agent_v3 import (
    TemporalRefinementAgent,
)
from train_timetok_physio_v2 import (
    ChannelDataset,
)
from evaluate_timetok_v3_adaptive import (
    decode_mixed,
)

LEVEL_NAMES = {
    0: "coarse",
    1: "mid",
    2: "fine",
}

LEVEL_COLORS = {
    0: "#4C78A8",  # blue
    1: "#F2CF5B",  # yellow
    2: "#E45756",  # red
}

def find_metadata_arrays(dataset):
    """
    Recover per-window recording IDs and local indices
    using the actual PreprocessedPSGDataset structure.
    """
    records = dataset.records

    counts = np.array(
        [int(r["windows"]) for r in records],
        dtype=np.int64,
    )

    rec_ids = np.repeat(
        np.array(
            [str(r["recording"]) for r in records]
        ),
        counts,
    )

    win_ids = np.concatenate([
        np.arange(
            int(r["windows"]),
            dtype=np.int32,
        )
        for r in records
    ])

    assert len(rec_ids) == len(dataset)
    assert len(win_ids) == len(dataset)

    return rec_ids, win_ids


def load_backbone(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model = PhysioTokenizerV3().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)

    return model, checkpoint

def load_agent(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    agent = TemporalRefinementAgent().to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()

    return agent, checkpoint

def rollout_with_threshold(
    agent,
    codes,
    budget,
    stop_threshold=0.0,
):
    """
    Sequential rollout with optional stop threshold.

    stop_threshold:
        Stop when best predicted gain <= stop_threshold.
    """
    b = codes.shape[0]
    device = codes.device

    levels = torch.zeros(
        b, 16,
        dtype=torch.long,
        device=device,
    )

    used = torch.full(
        (b,),
        16,
        dtype=torch.long,
        device=device,
    )

    for _ in range(32):
        output = agent(codes, levels)

        scores = torch.where(
            levels == 0,
            output[..., 0],
            output[..., 1],
        )

        costs = torch.where(
            levels == 0, 3, 4
        )

        eligible = (
            (levels < 2)
            &
            (used[:, None] + costs <= budget)
        )

        scores = scores.masked_fill(
            ~eligible,
            -torch.inf,
        )

        best_score, region = scores.max(dim=1)

        selected = (
            torch.isfinite(best_score)
            & (best_score > stop_threshold)
        )

        if not selected.any().item():
            break

        rows = torch.where(selected)[0]
        cols = region[selected]

        added = costs[rows, cols]
        levels[rows, cols] += 1
        used[rows] += added

    return levels, used

def normalize(x, mean, std, device):
    return (
        (
            x.to(device, non_blocking=True)
            - mean
        ) / std
    ).clamp(-8, 8)

def pick_example_indices(used_tokens):
    """
    Pick low / medium / high budget examples.
    """
    used_tokens = np.asarray(used_tokens)
    targets = np.quantile(
        used_tokens, [0.1, 0.5, 0.9]
    )

    chosen = []
    taken = np.zeros(len(used_tokens), dtype=bool)

    for target in targets:
        scores = np.abs(used_tokens - target)
        scores[taken] = np.inf
        idx = int(np.argmin(scores))
        chosen.append(idx)
        taken[idx] = True

    return chosen

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
    )
    parser.add_argument(
        "--recording",
        default=None,
        help="Recording ID, e.g. shhs1-200001",
    )
    parser.add_argument(
        "--list-recordings",
        action="store_true",
    )
    parser.add_argument(
        "--backbone",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
    )
    parser.add_argument(
        "--agent-checkpoint",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_agent_eeg/best.pt"
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit for plotting/debug",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures/agent_night_maps",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")

    source = PreprocessedPSGDataset(
        split=args.split
    )
    rec_ids, win_ids = find_metadata_arrays(source)

    unique, counts = np.unique(
        rec_ids,
        return_counts=True,
    )

    if args.list_recordings:
        order = np.argsort(-counts)
        print(
            f"Found {len(unique)} recordings in split={args.split}\n"
        )
        for i in order[:100]:
            print(
                f"{unique[i]:15s} | "
                f"{counts[i]:5d} windows"
            )
        return

    if args.recording is None:
        order = np.argsort(-counts)
        args.recording = unique[order[0]]
        print(
            f"[INFO] No recording specified."
            f" Using longest available night:"
            f" {args.recording}"
        )

    selected = np.where(
        rec_ids == args.recording
    )[0]

    if len(selected) == 0:
        raise ValueError(
            f"Recording {args.recording} not found "
            f"in split={args.split}"
        )

    selected = selected[np.argsort(
        win_ids[selected]
    )]

    if args.max_windows is not None:
        selected = selected[:args.max_windows]

    print(
        f"[INFO] Using recording={args.recording}"
        f" | split={args.split}"
        f" | windows={len(selected)}"
    )

    channel_dataset = ChannelDataset(
        source, "eeg"
    )

    loader = DataLoader(
        Subset(channel_dataset, selected.tolist()),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    backbone, back_ckpt = load_backbone(
        args.backbone, device
    )
    agent, _ = load_agent(
        args.agent_checkpoint, device
    )

    mean = back_ckpt["normalization"]["mean"]
    std = back_ckpt["normalization"]["std"]

    all_levels = []
    all_used = []
    all_x = []
    all_recon = []

    with torch.inference_mode():
        for raw in loader:
            x = normalize(raw, mean, std, device)

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                codes, _ = backbone.tokenize(x)

            codes = codes.float()

            levels, used = rollout_with_threshold(
                agent,
                codes,
                budget=args.budget,
                stop_threshold=args.stop_threshold,
            )

            recon = decode_mixed(
                backbone.reconstructor,
                codes,
                levels,
            ).float()

            all_levels.append(
                levels.cpu().numpy()
            )
            all_used.append(
                used.cpu().numpy()
            )
            all_x.append(
                x.cpu().numpy()
            )
            all_recon.append(
                recon.cpu().numpy()
            )

    levels = np.concatenate(
        all_levels, axis=0
    )  # [N,16]
    used = np.concatenate(
        all_used, axis=0
    )   # [N]
    signals = np.concatenate(
        all_x, axis=0
    )   # [N,1,3000]
    recons = np.concatenate(
        all_recon, axis=0
    )  # [N,1,3000]

    n_windows = len(levels)
    hours = (
        np.arange(n_windows) * 30.0 / 3600.0
    )

    coarse_count = (levels == 0).sum(axis=1)
    mid_count = (levels == 1).sum(axis=1)
    fine_count = (levels == 2).sum(axis=1)

    avg_used = float(used.mean())
    pct_coarse = float((levels == 0).mean() * 100)
    pct_mid = float((levels == 1).mean() * 100)
    pct_fine = float((levels == 2).mean() * 100)

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{args.recording}_"
        f"b{args.budget}_"
        f"thr{args.stop_threshold:.3f}"
    )

    # --------------------------------------------------
    # Save CSV summary
    # --------------------------------------------------
    csv_path = outdir / f"{stem}_windows.csv"

    header = [
        "recording",
        "window_in_night",
        "time_hours",
        "used_tokens",
        "n_coarse_regions",
        "n_mid_regions",
        "n_fine_regions",
    ] + [f"region_{i:02d}" for i in range(16)]

    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for i in range(n_windows):
            row = [
                args.recording,
                str(i),
                f"{hours[i]:.6f}",
                str(int(used[i])),
                str(int(coarse_count[i])),
                str(int(mid_count[i])),
                str(int(fine_count[i])),
            ] + [str(int(v)) for v in levels[i]]
            f.write(",".join(row) + "\n")

    # --------------------------------------------------
    # Summary figure
    # --------------------------------------------------
    cmap = ListedColormap([
        LEVEL_COLORS[0],
        LEVEL_COLORS[1],
        LEVEL_COLORS[2],
    ])
    norm = BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5],
        ncolors=3,
    )

    fig = plt.figure(
        figsize=(18, 12)
    )
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 1.0, 2.0],
        hspace=0.25,
    )

    # Panel 1: used tokens
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        hours, used,
        linewidth=1.2,
        color="#6F4E7C",
    )
    for y in [16, 32, 48, 64, 80, 96, 112, 128]:
        ax1.axhline(
            y,
            linestyle="--",
            linewidth=0.7,
            alpha=0.25,
            color="gray",
        )
    ax1.set_ylabel("Used tokens")
    ax1.set_title(
        f"Adaptive V3 night summary | {args.recording} | split={args.split}\n"
        f"budget={args.budget} | stop_threshold={args.stop_threshold:.3f} | "
        f"avg_used={avg_used:.1f}"
    )
    ax1.grid(alpha=0.2)
    ax1.set_xlim(hours[0], hours[-1] if len(hours) > 1 else 1)

    # Panel 2: counts of coarse/mid/fine regions
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.stackplot(
        hours,
        coarse_count,
        mid_count,
        fine_count,
        colors=[
            LEVEL_COLORS[0],
            LEVEL_COLORS[1],
            LEVEL_COLORS[2],
        ],
        alpha=0.9,
        labels=["coarse", "mid", "fine"],
    )
    ax2.set_ylabel("Number of regions\n(out of 16)")
    ax2.legend(
        loc="upper right",
        ncol=3,
        frameon=False,
    )
    ax2.grid(alpha=0.2)

    # Panel 3: heatmap
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    image = ax3.imshow(
        levels.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[
            hours[0],
            hours[-1] + 30 / 3600 if len(hours) > 1 else 30 / 3600,
            0,
            30,
        ],
    )
    ax3.set_xlabel("Time in night (hours)")
    ax3.set_ylabel("Local time inside each 30 s window (s)")
    ax3.set_yticks([
        0, 5, 10, 15, 20, 25, 30
    ])
    ax3.set_title(
        "Granularity map across the night"
    )

    cbar = fig.colorbar(
        image,
        ax=ax3,
        fraction=0.015,
        pad=0.01,
        ticks=[0, 1, 2],
    )
    cbar.ax.set_yticklabels(
        ["coarse", "mid", "fine"]
    )

    fig.text(
        0.5,
        0.01,
        (
            f"Region distribution over all segments: "
            f"coarse={pct_coarse:.1f}% | "
            f"mid={pct_mid:.1f}% | "
            f"fine={pct_fine:.1f}%"
        ),
        ha="center",
        fontsize=11,
    )

    summary_path = outdir / f"{stem}_summary.png"
    fig.savefig(
        summary_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # Example windows figure
    # --------------------------------------------------
    example_ids = pick_example_indices(used)
    example_titles = [
        "low-budget example",
        "median-budget example",
        "high-budget example",
    ]

    fig, axes = plt.subplots(
        len(example_ids),
        1,
        figsize=(18, 9),
        sharex=True,
    )

    if len(example_ids) == 1:
        axes = [axes]

    t = np.arange(3000) / 100.0

    for ax, idx, title in zip(
        axes,
        example_ids,
        example_titles,
    ):
        x = signals[idx, 0]
        y = recons[idx, 0]
        lv = levels[idx]

        # Colored background
        for r in range(16):
            start = r * 30.0 / 16.0
            end = (r + 1) * 30.0 / 16.0
            ax.axvspan(
                start,
                end,
                color=LEVEL_COLORS[int(lv[r])],
                alpha=0.14,
                linewidth=0,
            )

        ax.plot(
            t, x,
            color="black",
            linewidth=0.8,
            label="Ground truth EEG",
        )
        ax.plot(
            t, y,
            color="#7B3F98",
            linewidth=0.8,
            alpha=0.95,
            label="Agent reconstruction",
        )

        c = int((lv == 0).sum())
        m = int((lv == 1).sum())
        f = int((lv == 2).sum())

        ax.set_ylabel("Norm. EEG")
        ax.set_title(
            f"{title} | window={idx} | "
            f"time={hours[idx]:.2f} h | "
            f"used={used[idx]} tokens | "
            f"C/M/F regions = {c}/{m}/{f}"
        )
        ax.grid(alpha=0.18)

    axes[-1].set_xlabel("Time inside 30 s window (s)")

    legend_handles = [
        Patch(
            facecolor=LEVEL_COLORS[0],
            edgecolor="none",
            alpha=0.3,
            label="coarse region",
        ),
        Patch(
            facecolor=LEVEL_COLORS[1],
            edgecolor="none",
            alpha=0.3,
            label="mid region",
        ),
        Patch(
            facecolor=LEVEL_COLORS[2],
            edgecolor="none",
            alpha=0.3,
            label="fine region",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            alpha=0.0,
            label="black line: ground truth EEG",
        ),
        Patch(
            facecolor="white",
            edgecolor="#7B3F98",
            alpha=0.0,
            label="purple line: agent reconstruction",
        ),
    ]

    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=2,
    )

    examples_path = outdir / f"{stem}_examples.png"
    fig.savefig(
        examples_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved files:")
    print(" ", summary_path)
    print(" ", examples_path)
    print(" ", csv_path)

if __name__ == "__main__":
    main()
