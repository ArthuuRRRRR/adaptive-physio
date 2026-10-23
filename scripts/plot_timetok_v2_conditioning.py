import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_physio.data.preprocessed_dataset import PreprocessedPSGDataset
from adaptive_physio.models.timetok_physio_v2 import (
    PhysioTokenizerV2, LEVELS
)
from train_timetok_physio_v2 import ChannelDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/timetok_v2_eeg_ae/best.pt",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Index within a sampled validation subset"
    )
    parser.add_argument(
        "--n-sample-windows",
        type=int,
        default=32,
        help="How many validation windows to sample uniformly"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation/timetok_v2_eeg/conditioning_plot.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required")

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = PhysioTokenizerV2().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean = ckpt["normalization"]["mean"]
    std = ckpt["normalization"]["std"]

    source = PreprocessedPSGDataset(split="val")
    dataset = ChannelDataset(source, "eeg")

    n = min(args.n_sample_windows, len(dataset))
    indices = torch.linspace(
        0, len(dataset) - 1, steps=n
    ).long().tolist()

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=min(8, n),
        num_workers=0,
    )

    all_x = []
    all_codes = []

    with torch.inference_mode():
        for raw in loader:
            x = ((raw.to(device) - mean) / std).clamp(-8, 8)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                codes, _ = model.tokenize(x)
            all_x.append(x.float().cpu())
            all_codes.append(codes.float().cpu())

    x_all = torch.cat(all_x, dim=0)
    codes_all = torch.cat(all_codes, dim=0)

    idx = args.window_index
    if idx >= len(x_all):
        raise ValueError(f"window-index {idx} out of range (max {len(x_all)-1})")

    x = x_all[idx:idx+1].to(device)
    codes = codes_all[idx:idx+1].to(device)

    # shuffled source: previous sample cyclically
    shuffled_idx = (idx - 1) % len(codes_all)
    shuffled_codes = codes_all[shuffled_idx:shuffled_idx+1].to(device)

    recon = {}
    with torch.inference_mode():
        for k in LEVELS:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                recon[(k, "correct")] = model.reconstructor(codes, k).float().cpu()[0, 0]
                recon[(k, "shuffled")] = model.reconstructor(shuffled_codes, k).float().cpu()[0, 0]

    original = x.float().cpu()[0, 0]

    hz = 100
    n_points = min(int(args.seconds * hz), original.numel())
    t = torch.arange(n_points).numpy() / hz

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, sharey=True)

    for row, k in enumerate(LEVELS):
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]

        ax_left.plot(t, original[:n_points].numpy(), linewidth=1.0, label="Original")
        ax_left.plot(t, recon[(k, "correct")][:n_points].numpy(), linewidth=1.0, label=f"{k} correct")
        ax_left.set_title(f"{k} tokens - correct conditioning")
        ax_left.set_ylabel("EEG (normalized)")
        ax_left.grid(alpha=0.25)
        if row == 0:
            ax_left.legend()

        ax_right.plot(t, original[:n_points].numpy(), linewidth=1.0, label="Original")
        ax_right.plot(t, recon[(k, "shuffled")][:n_points].numpy(), linewidth=1.0, label=f"{k} shuffled")
        ax_right.set_title(f"{k} tokens - shuffled conditioning")
        ax_right.grid(alpha=0.25)
        if row == 0:
            ax_right.legend()

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    fig.suptitle(
        f"TimeTok-Physio V2 conditioning check | checkpoint step {ckpt['step']}",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
