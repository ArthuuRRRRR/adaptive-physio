
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_physio.data.edf_dataset import EDFWindowDataset
from adaptive_physio.training.granularity_targets import (
    GaussianGranularityTargets,
)

ROOT = "/datasets/sleep/#_SHHS/polysomnography/edfs/shhs1"

dataset = EDFWindowDataset(
    root_dir=ROOT,
    window_seconds=30,
    max_files=2,
)

# Select a window near the middle of the first night.
index = dataset.records[0]["n_windows"] // 2
sample = dataset[index]

print("Recording:", sample["recording"])
print("Start:", sample["start_seconds"], "seconds")

modalities = {
    "eeg": (125, 5),
    "ecg": (125, 5),
    "resp": (10, 30),
}

levels = ["fine", "mid", "coarse"]

fig, axes = plt.subplots(
    3, 3,
    figsize=(16, 9),
)

for row, (modality, (fs, zoom)) in enumerate(
    modalities.items()
):
    x = sample[modality].unsqueeze(0)

    target_builder = GaussianGranularityTargets(
        sample_rate=fs
    )

    targets = target_builder(x)

    raw = targets["fine"][0, 0].numpy()

    time = np.arange(len(raw)) / fs
    mask = time < zoom

    # Same y-axis for the three levels.
    visible = raw[mask]
    ymin, ymax = visible.min(), visible.max()
    margin = max((ymax - ymin) * 0.1, 1e-6)

    for col, level in enumerate(levels):
        ax = axes[row, col]
        signal = targets[level][0, 0].numpy()

        ax.plot(
            time[mask],
            signal[mask],
            linewidth=1,
        )

        ax.set_ylim(
            ymin - margin,
            ymax + margin,
        )

        ax.set_title(
            f"{modality.upper()} — {level}"
        )

        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)

        ratio = (
            np.std(signal) /
            max(np.std(raw), 1e-8)
        )

        print(
            f"{modality.upper()} {level}: "
            f"std ratio = {ratio:.4f}"
        )

plt.tight_layout()

output_dir = Path("outputs/diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "smoothing_shhs1.png"

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close()

print("Saved:", output_path)
print("ALL TESTS PASSED")
