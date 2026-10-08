
import torch
from torch.utils.data import DataLoader

from adaptive_physio.data.edf_dataset import EDFWindowDataset


root = "/datasets/sleep/#_SHHS/polysomnography/edfs/shhs1"

dataset = EDFWindowDataset(
    root_dir=root,
    window_seconds=30,
    max_files=2,
)

print("Recordings:", len(dataset.records))
print("Total windows:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))

for modality, expected_length in [
    ("eeg", 3750),
    ("ecg", 3750),
    ("resp", 300),
]:
    signal = batch[modality]

    print(
        f"{modality.upper()}:",
        tuple(signal.shape),
        "| mean:", round(signal.mean().item(), 4),
        "| std:", round(signal.std().item(), 4),
    )

    assert signal.shape == (4, 1, expected_length)
    assert torch.isfinite(signal).all()

print("Recordings:", batch["recording"])
print("Start times:", batch["start_seconds"].tolist())

print("ALL TESTS PASSED")
