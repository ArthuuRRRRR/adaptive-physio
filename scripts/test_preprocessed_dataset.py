
from torch.utils.data import DataLoader

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)


dataset = PreprocessedPSGDataset(
    split="train",
    max_records=2,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=0,
    shuffle=False,
)

batch = next(iter(loader))

print("\nFIRST BATCH")

for modality in ("eeg", "ecg", "resp"):
    x = batch[modality]

    assert x.shape == (4, 1, 3000)

    print(
        f"{modality.upper():4s} | "
        f"shape={tuple(x.shape)} | "
        f"mean={x.mean().item():.5f} | "
        f"std={x.std().item():.5f} | "
        f"min={x.min().item():.5f} | "
        f"max={x.max().item():.5f}"
    )

print("Recordings:", batch["recording"])
print("Window indices:", batch["window_index"].tolist())

print("\nALL TESTS PASSED")
