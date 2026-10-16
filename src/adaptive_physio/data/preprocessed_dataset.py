
from bisect import bisect_right
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedPSGDataset(Dataset):
    """
    Read synchronized physiological windows from
    Keondo's preprocessed SHHS1 memory maps.

    Output per modality: [1, 3000]

    No normalization or resampling is applied.
    """

    CHANNELS = {
        "eeg": "EEG/C4-A1",
        "ecg": "ECG/ECG",
        "resp": "ABD/ABD",
    }

    def __init__(
        self,
        split="train",
        root="/hdd2/kdpark/sleep_datasets/shhs1/mmap_wav2sleep",
        max_records=None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.root = Path(root) / split
        self.metadata = {}
        self.paths = {}
        self._mmaps = {}

        # Load metadata only, not signal arrays.
        for modality, channel in self.CHANNELS.items():
            folder = self.root / Path(channel).parent
            name = Path(channel).name

            meta_path = folder / f"{name}.pickle"
            mmap_path = folder / f"{name}.mmap"

            # Only load metadata from the trusted lab dataset.
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            shape = tuple(metadata["data_shape"])

            expected_bytes = (
                int(np.prod(shape))
                * np.dtype(np.float32).itemsize
            )

            if mmap_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"Unexpected mmap size: {mmap_path}"
                )

            if len(shape) != 2 or shape[1] != 3000:
                raise ValueError(
                    f"Unexpected shape: {shape}"
                )

            self.metadata[modality] = metadata
            self.paths[modality] = mmap_path

        # Find recordings shared by all modalities.
        common = set.intersection(*[
            set(meta["sig_info"])
            for meta in self.metadata.values()
        ])

        self.records = []
        self.cumulative = []

        total = 0
        skipped = 0

        for recording in sorted(common):
            positions = {}
            lengths = []

            for modality in self.CHANNELS:
                start, end = self.metadata[
                    modality
                ]["sig_info"][recording]["pos"]

                start, end = int(start), int(end)

                positions[modality] = start
                lengths.append(end - start)

            # Avoid pairing windows from records
            # with inconsistent lengths.
            if len(set(lengths)) != 1 or lengths[0] <= 0:
                skipped += 1
                continue

            n = lengths[0]

            self.records.append({
                "recording": recording,
                "positions": positions,
                "windows": n,
            })

            total += n
            self.cumulative.append(total)

            if (
                max_records is not None
                and len(self.records) >= max_records
            ):
                break

        if not self.records:
            raise ValueError("No matching PSG recordings")

        print(
            f"Loaded {len(self.records)} recordings, "
            f"{total} windows"
        )

        print(f"Skipped inconsistent records: {skipped}")

    def __len__(self):
        return self.cumulative[-1]

    def _get_mmap(self, modality):
        # Lazy opening, compatible with DataLoader workers.
        if modality not in self._mmaps:
            shape = tuple(
                self.metadata[modality]["data_shape"]
            )

            self._mmaps[modality] = np.memmap(
                self.paths[modality],
                dtype=np.float32,
                mode="r",
                shape=shape,
            )

        return self._mmaps[modality]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        return state

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        record_idx = bisect_right(
            self.cumulative, index
        )

        previous = (
            self.cumulative[record_idx - 1]
            if record_idx > 0 else 0
        )

        local_idx = index - previous
        record = self.records[record_idx]

        sample = {
            "recording": record["recording"],
            "window_index": local_idx,
        }

        for modality in self.CHANNELS:
            mmap = self._get_mmap(modality)

            position = (
                record["positions"][modality]
                + local_idx
            )

            # Copy only the requested 30-second window.
            signal = np.array(
                mmap[position], copy=True
            )

            if not np.isfinite(signal).all():
                raise ValueError(
                    f"Invalid {modality} signal in "
                    f"{record['recording']}"
                )

            sample[modality] = (
                torch.from_numpy(signal).unsqueeze(0)
            )

        return sample
