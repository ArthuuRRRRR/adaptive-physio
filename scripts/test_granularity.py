
import math
import torch

from adaptive_physio.training.granularity_targets import (
    GaussianGranularityTargets,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

for name, fs in {"EEG": 128, "RESP": 32}.items():
    t = torch.arange(30 * fs, device=device) / fs

    # Low frequency + high frequency components
    x = (
        torch.sin(2 * math.pi * 0.5 * t)
        + 0.3 * torch.sin(2 * math.pi * 10 * t)
    )

    x = x.reshape(1, 1, -1)

    targets = GaussianGranularityTargets(
        sample_rate=fs
    )(x)

    for level, signal in targets.items():
        assert signal.shape == x.shape
        assert torch.isfinite(signal).all()

        print(
            f"{name} {level}: {tuple(signal.shape)}"
        )

    assert torch.allclose(targets["fine"], x)

    coarse_error = (
        (targets["coarse"] - x).square().mean()
    )
    mid_error = (
        (targets["mid"] - x).square().mean()
    )

    assert coarse_error > mid_error

print("ALL TESTS PASSED")

