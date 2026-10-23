import torch
import torch.nn.functional as F
from types import SimpleNamespace

from evaluate_timetok_v3_adaptive import decode_mixed
from train_temporal_agent_physio_v5 import (
    setup, data_loader, normalize, get_codes,
)
from train_temporal_agent_physio_v6 import region_weights


@torch.inference_mode()
def main():
    cfg = SimpleNamespace(
        seed=42,
        backbone=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
        singlem="third_party/SingLEM",
        physio_weight=5.0,
    )

    device, backbone, ck, teacher = setup(cfg)

    weights = region_weights(device, sigma=1.25)

    times = torch.tensor(
        [
            10 * (i // 13) + 0.75 * (i % 13) + 0.5
            for i in range(39)
        ],
        device=device,
    )

    loader, _ = data_loader(
        "val", "eeg", 64, 4, 0, 42
    )

    results = {
        "coarse_to_mid": [],
        "mid_to_fine": [],
    }

    for raw in loader:
        x = normalize(
            raw, ck["normalization"], device
        )
        codes = get_codes(backbone, x)

        for level, name in [
            (0, "coarse_to_mid"),
            (1, "mid_to_fine"),
        ]:
            states = torch.full(
                (len(x), 16), level,
                device=device, dtype=torch.long,
            )

            before = decode_mixed(
                backbone.reconstructor, codes, states
            ).float()

            base_features = F.normalize(
                teacher(before, ck["normalization"]).float(),
                dim=-1,
            )

            for r in range(16):
                modified = states.clone()
                modified[:, r] += 1

                after = decode_mixed(
                    backbone.reconstructor, codes, modified
                ).float()

                new_features = F.normalize(
                    teacher(after, ck["normalization"]).float(),
                    dim=-1,
                )

                change = (
                    1.0 -
                    (base_features * new_features).sum(-1)
                ).clamp_min(0)

                center = (r + 0.5) * 30 / 16

                near = (
                    change * weights[r][None]
                ).sum(dim=1)

                far_mask = (
                    (times - center).abs() >= 5.0
                )
                far = change[:, far_mask].mean(dim=1)

                peak_times = times[
                    change.argmax(dim=1)
                ]
                peak_near = (
                    (peak_times - center).abs() <= 2.0
                ).float()

                delta = (
                    (after - before).float() ** 2
                ).sum(dim=1)

                start = round(r * 3000 / 16)
                end = round((r + 1) * 3000 / 16)

                wave_local = (
                    delta[:, start:end].sum(dim=1)
                    / delta.sum(dim=1).clamp_min(1e-12)
                )

                for i in range(len(x)):
                    results[name].append([
                        near[i].item(),
                        far[i].item(),
                        peak_near[i].item(),
                        wave_local[i].item(),
                    ])

    import numpy as np

    for name, rows in results.items():
        a = np.asarray(rows)
        near, far, peak, waveform = a.mean(axis=0)

        print(f"\n=== {name.upper()} ===")
        print(f"Comparisons: {len(a)}")
        print(f"SingLEM near change: {near:.8f}")
        print(f"SingLEM far change:  {far:.8f}")
        print(
            "Near/far ratio:",
            f"{near / max(far, 1e-12):.3f}"
        )
        print(
            "Largest embedding change near region:",
            f"{peak:.1%}"
        )
        print(
            "Waveform change inside refined region:",
            f"{waveform:.1%}"
        )


if __name__ == "__main__":
    main()
