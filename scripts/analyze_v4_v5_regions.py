"""Compare V4 and V5 refinement decisions at matched token budgets."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from adaptive_physio.models.timetok_physio_v3 import (
    reconstruction_stats,
)
from evaluate_timetok_v3_adaptive import decode_mixed
from train_temporal_agent_v4 import load_agent
from train_temporal_agent_physio_v5 import (
    setup,
    data_loader,
    normalize,
    get_codes,
    physio_error,
)


@torch.no_grad()
def fixed_budget(agent, codes, budget):
    """Same greedy allocation rule for both agents; STOP disabled."""
    b = codes.shape[0]
    device = codes.device

    levels = torch.zeros(
        b, 16, device=device, dtype=torch.long
    )

    used = torch.full(
        (b,), 16, device=device, dtype=torch.long
    )

    for _ in range(32):
        scores = agent(codes, levels).float()

        if scores.shape != (b, 16, 2):
            raise ValueError(
                f"Unexpected agent scores: {scores.shape}"
            )

        mid_valid = (
            (levels == 0)
            & ((used + 3) <= budget)[:, None]
        )

        fine_valid = (
            (levels == 1)
            & ((used + 4) <= budget)[:, None]
        )

        valid = torch.stack(
            (mid_valid, fine_valid), dim=-1
        )

        masked = scores.masked_fill(
            ~valid, float("-inf")
        )

        value, choice = masked.reshape(
            b, 32
        ).max(dim=1)

        active = torch.isfinite(value)

        if not bool(active.any()):
            break

        region = choice // 2
        kind = choice % 2

        rows = torch.arange(b, device=device)

        levels[rows, region] += active.long()

        cost = torch.where(
            kind == 0, 3, 4
        )

        used += cost * active.long()

    return levels, used


def regional_features(x):
    """Basic spectral diagnostics for 16 temporal regions."""
    signal = x.detach().float().cpu().numpy()[:, 0]

    features = np.zeros(
        (len(signal), 16, 5), dtype=np.float32
    )

    for region in range(16):
        a = round(region * 3000 / 16)
        z = round((region + 1) * 3000 / 16)

        segment = signal[:, a:z]

        rms = np.sqrt(
            np.mean(segment ** 2, axis=1)
        )

        line = np.mean(
            np.abs(np.diff(segment, axis=1)),
            axis=1,
        )

        centered = segment - segment.mean(
            axis=1, keepdims=True
        )

        window = np.hanning(segment.shape[1])

        power = np.abs(
            np.fft.rfft(
                centered * window[None],
                axis=1,
            )
        ) ** 2

        freq = np.fft.rfftfreq(
            segment.shape[1], d=1 / 100
        )

        total = power[
            :, (freq >= 0.5) & (freq <= 30)
        ].sum(axis=1) + 1e-12

        def fraction(low, high):
            return (
                power[
                    :, (freq >= low) & (freq < high)
                ].sum(axis=1) / total
            )

        features[:, region, 0] = rms
        features[:, region, 1] = fraction(0.5, 4)
        features[:, region, 2] = fraction(11, 16)
        features[:, region, 3] = fraction(16, 30)
        features[:, region, 4] = line

    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--budgets",
        type=int, nargs="+",
        default=[64, 80, 96],
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/evaluation/timetok_v3/"
            "physio_v5/region_comparison"
        ),
    )

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    config = SimpleNamespace(
        seed=42,
        backbone=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
        singlem="third_party/SingLEM",
        physio_weight=5.0,
    )

    device, backbone, ck, teacher = setup(config)

    v4, _ = load_agent(
        "outputs/checkpoints/"
        "timetok_v3_agent_cost_eeg/best.pt",
        device,
    )

    v5, _ = load_agent(
        "outputs/checkpoints/"
        "timetok_v3_agent_physio_beta5/best.pt",
        device,
    )

    v4.eval()
    v5.eval()

    loader, n = data_loader(
        "val", "eeg",
        args.windows,
        args.batch_size,
        0,
        42,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [
        "RMS",
        "Delta fraction",
        "Sigma fraction",
        "Beta fraction",
        "Line length",
    ]

    for budget in args.budgets:
        print(
            f"\n========== BUDGET {budget} ==========",
            flush=True,
        )

        collected = {
            k: [] for k in (
                "used_v4", "used_v5",
                "mae_v4", "mae_v5",
                "corr_v4", "corr_v5",
                "physio_v4", "physio_v5",
                "level_agreement",
                "regions_different",
                "fine_jaccard",
                "fine_v4", "fine_v5",
                "v4_only", "v5_only",
            )
        }

        levels_v4_all = []
        levels_v5_all = []
        v4_only_features = []
        v5_only_features = []

        for step, raw in enumerate(loader, 1):
            x = normalize(
                raw, ck["normalization"], device
            )

            codes = get_codes(backbone, x)

            with torch.no_grad():
                lv4, used4 = fixed_budget(
                    v4, codes, budget
                )

                lv5, used5 = fixed_budget(
                    v5, codes, budget
                )

                y4 = decode_mixed(
                    backbone.reconstructor,
                    codes, lv4
                ).float()

                y5 = decode_mixed(
                    backbone.reconstructor,
                    codes, lv5
                ).float()

                original = teacher(
                    x, ck["normalization"]
                )

                d4 = physio_error(
                    teacher(y4, ck["normalization"]),
                    original,
                )

                d5 = physio_error(
                    teacher(y5, ck["normalization"]),
                    original,
                )

                mae4, _, corr4 = reconstruction_stats(
                    y4, x
                )

                mae5, _, corr5 = reconstruction_stats(
                    y5, x
                )

            a = lv4.cpu().numpy()
            b = lv5.cpu().numpy()

            f4 = a == 2
            f5 = b == 2

            union = (f4 | f5).sum(axis=1)
            intersection = (f4 & f5).sum(axis=1)

            jaccard = np.where(
                union > 0,
                intersection / np.maximum(union, 1),
                1.0,
            )

            features = regional_features(x)

            only4 = f4 & ~f5
            only5 = f5 & ~f4

            if only4.any():
                v4_only_features.append(
                    features[only4]
                )

            if only5.any():
                v5_only_features.append(
                    features[only5]
                )

            collected["used_v4"].extend(
                used4.cpu().tolist()
            )
            collected["used_v5"].extend(
                used5.cpu().tolist()
            )

            for key, value in (
                ("mae_v4", mae4),
                ("mae_v5", mae5),
                ("corr_v4", corr4),
                ("corr_v5", corr5),
                ("physio_v4", d4),
                ("physio_v5", d5),
            ):
                collected[key].extend(
                    value.detach().cpu().tolist()
                )

            collected["level_agreement"].extend(
                (a == b).mean(axis=1).tolist()
            )

            collected["regions_different"].extend(
                (a != b).sum(axis=1).tolist()
            )

            collected["fine_jaccard"].extend(
                jaccard.tolist()
            )

            collected["fine_v4"].extend(
                f4.sum(axis=1).tolist()
            )

            collected["fine_v5"].extend(
                f5.sum(axis=1).tolist()
            )

            collected["v4_only"].extend(
                only4.sum(axis=1).tolist()
            )

            collected["v5_only"].extend(
                only5.sum(axis=1).tolist()
            )

            levels_v4_all.append(a)
            levels_v5_all.append(b)

            if step % 16 == 0:
                print(
                    f"Processed {min(step * args.batch_size, n)}/{n}",
                    flush=True,
                )

        data = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in collected.items()
        }

        def avg(key):
            return float(data[key].mean())

        print("\n--- Token allocation ---")
        print(f"V4 tokens: {avg('used_v4'):.2f}")
        print(f"V5 tokens: {avg('used_v5'):.2f}")
        print(
            "Equal token counts:",
            f"{np.mean(data['used_v4'] == data['used_v5']):.1%}"
        )

        print("\n--- Reconstruction ---")
        print(
            f"V4 MAE={avg('mae_v4'):.5f} "
            f"Corr={avg('corr_v4'):.5f} "
            f"SingLEM={avg('physio_v4'):.6f}"
        )
        print(
            f"V5 MAE={avg('mae_v5'):.5f} "
            f"Corr={avg('corr_v5'):.5f} "
            f"SingLEM={avg('physio_v5'):.6f}"
        )

        print("\n--- Decision differences ---")
        print(
            "Level agreement:",
            f"{avg('level_agreement'):.1%}"
        )
        print(
            "Different regions/window:",
            f"{avg('regions_different'):.2f}/16"
        )
        print(
            "Fine Jaccard:",
            f"{avg('fine_jaccard'):.3f}"
        )
        print(
            "V4 fine regions/window:",
            f"{avg('fine_v4'):.2f}"
        )
        print(
            "V5 fine regions/window:",
            f"{avg('fine_v5'):.2f}"
        )
        print(
            "V4-only fine/window:",
            f"{avg('v4_only'):.2f}"
        )
        print(
            "V5-only fine/window:",
            f"{avg('v5_only'):.2f}"
        )

        print("\n--- Exclusive fine regions ---")

        if v4_only_features and v5_only_features:
            x4 = np.concatenate(v4_only_features)
            x5 = np.concatenate(v5_only_features)

            print("V4-only regions:", len(x4))
            print("V5-only regions:", len(x5))
            print(
                f"{'Feature':18} | "
                f"{'V4-only':>10} | {'V5-only':>10}"
            )

            for i, name in enumerate(names):
                print(
                    f"{name:18} | "
                    f"{x4[:, i].mean():10.4f} | "
                    f"{x5[:, i].mean():10.4f}"
                )
        else:
            print("Not enough exclusive fine regions.")

        np.savez_compressed(
            output_dir / f"budget_{budget}.npz",
            **data,
            levels_v4=np.concatenate(levels_v4_all),
            levels_v5=np.concatenate(levels_v5_all),
        )

        print(
            "Saved:",
            output_dir / f"budget_{budget}.npz",
            flush=True,
        )


if __name__ == "__main__":
    main()
