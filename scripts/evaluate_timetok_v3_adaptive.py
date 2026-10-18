import argparse
import csv
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from adaptive_physio.data.preprocessed_dataset import PreprocessedPSGDataset
from adaptive_physio.models.timetok_physio_v3 import (
    PhysioTokenizerV3, reconstruction_stats
)
from train_timetok_physio_v2 import ChannelDataset

BUDGETS = (16, 32, 48, 64, 80, 96, 112, 128)


def decode_mixed(dec, codes, levels):
    """
    levels[B,16]:
        0 = coarse: 1 token
        1 = mid:    4 tokens
        2 = fine:   8 tokens
    """
    b = codes.shape[0]

    coarse = dec.from_coarse(codes[:, :16])
    mid = coarse.repeat_interleave(4, dim=1)

    dmid = dec.from_mid(
        codes[:, 16:64].reshape(b, 16, 3, 6)
    )

    dmid = torch.einsum(
        "bjmd,im->bjid",
        dmid.float(),
        dec.mid_basis
    ).reshape(b, 64, dec.channels)

    mid_mask = (
        (levels >= 1)
        .repeat_interleave(4, dim=1)
        .unsqueeze(-1)
    )

    mid = mid + dec.mid_gain * dmid * mid_mask

    fine = mid.repeat_interleave(2, dim=1)

    dfine = dec.from_fine(
        codes[:, 64:128]
    ) / math.sqrt(2)

    pair = torch.stack(
        (dfine, -dfine), dim=2
    ).reshape(b, 128, dec.channels)

    fine_mask = (
        (levels >= 2)
        .repeat_interleave(8, dim=1)
        .unsqueeze(-1)
    )

    fine = fine + dec.fine_gain * pair * fine_mask

    dense = F.interpolate(
        fine.transpose(1, 2),
        size=dec.length,
        mode="nearest"
    )

    dense = torch.cat(
        (
            dense,
            dec.pos.expand(b, -1, -1).to(dense.dtype)
        ),
        dim=1
    )

    return dec.decode(dense)


def regional_error(pred, real):
    values = []
    length = real.shape[-1]

    for r in range(16):
        start = round(r * length / 16)
        end = round((r + 1) * length / 16)

        error = (
            pred[..., start:end]
            - real[..., start:end]
        ).abs().sum((1, 2))

        values.append(error)

    return torch.stack(values, dim=1)


def allocate(budget, policy, gains_mid, gains_fine, seed):
    """
    Random: random valid refinements.
    Oracle: highest target-informed gain/token.
    """
    rng = random.Random(seed)

    levels = [0] * 16
    used = 16

    while True:
        actions = []

        for region in range(16):
            if levels[region] == 0 and used + 3 <= budget:
                actions.append((
                    gains_mid[region] / 3,
                    region, 3
                ))

            elif levels[region] == 1 and used + 4 <= budget:
                actions.append((
                    gains_fine[region] / 4,
                    region, 4
                ))

        if not actions:
            break

        if policy == "random":
            gain, region, cost = rng.choice(actions)
        else:
            gain, region, cost = max(
                actions, key=lambda a: a[0]
            )

            # Oracle stopping rule.
            if gain <= 0:
                break

        levels[region] += 1
        used += cost

    return levels, used


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/timetok_v3_eeg_ae/best.pt"
    )
    p.add_argument("--windows", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        default="outputs/evaluation/timetok_v3/adaptive_probe.csv"
    )

    args = p.parse_args()

    if args.windows < 1 or args.batch_size < 1:
        p.error("Positive evaluation counts required")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    model = PhysioTokenizerV3().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    source = PreprocessedPSGDataset(split="val")
    dataset = ChannelDataset(
        source, ckpt["modality"]
    )

    n = min(args.windows, len(dataset))

    indices = torch.linspace(
        0, len(dataset) - 1, n
    ).long().tolist()

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )

    mean = ckpt["normalization"]["mean"]
    std = ckpt["normalization"]["std"]

    accum = {
        (policy, budget): [0., 0., 0., 0]
        for policy in ("random", "oracle_greedy")
        for budget in BUDGETS
    }

    uniform = {
        k: [0., 0., 0]
        for k in (16, 64, 128)
    }

    checked = False
    offset = 0

    with torch.inference_mode():
        for raw in loader:
            x = (
                (raw.to(device) - mean) / std
            ).clamp(-8, 8)

            codes, _ = model.tokenize(x)

            predictions = {
                k: model.reconstructor(codes, k)
                for k in (16, 64, 128)
            }

            # Verify the mixed decoder reproduces
            # the original V3 uniform behavior.
            if not checked:
                for idx, k in enumerate((16, 64, 128)):
                    levels = torch.full(
                        (len(x), 16),
                        idx,
                        dtype=torch.long,
                        device=device
                    )

                    mixed = decode_mixed(
                        model.reconstructor,
                        codes, levels
                    )

                    difference = (
                        mixed - predictions[k]
                    ).abs().max().item()

                    assert difference < 1e-4, (
                        f"Decoder mismatch for {k}: "
                        f"{difference}"
                    )

                print(
                    "PASS: mixed decoder matches "
                    "16/64/128 uniform modes",
                    flush=True
                )
                checked = True

            for k, pred in predictions.items():
                mae, _, corr = reconstruction_stats(
                    pred, x
                )
                uniform[k][0] += mae.sum().item()
                uniform[k][1] += corr.sum().item()
                uniform[k][2] += len(x)

            errors = {
                k: regional_error(predictions[k], x)
                    .cpu().tolist()
                for k in (16, 64, 128)
            }

            for policy in ("random", "oracle_greedy"):
                for budget in BUDGETS:
                    plans = []
                    spent = []

                    for j in range(len(x)):
                        gain_mid = [
                            a - b
                            for a, b in zip(
                                errors[16][j],
                                errors[64][j]
                            )
                        ]

                        gain_fine = [
                            a - b
                            for a, b in zip(
                                errors[64][j],
                                errors[128][j]
                            )
                        ]

                        plan, cost = allocate(
                            budget,
                            policy,
                            gain_mid,
                            gain_fine,
                            args.seed
                            + (offset + j) * 1009
                            + budget
                        )

                        plans.append(plan)
                        spent.append(cost)

                    levels = torch.tensor(
                        plans,
                        dtype=torch.long,
                        device=device
                    )

                    pred = decode_mixed(
                        model.reconstructor,
                        codes, levels
                    )

                    mae, _, corr = reconstruction_stats(
                        pred, x
                    )

                    result = accum[(policy, budget)]
                    result[0] += mae.sum().item()
                    result[1] += corr.sum().item()
                    result[2] += sum(spent)
                    result[3] += len(x)

            offset += len(x)

            if offset % 64 == 0 or offset == n:
                print(
                    f"Evaluated {offset}/{n} windows",
                    flush=True
                )

    print("\n=== UNIFORM BASELINES ===")
    print("Tokens | MAE | Corr")

    for k in (16, 64, 128):
        mae, corr, count = uniform[k]
        print(
            f"{k:6d} | "
            f"{mae/count:.4f} | "
            f"{corr/count:.4f}"
        )

    print("\n=== ADAPTIVE GRANULARITY ===")
    print("Policy | Budget | Used | MAE | Corr")

    rows = []

    for budget in BUDGETS:
        for policy in ("random", "oracle_greedy"):
            mae, corr, used, count = (
                accum[(policy, budget)]
            )

            row = {
                "policy": policy,
                "budget": budget,
                "avg_used": round(used / count, 2),
                "mae": round(mae / count, 5),
                "corr": round(corr / count, 5)
            }

            rows.append(row)

            print(
                f"{policy:14} | "
                f"{budget:3d} | "
                f"{row['avg_used']:5.1f} | "
                f"{row['mae']:.4f} | "
                f"{row['corr']:.4f}"
            )

    path = Path(args.output)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", path)
    print(
        "NOTE: oracle_greedy uses the real EEG "
        "and is not a globally optimal oracle."
    )


if __name__ == "__main__":
    main()
