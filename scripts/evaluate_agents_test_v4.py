"""Frozen held-out evaluation of adaptive EEG controllers."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from train_temporal_agent_v3 import (
    load_backbone,
    data_loader,
    normalize,
    rollout,
    regional_mae,
)
from train_temporal_agent_v4 import (
    load_agent,
    rollout_lambdas,
)
from evaluate_timetok_v3_adaptive import decode_mixed
from adaptive_physio.models.timetok_physio_v3 import (
    reconstruction_stats,
)

BANDS = (
    (0.5, 4),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 30),
)


def spectral_band_errors(pred, real):
    p = pred.float().flatten(1)
    x = real.float().flatten(1)

    n = p.shape[-1]
    window = torch.hann_window(
        n, device=p.device
    )

    pp = torch.fft.rfft(
        p * window, dim=-1
    ).abs().square()

    xx = torch.fft.rfft(
        x * window, dim=-1
    ).abs().square()

    freqs = torch.fft.rfftfreq(
        n, d=0.01, device=p.device
    )

    errors = []

    for low, high in BANDS:
        mask = (
            (freqs >= low) & (freqs < high)
        )

        power_pred = pp[:, mask].sum(
            -1
        ).clamp_min(1e-8)

        power_real = xx[:, mask].sum(
            -1
        ).clamp_min(1e-8)

        errors.append(
            (
                10 * torch.log10(
                    power_pred / power_real
                )
            ).abs()
        )

    return torch.stack(errors, dim=-1)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--split",
        choices=("val", "test"),
        default="test"
    )
    p.add_argument(
        "--windows", type=int, default=2048
    )
    p.add_argument(
        "--batch-size", type=int, default=16
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--backbone",
        default="outputs/checkpoints/timetok_v3_eeg_ae/frozen_for_agent.pt"
    )
    p.add_argument(
        "--old-agent",
        default="outputs/checkpoints/timetok_v3_agent_eeg/best.pt"
    )
    p.add_argument(
        "--cost-agent",
        default="outputs/checkpoints/timetok_v3_agent_cost_eeg/best.pt"
    )
    p.add_argument(
        "--output",
        default="outputs/evaluation/timetok_v3/final_test/frozen_test_comparison.csv"
    )

    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0")

    backbone, bc = load_backbone(
        args.backbone, device
    )
    old_agent, old_ckpt = load_agent(
        args.old_agent, device
    )
    cost_agent, cost_ckpt = load_agent(
        args.cost_agent, device
    )

    for checkpoint in (old_ckpt, cost_ckpt):
        if checkpoint["backbone_step"] != bc["step"]:
            raise ValueError(
                "Agent/backbone checkpoint mismatch"
            )

    old_agent.eval()
    cost_agent.eval()

    loader, n = data_loader(
        args.split,
        "eeg",
        args.windows,
        args.batch_size,
        2,
        args.seed,
    )

    rng = torch.Generator(
        device=device
    ).manual_seed(args.seed + 2027)

    totals = {}

    def add(name, reconstruction, original, used):
        mae, _, corr = reconstruction_stats(
            reconstruction, original
        )

        bands = spectral_band_errors(
            reconstruction, original
        )

        if isinstance(used, int):
            token_total = float(
                used * len(original)
            )
        else:
            token_total = used.sum().item()

        if name not in totals:
            totals[name] = np.zeros(
                9, dtype=np.float64
            )

        totals[name] += np.array([
            mae.sum().item(),
            corr.sum().item(),
            token_total,
            *bands.sum(0).tolist(),
            len(original),
        ])

    processed = 0

    with torch.inference_mode():
        for raw in loader:
            x = normalize(
                raw, bc["normalization"], device
            )

            with torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                codes, _ = backbone.tokenize(x)

            codes = codes.float()

            uniform = {
                k: backbone.reconstructor(
                    codes, k
                ).float()
                for k in (16, 64, 128)
            }

            for k, reconstruction in uniform.items():
                add(
                    f"uniform_{k}",
                    reconstruction, x, k
                )

            e16, e64, e128 = (
                regional_mae(uniform[k], x)
                for k in (16, 64, 128)
            )

            gain_mid = (e16 - e64) / 3
            gain_fine = (e64 - e128) / 4

            for budget in (64, 80, 96):
                for policy in (
                    "random", "agent", "oracle"
                ):
                    levels, used = rollout(
                        old_agent,
                        codes,
                        budget,
                        policy,
                        gain_mid,
                        gain_fine,
                        rng=rng,
                        allow_stop=False,
                    )

                    reconstruction = decode_mixed(
                        backbone.reconstructor,
                        codes,
                        levels
                    )

                    add(
                        f"{policy}_budget_{budget}",
                        reconstruction, x, used
                    )

            lambdas = [
                0.0004,
                0.0008,
                0.0012,
            ]

            repeated, states, used = (
                rollout_lambdas(
                    cost_agent,
                    codes,
                    lambdas,
                    maximum=128
                )
            )

            reconstruction = decode_mixed(
                backbone.reconstructor,
                repeated,
                states
            )

            reference = x.repeat_interleave(
                len(lambdas), dim=0
            )

            for j, lam in enumerate(lambdas):
                indices = torch.arange(
                    j,
                    len(reconstruction),
                    len(lambdas),
                    device=device
                )

                add(
                    f"cost_stop_lambda_{lam:g}",
                    reconstruction.index_select(
                        0, indices
                    ),
                    reference.index_select(
                        0, indices
                    ),
                    used.index_select(
                        0, indices
                    ),
                )

            processed += len(x)

            if (
                processed % 128 == 0
                or processed == n
            ):
                print(
                    f"Evaluated {processed}/{n}",
                    flush=True
                )

    columns = [
        "policy",
        "windows",
        "avg_used",
        "mae",
        "corr",
        "delta_err_db",
        "theta_err_db",
        "alpha_err_db",
        "sigma_err_db",
        "beta_err_db",
    ]

    rows = []

    for name, values in totals.items():
        count = values[-1]

        row = dict(zip(
            columns,
            [
                name,
                int(count),
                *(values[[2, 0, 1, 3, 4, 5, 6, 7]] / count),
            ]
        ))

        rows.append(row)

    rows.sort(
        key=lambda r: r["avg_used"]
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n=== FROZEN {args.split.upper()} "
        f"EVALUATION — {processed} WINDOWS ==="
    )

    print(
        "Policy | Used | MAE | Corr | "
        "Sigma error (dB)"
    )

    for r in rows:
        print(
            f"{r['policy']:28s} | "
            f"{r['avg_used']:5.1f} | "
            f"{r['mae']:.4f} | "
            f"{r['corr']:.4f} | "
            f"{r['sigma_err_db']:.2f}"
        )

    print("\nSaved:", output)
    print(
        "Band-power errors do not establish "
        "preservation of individual EEG events."
    )


if __name__ == "__main__":
    main()
