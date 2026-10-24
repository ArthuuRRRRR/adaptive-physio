"""Evaluate V8 at 64 tokens against Uniform64 without retraining."""

import json
from pathlib import Path

import torch

import train_belief_agent_v8 as v8

from adaptive_physio.models.timetok_physio_v3 import (
    reconstruction_stats,
)
from evaluate_timetok_v3_adaptive import decode_mixed
from train_temporal_agent_physio_v5 import physio_error


@torch.inference_mode()
def evaluate_64(model, backbone, teacher, norm, device, args):

    model.eval()
    backbone.eval()
    teacher.eval()

    keys = [
        "mae_v8_64",
        "mae_uniform64",
        "corr_v8_64",
        "corr_uniform64",
        "singlem_v8_64",
        "singlem_uniform64",
    ]

    totals = {key: 0.0 for key in keys}
    seen = 0

    for raw in v8.loader(
        "val", args.val_windows, args
    ):
        x = v8.normalize(raw, norm, device)
        codes = v8.get_codes(backbone, x)

        # V8: same trained agent, different budget.
        levels, used = v8.choose_levels(
            model, codes, budget=64
        )

        if not bool((used == 64).all()):
            raise RuntimeError(
                "V8 did not reach exactly 64 tokens"
            )

        pred_v8 = decode_mixed(
            backbone.reconstructor, codes, levels
        ).float()

        # Uniform64: all 16 regions at Mid resolution.
        pred_uniform = backbone.reconstructor(
            codes, 64
        ).float()

        mae_v8, _, corr_v8 = reconstruction_stats(
            pred_v8, x
        )

        mae_uni, _, corr_uni = reconstruction_stats(
            pred_uniform, x
        )

        reference = teacher(x, norm)

        singlem_v8 = physio_error(
            teacher(pred_v8, norm),
            reference,
        )

        singlem_uni = physio_error(
            teacher(pred_uniform, norm),
            reference,
        )

        batch = {
            "mae_v8_64": mae_v8,
            "mae_uniform64": mae_uni,
            "corr_v8_64": corr_v8,
            "corr_uniform64": corr_uni,
            "singlem_v8_64": singlem_v8,
            "singlem_uniform64": singlem_uni,
        }

        for key, values in batch.items():
            totals[key] += values.sum().item()

        seen += len(x)

        if seen % 128 < len(x):
            print(
                f"Evaluated {seen}/{args.val_windows}",
                flush=True,
            )

    results = {
        key: value / seen
        for key, value in totals.items()
    }

    results["samples"] = seen
    results["budget"] = 64

    results["singlem_gain_pct"] = (
        100.0
        * (
            results["singlem_uniform64"]
            - results["singlem_v8_64"]
        )
        / max(results["singlem_uniform64"], 1e-8)
    )

    results["mae_gain_pct"] = (
        100.0
        * (
            results["mae_uniform64"]
            - results["mae_v8_64"]
        )
        / max(results["mae_uniform64"], 1e-8)
    )

    results["corr_gain"] = (
        results["corr_v8_64"]
        - results["corr_uniform64"]
    )

    out = Path(
        "outputs/evaluation/timetok_v3/"
        "belief_v8/val_64_vs_uniform64.json"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 55)
    print("V8_64 VS UNIFORM64 — VALIDATION")
    print("=" * 55)

    print(f"Samples: {seen}")
    print(f"Budget: 64 tokens")

    print(
        f"MAE:     V8={results['mae_v8_64']:.6f} "
        f"Uniform={results['mae_uniform64']:.6f}"
    )

    print(
        f"Corr:    V8={results['corr_v8_64']:.6f} "
        f"Uniform={results['corr_uniform64']:.6f}"
    )

    print(
        f"SingLEM: V8={results['singlem_v8_64']:.6f} "
        f"Uniform={results['singlem_uniform64']:.6f}"
    )

    print(
        f"SingLEM improvement: "
        f"{results['singlem_gain_pct']:+.2f}%"
    )

    print("Saved:", out, flush=True)

    # Evaluation handled here; no training and no legacy 80-token evaluation.
    raise SystemExit(0)


# Reuse V8's original checkpoint loading and CLI.
# Replace only the evaluation function.
v8.eval_policy = evaluate_64

if __name__ == "__main__":
    v8.main()
