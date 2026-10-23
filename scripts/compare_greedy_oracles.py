import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch

from adaptive_physio.models.timetok_physio_v3 import reconstruction_stats
from analyze_v4_v5_regions import fixed_budget
from evaluate_timetok_v3_adaptive import decode_mixed
from train_temporal_agent_v4 import load_agent, quality_error
from train_temporal_agent_physio_v5 import (
    setup, data_loader, normalize, get_codes, physio_error,
)


@torch.inference_mode()
def oracle_greedy(backbone, teacher, codes, x, norm, budget, criterion):
    """True marginal greedy gains, with full EEG available to the oracle."""
    states = torch.zeros((1, 16), device=x.device, dtype=torch.long)
    used = 16
    reference = teacher(x, norm) if criterion == "singlem" else None

    def distortion(pred):
        if criterion == "wave":
            return quality_error(
                pred, x.expand(len(pred), -1, -1)
            ).float()

        features = teacher(pred, norm)
        return physio_error(
            features, reference.expand(len(pred), -1, -1)
        ).float()

    pred = decode_mixed(
        backbone.reconstructor, codes, states
    ).float()
    current = distortion(pred)[0]

    for _ in range(32):
        actions, costs = [], []

        for r in range(16):
            level = int(states[0, r])
            if level < 2:
                cost = 3 if level == 0 else 4
                if used + cost <= budget:
                    actions.append(r)
                    costs.append(cost)

        if not actions:
            break

        trials = states.expand(len(actions), -1).clone()

        for i, region in enumerate(actions):
            trials[i, region] += 1

        candidates = decode_mixed(
            backbone.reconstructor,
            codes.expand(len(actions), -1, -1),
            trials,
        ).float()

        next_errors = distortion(candidates)

        cost_tensor = torch.tensor(
            costs, dtype=torch.float32, device=x.device
        )

        gains = (current - next_errors) / cost_tensor
        best = int(gains.argmax())

        states = trials[best:best + 1]
        used += costs[best]
        current = next_errors[best]

    return states, used


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--budgets", type=int, nargs="+", default=[80])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        default=(
            "outputs/evaluation/timetok_v3/"
            "physio_v6/oracle_pilot.csv"
        ),
    )
    args = p.parse_args()

    cfg = SimpleNamespace(
        seed=args.seed,
        backbone=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
        singlem="third_party/SingLEM",
        physio_weight=5.0,
    )

    device, backbone, bc, teacher = setup(cfg)

    models = {}
    for name, path in (
        ("V4", "outputs/checkpoints/timetok_v3_agent_cost_eeg/best.pt"),
        ("V5", "outputs/checkpoints/timetok_v3_agent_physio_beta5/best.pt"),
        ("V6", "outputs/checkpoints/timetok_v3_agent_physio_local_v6/best.pt"),
    ):
        agent, _ = load_agent(path, device)
        agent.eval()
        models[name] = agent

    loader, total = data_loader(
        "val", "eeg",
        args.windows, args.batch_size, 0, args.seed,
    )

    rows = []
    index = 0

    for raw in loader:
        xbatch = normalize(raw, bc["normalization"], device)
        all_codes = get_codes(backbone, xbatch)

        for i in range(len(xbatch)):
            x = xbatch[i:i + 1]
            codes = all_codes[i:i + 1]
            reference = teacher(x, bc["normalization"])

            for budget in args.budgets:
                allocations = {}

                for name in ("wave", "singlem"):
                    allocations["oracle_" + name] = oracle_greedy(
                        backbone, teacher, codes, x,
                        bc["normalization"], budget, name,
                    )

                for name, agent in models.items():
                    levels, spent = fixed_budget(
                        agent, codes, budget
                    )
                    allocations[name] = (
                        levels, int(spent[0])
                    )

                for name, (levels, spent) in allocations.items():
                    y = decode_mixed(
                        backbone.reconstructor, codes, levels
                    ).float()

                    mae, _, corr = reconstruction_stats(y, x)
                    wave = quality_error(y, x)
                    sing = physio_error(
                        teacher(y, bc["normalization"]),
                        reference,
                    )

                    rows.append({
                        "window": index,
                        "budget": budget,
                        "policy": name,
                        "tokens": spent,
                        "mae": float(mae[0]),
                        "corr": float(corr[0]),
                        "singlem": float(sing[0]),
                        "wave_loss": float(wave[0]),
                    })

            index += 1

            if index % 4 == 0 or index == total:
                print(
                    f"Evaluated {index}/{total}",
                    flush=True,
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    for budget in args.budgets:
        print(
            f"\nBUDGET {budget}\n"
            "Policy | Tokens | SingLEM | Wave loss | MAE | Corr"
        )

        for name in (
            "oracle_wave", "oracle_singlem", "V4", "V5", "V6"
        ):
            subset = [
                row for row in rows
                if row["budget"] == budget
                and row["policy"] == name
            ]

            def avg(key):
                return sum(
                    row[key] for row in subset
                ) / len(subset)

            print(
                f"{name:15} | "
                f"{avg('tokens'):6.2f} | "
                f"{avg('singlem'):.6f} | "
                f"{avg('wave_loss'):.5f} | "
                f"{avg('mae'):.5f} | "
                f"{avg('corr'):.5f}"
            )

    print("\nSaved:", output)
    print(
        "Greedy oracles use original EEG. "
        "They are not globally optimal."
    )


if __name__ == "__main__":
    main()
