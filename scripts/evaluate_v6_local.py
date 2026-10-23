import argparse
import csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from adaptive_physio.models.timetok_physio_v3 import reconstruction_stats
from evaluate_timetok_v3_adaptive import decode_mixed
from analyze_v4_v5_regions import fixed_budget
from train_temporal_agent_v4 import load_agent
from train_temporal_agent_physio_v5 import (
    setup, data_loader, normalize, get_codes,
)
from train_temporal_agent_physio_v6 import (
    region_weights, feature_distances,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 80, 96])
    p.add_argument(
        "--output",
        default="outputs/evaluation/timetok_v3/physio_v6/local_comparison.csv",
    )
    args = p.parse_args()

    torch.manual_seed(42)

    cfg = SimpleNamespace(
        seed=42,
        backbone="outputs/checkpoints/timetok_v3_eeg_ae/frozen_for_agent.pt",
        singlem="third_party/SingLEM",
        physio_weight=5.0,
    )
    device, backbone, ck, teacher = setup(cfg)

    paths = {
        "V4": "outputs/checkpoints/timetok_v3_agent_cost_eeg/best.pt",
        "V5": "outputs/checkpoints/timetok_v3_agent_physio_beta5/best.pt",
        "V6": "outputs/checkpoints/timetok_v3_agent_physio_local_v6/best.pt",
    }
    agents = {}
    for name, path in paths.items():
        agents[name], _ = load_agent(path, device)
        agents[name].eval()

    weights = region_weights(device, sigma=1.25)

    loader, total = data_loader(
        "val", "eeg", args.windows, args.batch_size, 0, 42
    )

    rows = []
    comparisons = defaultdict(list)
    offset = 0

    with torch.inference_mode():
        for raw in loader:
            x = normalize(raw, ck["normalization"], device)
            codes = get_codes(backbone, x)
            reference = teacher(x, ck["normalization"])

            for budget in args.budgets:
                levels = {}

                for name, agent in agents.items():
                    lv, used = fixed_budget(agent, codes, budget)
                    levels[name] = lv

                    reconstructed = decode_mixed(
                        backbone.reconstructor, codes, lv
                    ).float()

                    features = teacher(
                        reconstructed, ck["normalization"]
                    )
                    global_dist, local_dist = feature_distances(
                        features, reference, weights
                    )

                    mae, _, corr = reconstruction_stats(
                        reconstructed, x
                    )

                    worst4 = local_dist.topk(
                        4, dim=1
                    ).values.mean(dim=1)

                    for i in range(len(x)):
                        rows.append({
                            "window": offset + i,
                            "budget": budget,
                            "model": name,
                            "tokens": int(used[i].item()),
                            "mae": float(mae[i]),
                            "corr": float(corr[i]),
                            "global_sing": float(global_dist[i]),
                            "local_mean": float(local_dist[i].mean()),
                            "local_worst4": float(worst4[i]),
                            "local_max": float(local_dist[i].max()),
                        })

                for other in ("V4", "V5"):
                    a = levels[other]
                    b = levels["V6"]

                    agreement = (a == b).float().mean(dim=1)

                    fa, fb = a == 2, b == 2
                    union = (fa | fb).sum(dim=1)
                    common = (fa & fb).sum(dim=1)

                    jaccard = torch.where(
                        union > 0,
                        common.float() / union.clamp_min(1),
                        torch.ones_like(union, dtype=torch.float),
                    )

                    comparisons[(budget, other)].extend(
                        zip(
                            agreement.cpu().tolist(),
                            jaccard.cpu().tolist(),
                        )
                    )

            offset += len(x)
            if offset % 128 == 0 or offset == total:
                print(f"Evaluated {offset}/{total}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for budget in args.budgets:
        print(f"\n========== BUDGET {budget} ==========")
        print(
            f"{'Model':6} {'Tokens':>8} {'Global':>10} "
            f"{'LocalMean':>10} {'Worst4':>10} "
            f"{'LocalMax':>10}"
        )

        for name in agents:
            subset = [
                r for r in rows
                if r["budget"] == budget and r["model"] == name
            ]

            def mean(key):
                return np.mean([r[key] for r in subset])

            print(
                f"{name:6} {mean('tokens'):8.2f} "
                f"{mean('global_sing'):10.6f} "
                f"{mean('local_mean'):10.6f} "
                f"{mean('local_worst4'):10.6f} "
                f"{mean('local_max'):10.6f}"
            )

        for other in ("V4", "V5"):
            arr = np.asarray(comparisons[(budget, other)])
            print(
                f"{other} vs V6: "
                f"level agreement={arr[:, 0].mean():.1%}, "
                f"fine Jaccard={arr[:, 1].mean():.3f}"
            )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
