import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

sys.path.insert(0, str(Path.home() / "adaptive-physio/scripts"))

from run_arousal_adaptive_eval import (
    build_table, eeg_condition, pasl_arousal, OUTPUT
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(
    OUTPUT / "ground_truth_probe.pt",
    map_location="cpu",
    weights_only=False,
)

head = pasl_arousal.ArousalWakeLinear(input_dim=256).to(device)
head.load_state_dict(ck["head"])
head.eval()

mu, sd = ck["mu"], ck["sd"]

val = build_table("val", 150)
test = build_table("test", 0)

conditions = [
    "ground_truth",
    "uniform64",
    "random80",
    "fixed80",
    "V8_80", "V7_80",
    "uniform128",
]

rows = []

for name in conditions:
    scores = {}

    for split, table in [("val", val), ("test", test)]:
        table = replace(
            table,
            eeg=eeg_condition(split, name),
        )

        X = pasl_arousal.epoch_features(table)

        arousal_prob, _ = pasl_arousal.probs(
            head, X, mu, sd, device
        )
        del X

        mask = table.sleep.astype(bool)

        scores[split] = (
            table.arousal[mask].astype(int),
            arousal_prob[mask],
        )

    y_val, p_val = scores["val"]
    y_test, p_test = scores["test"]

    # Calibrate the threshold exclusively on validation.
    thresholds = np.arange(0.10, 0.901, 0.025)

    threshold = max(
        thresholds,
        key=lambda t: f1_score(
            y_val, p_val >= t, zero_division=0
        ),
    )

    predicted = p_test >= threshold

    result = {
        "condition": name,
        "threshold": round(float(threshold), 4),
        "AUPRC": average_precision_score(y_test, p_test),
        "AUROC": roc_auc_score(y_test, p_test),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "positive_rate": float(predicted.mean()),
    }

    rows.append(result)

    print(
        f"{name:12s} "
        f"thr={threshold:.3f} "
        f"AUPRC={result['AUPRC']:.4f} "
        f"AUROC={result['AUROC']:.4f} "
        f"F1={result['F1']:.4f} "
        f"pred_positive={result['positive_rate']:.1%}",
        flush=True,
    )

path = OUTPUT / "test_metrics_calibrated.csv"

with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("SAVED:", path)
