#!/usr/bin/env python3
"""One ground-truth arousal probe, evaluated on adaptive reconstructions."""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

AD = Path.home() / "adaptive-physio"
PA = Path.home() / "physiology_as_language_repro"

sys.path.insert(
    0, str(PA / "paper_architecture/downstream/arousal")
)
import train_arousal_linear as pasl_arousal

SOURCE = (
    PA / "retraining_wav2sleep/outputs/downstream/seed_42/shhs1"
)
GENERATED = (
    AD / "outputs/downstream/staging/adaptive_pasl"
)
AROUSAL = Path(
    "/hdd2/kdpark/sleep_datasets/shhs1/arousal_epoch_labels"
)
OUTPUT = (
    AD / "outputs/downstream/arousal/adaptive_frozen_probe"
)

DEFAULT_CONDITIONS = (
    "ground_truth",
    "uniform64",
    "random80",
    "fixed80",
    "V7_80",
    "uniform128",
)


def load_metadata(folder):
    meta = json.loads(
        (folder / "metadata.json").read_text()
    )
    shape = tuple(meta["eeg_shape"])

    if len(shape) != 3 or shape[1:] != (256, 512):
        raise RuntimeError(
            f"Bad EEG shape {shape}: {folder}"
        )

    return meta, shape


def eeg_source(split):
    folder = SOURCE / split
    meta, shape = load_metadata(folder)
    path = folder / "ground_truth_eeg.mmap"

    expected = (
        int(np.prod(shape))
        * np.dtype(meta["eeg_dtype"]).itemsize
    )
    if path.stat().st_size != expected:
        raise RuntimeError(
            f"Incorrect GT mmap size: {path}"
        )

    return np.memmap(
        path,
        dtype=np.dtype(meta["eeg_dtype"]),
        mode="r",
        shape=shape,
    )


def eeg_condition(split, name):
    if name == "ground_truth":
        return eeg_source(split)

    folder = GENERATED / name / "shhs1" / split
    gt_folder = SOURCE / split

    # Require the exact same original metadata and labels.
    for f in (
        "metadata.json",
        "index.jsonl",
        "labels.mmap",
        "ground_truth_eeg.mmap",
    ):
        target = folder / f
        original = gt_folder / f

        if (
            not target.exists()
            or not target.samefile(original)
        ):
            raise RuntimeError(
                f"Dataset alignment mismatch: {target}"
            )

    if not (folder / ".complete").exists():
        raise RuntimeError(
            f"Missing complete flag: {name} {split}"
        )

    meta, shape = load_metadata(folder)
    path = folder / "synthesized_eeg.mmap"

    if not path.exists():
        raise FileNotFoundError(path)

    expected = (
        int(np.prod(shape))
        * np.dtype(meta["eeg_dtype"]).itemsize
    )

    if path.stat().st_size != expected:
        raise RuntimeError(
            f"Incorrect mmap size: {path}"
        )

    return np.memmap(
        path,
        dtype=np.dtype(meta["eeg_dtype"]),
        mode="r",
        shape=shape,
    )


def build_table(split, subject_cap):
    table = pasl_arousal.build_split_table(
        SOURCE,
        AROUSAL,
        split,
        subject_cap,
    )

    if not len(table.arousal):
        raise RuntimeError(
            f"No arousal annotations for {split}"
        )

    # Train/validation always use ORIGINAL EEG,
    # never PaSL-synthesized EEG.
    return replace(
        table,
        eeg=eeg_source(split),
    )


def check_subjects(train, val, test):
    for a, b in (
        (train, val),
        (train, test),
        (val, test),
    ):
        overlap = (
            set(a.selected_subjects)
            & set(b.selected_subjects)
        )

        if overlap:
            raise RuntimeError(
                f"Patient leakage {a.split}/{b.split}"
            )

    print("PATIENT SPLITS DISJOINT: PASS", flush=True)


def ap_auc(y, score):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    return (
        float(average_precision_score(y, score)),
        float(roc_auc_score(y, score)),
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=["audit", "run"],
        default="run",
    )
    p.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--probe-train-subjects",
        type=int,
        default=300,
    )
    p.add_argument(
        "--probe-val-subjects",
        type=int,
        default=150,
    )
    p.add_argument(
        "--test-subjects",
        type=int,
        default=0,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
    )
    args = p.parse_args()

    conditions = tuple(dict.fromkeys(
        x.strip()
        for x in args.conditions.split(",")
        if x.strip()
    ))

    if not conditions:
        p.error("No conditions provided")

    if not AROUSAL.is_dir():
        raise FileNotFoundError(AROUSAL)

    pasl_arousal.set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device, flush=True)

    train = build_table(
        "train", args.probe_train_subjects
    )
    val = build_table(
        "val", args.probe_val_subjects
    )
    test = build_table(
        "test", args.test_subjects
    )

    check_subjects(train, val, test)

    for table in (train, val, test):
        print(
            f"{table.split}: "
            f"{len(table.selected_subjects)} subjects, "
            f"{len(table.arousal):,} 30s epochs, "
            f"{int(table.arousal.sum()):,} arousal positives, "
            f"{int(table.sleep.sum()):,} sleep epochs",
            flush=True,
        )

    for condition in conditions:
        x = eeg_condition("test", condition)

        if tuple(x.shape) != tuple(test.eeg.shape):
            raise RuntimeError(
                f"Shape mismatch: {condition}"
            )

        print(
            "READY:", condition, tuple(x.shape),
            flush=True,
        )

    if args.mode == "audit":
        print("AROUSAL AUDIT PASSED")
        return

    args.output.mkdir(
        parents=True, exist_ok=True
    )
    ck_path = args.output / "ground_truth_probe.pt"

    if ck_path.exists():
        ck = torch.load(
            ck_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            ck.get("train_subjects") != train.selected_subjects
            or ck.get("val_subjects") != val.selected_subjects
        ):
            raise RuntimeError(
                "Existing probe has different train/val subjects"
            )

        head = pasl_arousal.ArousalWakeLinear(
            input_dim=256
        ).to(device)
        head.load_state_dict(ck["head"])
        head.eval()

        mu = ck["mu"]
        sd = ck["sd"]
        ar_thr = ck["arousal_threshold"]
        wake_thr = ck["wake_threshold"]

        print(
            "Loaded existing GT probe:", ck_path,
            flush=True,
        )

    else:
        print(
            "Training ONE arousal/wake probe on GT...",
            flush=True,
        )

        X_train = pasl_arousal.epoch_features(train)

        Y_train = torch.from_numpy(
            np.stack(
                [train.arousal, train.wake],
                axis=1,
            )
        )

        pos_ar = float(train.arousal.mean())
        pos_wk = float(train.wake.mean())

        if pos_ar <= 0 or pos_wk <= 0:
            raise RuntimeError(
                "Missing positive training labels"
            )

        head, (mu, sd) = (
            pasl_arousal.fit_linear_probe(
                X=X_train,
                y=Y_train,
                device=device,
                pos_weight=[
                    (1 - pos_ar) / pos_ar,
                    (1 - pos_wk) / pos_wk,
                ],
                epochs=args.epochs,
                lr=1e-2,
                batch=65536,
            )
        )
        del X_train, Y_train

        X_val = pasl_arousal.epoch_features(val)

        val_ar_p, val_wk_p = pasl_arousal.probs(
            head, X_val, mu, sd, device
        )
        del X_val

        val_ar = val.arousal > 0.5
        val_wk = val.wake > 0.5
        val_slp = val.sleep.astype(bool)

        grid = np.arange(0.1, 0.91, 0.05)

        ar_thr = float(max(
            grid,
            key=lambda t: pasl_arousal.f1_of(
                *pasl_arousal.counts(
                    val_ar_p[val_slp] >= t,
                    val_ar[val_slp],
                )
            )[0],
        ))

        wake_thr = float(max(
            grid,
            key=lambda t: pasl_arousal.balanced_acc(
                val_wk_p >= t,
                val_wk,
            ),
        ))

        torch.save(
            {
                "head": {
                    k: v.detach().cpu()
                    for k, v in head.state_dict().items()
                },
                "mu": mu.cpu(),
                "sd": sd.cpu(),
                "arousal_threshold": ar_thr,
                "wake_threshold": wake_thr,
                "train_subjects": train.selected_subjects,
                "val_subjects": val.selected_subjects,
                "source": str(
                    SOURCE / "train/ground_truth_eeg.mmap"
                ),
            },
            ck_path,
        )

        print(
            "Saved frozen GT probe:", ck_path,
            flush=True,
        )

    print(
        f"Thresholds: arousal={ar_thr:.2f}, "
        f"wake={wake_thr:.2f}",
        flush=True,
    )

    gt_ar = test.arousal > 0.5
    gt_sleep = test.sleep.astype(bool)

    rows = []

    for condition in conditions:
        print("EVAL:", condition, flush=True)

        # Only EEG changes. Subjects, labels and epoch
        # indices are identical across conditions.
        table = replace(
            test,
            eeg=eeg_condition("test", condition),
        )

        X_test = pasl_arousal.epoch_features(table)

        prob_ar, prob_wake = pasl_arousal.probs(
            head, X_test, mu, sd, device
        )

        del X_test, table

        # Primary metric: arousal detection during
        # ground-truth sleep epochs only.
        ar_sleep = gt_ar[gt_sleep]
        score_sleep = prob_ar[gt_sleep]

        scores = pasl_arousal.reading(
            score_sleep >= ar_thr,
            ar_sleep,
            score_sleep,
        )

        auprc, auroc = ap_auc(
            ar_sleep, score_sleep
        )

        # Additional end-to-end metric with predicted
        # wake gate, matching the prior PaSL protocol.
        gated = pasl_arousal.reading(
            (prob_ar >= ar_thr)
            & (prob_wake < wake_thr),
            gt_ar & gt_sleep,
            prob_ar * (1 - prob_wake),
        )

        row = {
            "condition": condition,
            "subjects": len(test.selected_subjects),
            "test_sleep_epochs": len(ar_sleep),
            "arousal_positives_sleep": int(
                ar_sleep.sum()
            ),
            "AUPRC_sleep": auprc,
            "AUROC_sleep": auroc,
            "F1_sleep": scores["f1"],
            "precision_sleep": scores["precision"],
            "recall_sleep": scores["recall"],
            "F1_gated": gated["f1"],
            "arousal_threshold": ar_thr,
            "wake_threshold": wake_thr,
        }

        print(
            f"  AUPRC={auprc:.5f} "
            f"AUROC={auroc:.5f} "
            f"F1={scores['f1']:.5f} "
            f"Gated-F1={gated['f1']:.5f}",
            flush=True,
        )

        rows.append(row)

        with (
            args.output / "test_metrics.csv"
        ).open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)

    print(
        "SAVED:",
        args.output / "test_metrics.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
