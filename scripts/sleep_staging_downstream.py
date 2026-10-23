"""SHHS1 adaptive-physio downstream sleep staging: prepare / train / eval.

5-stage labels: Wake, N1, N2, N3 (stage3+stage4), REM.
The original-EEG classifier is frozen for all downstream comparisons.
"""
import argparse
import csv
import json
import math
import random
import re
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adaptive_physio.data.preprocessed_dataset import PreprocessedPSGDataset
from adaptive_physio.models.timetok_physio_v3 import PhysioTokenizerV3
from evaluate_timetok_v3_adaptive import decode_mixed
from analyze_v4_v5_regions import fixed_budget
from train_temporal_agent_v4 import load_agent

CLASSES = ("Wake", "N1", "N2", "N3", "REM")
# SHHS Profusion/NSRR raw stage numbering, merge N3/N4.
RAW_TO_FIVE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
DEFAULT_SOURCE = "/hdd2/kdpark/sleep_datasets/shhs1/mmap_wav2sleep"
DEFAULT_ROOT = "/hdd2/kdpark/sleep_datasets/shhs1"
BACKBONE_CK = "outputs/checkpoints/timetok_v3_eeg_ae/frozen_for_agent.pt"
AGENT_CK = {
    "V4_80": "outputs/checkpoints/timetok_v3_agent_cost_eeg/best.pt",
    "V5_80": "outputs/checkpoints/timetok_v3_agent_physio_beta5/best.pt",
    "V7_80": "outputs/checkpoints/timetok_v3_agent_physio_v7/best.pt",
}


def canonical_record(name):
    match = re.search(r"shhs1[-_](\d{6})", str(name), re.I)
    if match:
        return "shhs1-" + match.group(1)
    return None


def xml_paths(root, explicit=None):
    if explicit:
        locations = [Path(explicit)]
    else:
        base = Path(root)
        locations = [
            base / "annotations",
            base / "annotations_notruncate",
            base / "polysomnography" / "annotations-events-profusion" / "shhs1",
            base / "polysomnography" / "annotations-events-nsrr" / "shhs1",
        ]
    files = {}
    for location in locations:
        if not location.exists():
            continue
        candidates = [location] if location.is_file() else location.rglob("*.xml")
        for path in candidates:
            record = canonical_record(path.name)
            if record is None:
                continue
            score = 2 if "profusion" in path.name.lower() else 1
            if record not in files or score > files[record][0]:
                files[record] = (score, path)
    return {record: pair[1] for record, pair in files.items()}


def localtag(t):
    return t.rsplit("}", 1)[-1].lower()


def childtext(element, wanted):
    wanted = wanted.lower()
    for child in element:
        if localtag(child.tag) == wanted:
            return (child.text or "").strip()
    return ""


def stage_code(text):
    text = str(text).strip()
    if not text:
        return -1
    # NSRR EventConcept examples: 'Wake|0', 'REM sleep|5'.
    match = re.search(r"\|\s*(\d+)\s*$", text)
    if match:
        return RAW_TO_FIVE.get(int(match.group(1)), -1)
    if re.fullmatch(r"\d+", text):
        return RAW_TO_FIVE.get(int(text), -1)
    low = text.lower()
    if "rem" in low:
        return 4
    if "wake" in low or "awake" in low:
        return 0
    match = re.search(r"(?:stage\s*|\bn)([1-4])", low)
    if match:
        return RAW_TO_FIVE[int(match.group(1))]
    return -1


def parse_xml(path):
    root = ET.parse(path).getroot()
    # Profusion XML has one SleepStage per 30-second epoch.
    for element in root.iter():
        if localtag(element.tag) != "sleepstages":
            continue
        entries = [
            e for e in element
            if localtag(e.tag) == "sleepstage"
        ]
        if entries:
            duration = 30.0
            for element2 in root.iter():
                if localtag(element2.tag) == "epochlength":
                    duration = float(element2.text)
                    break
            if abs(duration - 30.0) > 1e-4:
                raise ValueError(f"Expected 30s Profusion epochs, got {duration}: {path}")
            return np.array([stage_code(e.text) for e in entries], dtype=np.int8)

    # NSRR XML lists time-scored stage events, possibly spanning several epochs.
    events = []
    for ev in root.iter():
        if localtag(ev.tag) != "scoredevent":
            continue
        typ = childtext(ev, "EventType")
        concept = childtext(ev, "EventConcept")
        if "stage" not in typ.lower() and not (
            "stage" in concept.lower() or "wake" in concept.lower() or "rem" in concept.lower()
        ):
            continue
        label = stage_code(concept)
        start, length = childtext(ev, "Start"), childtext(ev, "Duration")
        if not (start and length):
            continue
        seconds, duration = float(start), float(length)
        if seconds < 0 or duration <= 0:
            continue
        a = int(round(seconds / 30.0))
        count = max(1, int(round(duration / 30.0)))
        events.append((a, count, label))
    if not events:
        raise ValueError(f"No scored sleep-stage annotations in {path}")
    end = max(a + n for a, n, _ in events)
    stages = np.full(end, -1, dtype=np.int8)
    for a, n, label in events:
        stages[a:a+n] = label
    return stages


def load_source(split, args):
    return PreprocessedPSGDataset(split=split, root=args.source)


def metadata_for(base):
    return {
        "records": [r["recording"] for r in base.records],
        "lengths": [int(r["windows"]) for r in base.records],
        "windows": len(base),
    }



def parse_shhs_csv(path):
    mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    # Exactly six 5-second labels per 30-second EEG epoch.
    n = len(rows) // 6
    labels = np.full(n, -1, dtype=np.int8)

    for epoch in range(n):
        block = rows[epoch * 6:(epoch + 1) * 6]
        names = [r["StageName"].strip().upper() for r in block]

        if len(set(names)) != 1 or names[0] not in mapping:
            continue

        valid_grid = True
        for j, row in enumerate(block):
            start = float(row["Start"])
            stop = float(row["Stop"])
            expected = epoch * 30 + j * 5

            if abs(start - expected) > 0.001 or abs(stop - expected - 5) > 0.001:
                valid_grid = False
                break

        if valid_grid:
            labels[epoch] = mapping[names[0]]

    return labels


def prepare(args):
    root = Path(
        args.annotations_root
        or Path(args.shhs_root) / "sleep_staging_labels_5sec"
    )

    files = {
        canonical_record(f.name): f
        for f in root.glob("shhs1-*_5sec.csv")
    }

    print(f"Found {len(files)} SHHS1 CSV label files in {root}", flush=True)

    if not files:
        raise RuntimeError(f"No SHHS1 CSV labels found: {root}")

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    record_sets = {}

    for split in ("train", "val", "test"):
        base = load_source(split, args)
        aligned = np.full(len(base), -1, dtype=np.int8)

        previous = 0
        covered = 0
        missing = 0
        mismatch = 0
        record_sets[split] = set()

        for record, end in zip(base.records, base.cumulative):
            name = canonical_record(record["recording"])
            record_sets[split].add(name)

            length = end - previous
            path = files.get(name)

            if path is None:
                missing += 1
                previous = end
                continue

            try:
                labels = parse_shhs_csv(path)
            except Exception as exc:
                print(f"BAD CSV {name}: {exc}", flush=True)
                mismatch += 1
                previous = end
                continue

            # Reject large differences between signal and label length.
            if abs(len(labels) - length) > 20:
                mismatch += 1
                if mismatch <= 10:
                    print(
                        f"ALIGNMENT CHECK {name}: "
                        f"EEG={length}, labels={len(labels)}"
                    )
                previous = end
                continue

            count = min(length, len(labels))
            aligned[previous:previous + count] = labels[:count]

            covered += 1
            previous = end

        valid = int((aligned >= 0).sum())

        print(
            f"{split}: records={covered}/{len(base.records)} "
            f"valid_epochs={valid:,}/{len(base):,} "
            f"missing={missing} mismatched={mismatch}",
            flush=True
        )

        counts = Counter(aligned[aligned >= 0].tolist())
        print(
            "Classes:",
            {CLASSES[k]: counts.get(k, 0) for k in range(5)}
        )

        if valid < 0.5 * len(base):
            raise RuntimeError(
                f"Too few aligned labels in {split}: "
                f"{valid / len(base):.1%}"
            )

        np.save(cache / f"stages_{split}.npy", aligned)

        (cache / f"metadata_{split}.json").write_text(
            json.dumps(metadata_for(base), indent=2)
        )

    for x, y in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ):
        overlap = record_sets[x] & record_sets[y]
        if overlap:
            raise RuntimeError(
                f"Recording overlap {x}/{y}: {list(overlap)[:5]}"
            )

    print("PREPARE PASSED: SHHS1 5-sec CSV aligned to 30-sec EEG")


class EEGStageDataset(Dataset):
    """Only EEG mmap is read; labels/indices correspond to source records."""
    def __init__(self, base, indices, labels):
        self.records = base.records
        self.cumulative = base.cumulative
        self.eeg_path = base.paths["eeg"]
        self.eeg_shape = tuple(base.metadata["eeg"]["data_shape"])
        self.indices = indices.astype(np.int64)
        self.labels = labels
        self._mmap = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmap"] = None
        return state

    def __getitem__(self, i):
        if self._mmap is None:
            self._mmap = np.memmap(
                self.eeg_path, dtype=np.float32, mode="r", shape=self.eeg_shape
            )
        idx = int(self.indices[i])
        pos = bisect_right(self.cumulative, idx)
        prior = self.cumulative[pos - 1] if pos else 0
        local = idx - prior
        start = int(self.records[pos]["positions"]["eeg"])
        sample = np.array(self._mmap[start + local], copy=True)
        if not np.isfinite(sample).all():
            raise RuntimeError(f"Non-finite EEG at sample {idx}")
        return torch.from_numpy(sample[None]), int(self.labels[idx]), idx


def labeled_dataset(split, n, args):
    base = load_source(split, args)
    d = Path(args.cache)
    labels = np.load(d / f"stages_{split}.npy", mmap_mode="r")
    metadata = json.loads((d / f"metadata_{split}.json").read_text())
    if len(labels) != len(base) or metadata != metadata_for(base):
        raise RuntimeError(f"{split}: label cache doesn't match source mmap record positions")
    valid = np.flatnonzero(labels >= 0)
    # One deterministic frozen set for all downstream conditions.
    size = min(n, len(valid)) if n > 0 else len(valid)
    key = d / f"indices_{split}_{size}_seed{args.seed}.npy"
    if key.exists():
        indices = np.load(key)
        if len(indices) != size or not np.isin(indices, valid).all():
            raise RuntimeError(f"Invalid cached indices: {key}")
    else:
        rng = np.random.default_rng(args.seed + {"train": 0, "val": 1, "test": 2}[split])
        indices = np.sort(rng.choice(valid, size, replace=False))
        np.save(key, indices)
    class_counts = np.bincount(np.asarray(labels[indices], dtype=np.int64), minlength=5)
    print(f"{split}: sampled={size:,} stage distribution={dict(zip(CLASSES,class_counts.tolist()))}")
    return EEGStageDataset(base, indices, labels)


class SleepStageCNN(nn.Module):
    """One 30-second EEG epoch -> W/N1/N2/N3/REM. No neighbor-epoch context."""
    def __init__(self):
        super().__init__()
        self.register_buffer("hann", torch.hann_window(256), persistent=False)
        layers = []
        channels = [(1, 24), (24, 48), (48, 72), (72, 96)]
        for inp, out in channels:
            layers += [
                nn.Conv2d(inp, out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, out), out),
                nn.GELU(),
            ]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(96 * 2, 128), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(128, 5),
        )

    def forward(self, eeg):
        spec = torch.stft(
            eeg[:, 0].float(), n_fft=256, hop_length=64,
            win_length=256, window=self.hann,
            return_complex=True,
        )
        feat = torch.log1p(spec.abs()).unsqueeze(1)
        z = self.features(feat)
        pooled = torch.cat(
            (z.mean(dim=(-2, -1)), z.amax(dim=(-2, -1))), dim=1
        )
        return self.head(pooled)


def normalize_eeg(raw, norm, device):
    return ((raw.to(device, non_blocking=True) - norm["mean"]) / norm["std"]).clamp(-8, 8)


def metrics_from_confusion(cm):
    cm = cm.astype(np.float64)
    n = cm.sum()
    acc = np.trace(cm) / max(n, 1)
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    expected = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / max(n*n, 1)
    kappa = (acc - expected) / max(1 - expected, 1e-12)
    return {"acc": float(acc), "macro_f1": float(f1.mean()),
            "kappa": float(kappa), "f1_per_class": f1.tolist()}


@torch.no_grad()
def validate(model, loader, norm, device):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    total_loss, count = 0., 0
    for raw, y, _ in loader:
        x = normalize_eeg(raw, norm, device)
        y = y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(dim=1)
        true_np, pred_np = y.cpu().numpy(), pred.cpu().numpy()
        cm += np.bincount(true_np * 5 + pred_np, minlength=25).reshape(5, 5)
        count += len(y)
    result = metrics_from_confusion(cm)
    result["loss"] = total_loss / max(count, 1)
    return result


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: training on CPU")
    backbone_ck = torch.load(BACKBONE_CK, map_location="cpu", weights_only=False)
    norm = backbone_ck["normalization"]
    train_ds = labeled_dataset("train", args.train_samples, args)
    val_ds = labeled_dataset("val", args.val_samples, args)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    counts = np.bincount(np.asarray(train_ds.labels[train_ds.indices], dtype=np.int64), minlength=5)
    # Sublinear inverse-frequency weights, avoids exploding N1 weighting.
    weights = np.sqrt(counts.sum() / (5 * np.maximum(counts, 1)))
    weights = np.minimum(weights, 4.0)
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    print("Class weights:", weights.cpu().numpy().round(3))
    model = SleepStageCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = -1.0
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, seen = 0., 0
        for step, (raw, labels, _) in enumerate(train_loader, 1):
            x = normalize_eeg(raw, norm, device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            seen += len(labels)
            if step % args.log_every == 0:
                print(f"epoch={epoch}/{args.epochs} step={step} loss={loss.item():.4f}", flush=True)
        val = validate(model, val_loader, norm, device)
        print(f"EPOCH {epoch} train_loss={loss_sum/seen:.5f} "
              f"val_acc={val['acc']:.4f} val_MF1={val['macro_f1']:.4f} "
              f"val_kappa={val['kappa']:.4f}", flush=True)
        checkpoint = {"model": model.state_dict(), "normalization": norm,
                      "epoch": epoch, "val": val, "train_samples":len(train_ds),
                      "class_names":CLASSES, "architecture":"epoch_spectrogram_cnn_v1"}
        torch.save(checkpoint, output / "last.pt")
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save(checkpoint, output / "best.pt")
            print(f"BEST epoch={epoch} val_MF1={best:.4f}", flush=True)
    print("TRAIN FINISHED:", output / "best.pt")


def random_levels(batch_indices, budget, device, seed):
    result = []
    for sample in batch_indices:
        # Reproducible per epoch ID, independent of evaluation batch boundaries.
        rng = random.Random(int(seed)*1000003 + int(sample))
        levels = [0] * 16
        used = 16
        while True:
            available = [
                (r, 3 if levels[r] == 0 else 4)
                for r in range(16)
                if levels[r] < 2 and used + (3 if levels[r] == 0 else 4) <= budget
            ]
            if not available:
                break
            region, cost = rng.choice(available)
            levels[region] += 1
            used += cost
        result.append(levels)
    return torch.tensor(result, dtype=torch.long, device=device)


@torch.inference_mode()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    staging_ck = torch.load(Path(args.out) / "best.pt", map_location="cpu", weights_only=False)
    model = SleepStageCNN().to(device).eval()
    model.load_state_dict(staging_ck["model"])
    norm = staging_ck["normalization"]
    ds = labeled_dataset("test", args.test_samples, args)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    ae_ck = torch.load(BACKBONE_CK, map_location="cpu", weights_only=False)
    if ae_ck["normalization"] != norm:
        raise RuntimeError("Downstream normalization and tokenizer normalization mismatch")
    backbone = PhysioTokenizerV3().to(device).eval()
    backbone.load_state_dict(ae_ck["model"])
    agents = {}
    for label, path in AGENT_CK.items():
        if Path(path).exists():
            agent, _ = load_agent(path, device)
            agents[label] = agent.eval()
        else:
            print("WARNING skipping missing agent:", path)
    conditions = ["original", "uniform16", "uniform64", "uniform128", "random80", "fixed80"] + list(agents)
    matrices = {key: np.zeros((5,5), dtype=np.int64) for key in conditions}
    counts_tokens = {key: 0. for key in conditions}
    prediction_file = Path(args.out) / "test_predictions.csv"
    with prediction_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "recording", "epoch", "true", "pred", "tokens"])
        writer.writeheader()
        seen = 0
        for step, (raw, labels, global_indices) in enumerate(loader, 1):
            x = normalize_eeg(raw, norm, device)
            codes, _ = backbone.tokenize(x)
            variants = {
                "original": (x, torch.full((len(x),), 128, device=device)),
                "uniform16": (backbone.reconstructor(codes, 16).float(), torch.full((len(x),), 16, device=device)),
                "uniform64": (backbone.reconstructor(codes, 64).float(), torch.full((len(x),), 64, device=device)),
                "uniform128": (backbone.reconstructor(codes, 128).float(), torch.full((len(x),), 128, device=device)),
            }
            random_plan = random_levels(global_indices, 80, device, args.seed)
            variants["random80"] = (
                decode_mixed(backbone.reconstructor, codes, random_plan).float(),
                (16 + 3*(random_plan>=1).sum(1) + 4*(random_plan>=2).sum(1)),
            )
            fixed_plan = torch.ones(len(x), 16, dtype=torch.long, device=device)
            fixed_plan[:, [0,4,8,12]] = 2
            variants["fixed80"] = (
                decode_mixed(backbone.reconstructor, codes, fixed_plan).float(),
                torch.full((len(x),), 80, device=device),
            )
            for name, agent in agents.items():
                levels, used = fixed_budget(agent, codes, budget=80)
                variants[name] = (decode_mixed(backbone.reconstructor, codes, levels).float(), used)
            y = labels.numpy()
            indices = global_indices.numpy()
            for name, (signal, tokens) in variants.items():
                pred = model(signal).argmax(dim=1).cpu().numpy()
                matrices[name] += np.bincount(y * 5 + pred, minlength=25).reshape(5,5)
                counts_tokens[name] += float(tokens.sum().item())
                for idx, true, predicted, ntokens in zip(indices, y, pred, tokens.cpu().numpy()):
                    pos = bisect_right(ds.cumulative, int(idx))
                    prior = ds.cumulative[pos-1] if pos else 0
                    writer.writerow({"condition":name,
                                     "recording":ds.records[pos]["recording"],
                                     "epoch":int(idx)-prior,
                                     "true":int(true), "pred":int(predicted),
                                     "tokens":int(ntokens)})
            seen += len(labels)
            if step % args.log_every == 0:
                print(f"TEST {seen}/{len(ds)} epochs", flush=True)
    report = []
    for name in conditions:
        results = metrics_from_confusion(matrices[name])
        used = None if name == "original" else counts_tokens[name]/len(ds)
        row = {"condition":name, "avg_tokens": used,
               "accuracy":results["acc"], "macro_f1":results["macro_f1"],
               "kappa":results["kappa"]}
        report.append(row)
        print(f"{name:12} tokens={str(round(used,1)) if used is not None else 'raw':>5} "
              f"acc={row['accuracy']:.4f} MF1={row['macro_f1']:.4f} "
              f"kappa={row['kappa']:.4f} "
              f"F1={np.array(results['f1_per_class']).round(3).tolist()}", flush=True)
    csvpath = Path(args.out) / "test_metrics.csv"
    with csvpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    print("Saved:", csvpath, "and", prediction_file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "train", "eval"], required=True)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--shhs-root", default=DEFAULT_ROOT)
    p.add_argument("--annotations-root", default=None)
    p.add_argument("--cache", default="outputs/downstream/staging/stage_cache")
    p.add_argument("--out", default="outputs/downstream/staging/epoch_cnn_original")
    p.add_argument("--train-samples", type=int, default=120000)
    p.add_argument("--val-samples", type=int, default=12000)
    p.add_argument("--test-samples", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
