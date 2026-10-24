import argparse, hashlib, importlib.util, json, os, random, re, sys
from pathlib import Path
import numpy as np
import torch

AD = Path.home() / "adaptive-physio"
PA = Path.home() / "physiology_as_language_repro"
sys.path[:0] = [str(AD / "src"), str(AD / "scripts"), str(PA)]

from adaptive_physio.data.preprocessed_dataset import PreprocessedPSGDataset
from adaptive_physio.models.timetok_physio_v3 import PhysioTokenizerV3
from evaluate_timetok_v3_adaptive import decode_mixed
from analyze_v4_v5_regions import fixed_budget
from train_temporal_agent_v4 import load_agent

SOURCE = PA / "retraining_wav2sleep/outputs/downstream/seed_42/shhs1"
OUT = AD / "outputs/downstream/staging/adaptive_pasl"

AGENTS = {
    "V4": AD / "outputs/checkpoints/timetok_v3_agent_cost_eeg/best.pt",
    "V5": AD / "outputs/checkpoints/timetok_v3_agent_physio_beta5/best.pt",
    "V7": AD / "outputs/checkpoints/timetok_v3_agent_physio_v7/best.pt",
}

STAGE_TO_4 = np.array([0, 1, 1, 2, 3], dtype=np.int64)


def ident(s):
    m = re.search(r"shhs1[-_](\d{6})", str(s), re.I)
    if not m:
        raise ValueError(f"Invalid SHHS1 identifier: {s}")
    return m.group(1)



def find_norm(explicit):
    path = (
        Path(explicit).expanduser()
        if explicit
        else PA / "retraining_wav2sleep/outputs/shared/preprocessing/normalization.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())
    low = float(data["min_db"])
    high = float(data["max_db"])

    if not np.isfinite([low, high]).all() or high <= low:
        raise RuntimeError("Invalid PaSL normalization")

    print("PaSL normalization:", path, low, high, flush=True)
    return low, high


def load_preprocessing():
    path = PA / "src/data/preprocessing.py"
    spec = importlib.util.spec_from_file_location("pasl_proc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spectrum(wave, proc, low, high):
    wave = np.asarray(wave, dtype=np.float32)

    if wave.ndim != 2 or wave.shape[1] != 3000:
        raise ValueError(f"Unexpected EEG shape {wave.shape}")

    if not np.isfinite(wave).all():
        raise RuntimeError("Non-finite EEG")

    psd, freqs = proc.eeg_to_multitaper_spectrogram(
        wave.reshape(-1),
        fs_eeg=100.0,
        epoch_sec=30,
        fmin=0.5,
        fmax=32.0,
    )

    psd, _ = proc.resample_frequency_axis(
        psd, freqs, n_freq_bins=256
    )
    db = proc.spectrogram_to_db(psd)

    result = np.clip(
        (db - low) / (high - low), 0, 1
    ).astype(np.float32)

    assert result.shape == (256, len(wave))
    return result


def load_split(split):
    folder = SOURCE / split
    meta = json.loads((folder / "metadata.json").read_text())

    rows = [
        json.loads(line)
        for line in (folder / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]

    shape = tuple(meta["eeg_shape"])
    label_shape = tuple(meta["label_shape"])

    if shape != (len(rows), 256, 512):
        raise RuntimeError("Invalid PaSL EEG shape")
    if label_shape != (len(rows), 512):
        raise RuntimeError("Invalid PaSL label shape")

    ground = np.memmap(
        folder / "ground_truth_eeg.mmap",
        dtype=np.dtype(meta["eeg_dtype"]),
        mode="r",
        shape=shape,
    )

    labels = np.memmap(
        folder / "labels.mmap",
        dtype=np.dtype(meta["label_dtype"]),
        mode="r",
        shape=label_shape,
    )

    ds = PreprocessedPSGDataset(split=split)

    cache = (
        AD / "outputs/downstream/staging/stage_cache"
    )

    stage_meta = json.loads(
        (cache / f"metadata_{split}.json").read_text()
    )

    if (
        stage_meta["records"] != [
            r["recording"] for r in ds.records
        ]
        or stage_meta["windows"] != len(ds)
    ):
        raise RuntimeError("SHHS1 stage cache mismatch")

    stages = np.load(
        cache / f"stages_{split}.npy",
        mmap_mode="r",
    )

    eeg = np.memmap(
        ds.paths["eeg"],
        dtype=np.float32,
        mode="r",
        shape=tuple(ds.metadata["eeg"]["data_shape"]),
    )

    records = {}
    offset = 0

    for record in ds.records:
        records[ident(record["recording"])] = (
            int(record["positions"]["eeg"]),
            offset,
            int(record["windows"]),
        )
        offset += int(record["windows"])

    mapping = []

    for i, row in enumerate(rows):
        sid = ident(row["subject_id"])

        if sid not in records:
            raise RuntimeError(f"Subject missing in adaptive split: {sid}")

        signal_start, stage_start, n_epochs = records[sid]
        start = int(row["window_index"]) * 512

        if row.get("start_sec") is not None:
            if abs(float(row["start_sec"]) / 30 - start) > 1e-4:
                raise RuntimeError(f"Start time mismatch: {sid}")

        if start + 512 > n_epochs:
            raise RuntimeError(f"Window outside recording: {sid}")

        expected = STAGE_TO_4[
            np.asarray(
                stages[stage_start + start:stage_start + start + 512],
                dtype=np.int64,
            )
        ]

        actual = np.asarray(labels[i], dtype=np.int64)
        valid = (actual >= 0) & (actual < 4)

        if not np.array_equal(actual[valid], expected[valid]):
            raise RuntimeError(f"LABEL MISALIGNMENT: {sid}")

        mapping.append((signal_start + start, sid, start))

    print(
        f"{split}: {len(mapping)} windows / "
        f"{len(ds.records)} recordings VERIFIED",
        flush=True,
    )

    return folder, meta, mapping, eeg, ground


def reconstruct(name, model, agents, codes, sample_ids):
    batch = len(codes)
    device = codes.device

    if name.startswith("uniform"):
        n = int(name.removeprefix("uniform"))
        if n not in (16, 64, 128):
            raise ValueError(name)
        return model.reconstructor(codes, n).float(), n

    if name == "fixed80":
        levels = torch.ones(
            batch, 16, device=device, dtype=torch.long
        )
        levels[:, [0, 4, 8, 12]] = 2
        return (
            decode_mixed(model.reconstructor, codes, levels).float(),
            80,
        )

    if name == "random80":
        states = []
        for sid in sample_ids:
            seed = int(
                hashlib.sha256(sid.encode()).hexdigest()[:16],
                16,
            )
            rng = random.Random(seed)
            levels = [0] * 16
            spent = 16

            while True:
                choices = [
                    i for i, level in enumerate(levels)
                    if level < 2
                    and spent + (3 if level == 0 else 4) <= 80
                ]
                if not choices:
                    break
                r = rng.choice(choices)
                spent += 3 if levels[r] == 0 else 4
                levels[r] += 1

            states.append(levels)

        states = torch.tensor(
            states, device=device, dtype=torch.long
        )
        return (
            decode_mixed(model.reconstructor, codes, states).float(),
            80,
        )

    match = re.fullmatch(r"(V4|V5|V7)_(64|80|96)", name)
    if not match:
        raise ValueError(name)

    agent = agents[match.group(1)]
    levels, used = fixed_budget(
        agent, codes, int(match.group(2))
    )

    return (
        decode_mixed(model.reconstructor, codes, levels).float(),
        used,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "build"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--condition", default="V4_80")
    parser.add_argument("--norm-file")
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-windows", type=int, default=2)
    parser.add_argument("--min-corr", type=float, default=0.95)
    parser.add_argument("--max-mae", type=float, default=0.08)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.microbatch < 1:
        raise ValueError("Invalid microbatch")

    low, high = find_norm(args.norm_file)
    proc = load_preprocessing()

    folder, meta, mapping, eeg, ground = load_split(args.split)

    # Confirm that original adaptive EEG matches the PaSL reference.
    count = min(args.audit_windows, len(mapping))

    for i in range(count):
        start, sid, epoch = mapping[i]
        raw = np.asarray(eeg[start:start+512], dtype=np.float32)

        pieces = [
            spectrum(
                raw[j:j+args.microbatch], proc, low, high
            )
            for j in range(0, 512, args.microbatch)
        ]
        reconstructed_original = np.concatenate(pieces, axis=1)
        reference = np.asarray(ground[i], dtype=np.float32)

        corr = float(np.corrcoef(
            reconstructed_original.ravel(),
            reference.ravel(),
        )[0, 1])

        mae = float(np.abs(
            reconstructed_original - reference
        ).mean())

        print(
            f"AUDIT {args.split} window {i} "
            f"corr={corr:.5f} MAE={mae:.5f}",
            flush=True,
        )

        if (
            not np.isfinite(corr)
            or corr < args.min_corr
            or mae > args.max_mae
        ):
            raise RuntimeError(
                "Spectrogram mismatch: STOP. "
                "Do not compare downstream scores until "
                "the PaSL preprocessing is aligned."
            )

    print("AUDIT PASSED", flush=True)

    if args.mode == "audit":
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    target = args.out / args.condition / "shhs1" / args.split
    complete = target / ".complete"

    if complete.exists():
        print("Already completed:", target)
        return

    target.mkdir(parents=True, exist_ok=True)

    for filename in (
        "metadata.json",
        "index.jsonl",
        "labels.mmap",
        "ground_truth_eeg.mmap",
    ):
        link = target / filename
        if not link.exists():
            link.symlink_to((folder / filename).resolve())

    temporary = target / "synthesized_eeg.tmp"
    final = target / "synthesized_eeg.mmap"

    # Resume interrupted V8 generation without deleting completed windows.
    resume_from = 0
    shape = tuple(meta["eeg_shape"])
    dtype = np.dtype(meta["eeg_dtype"])

    if args.resume:
        if args.condition != "V8_80":
            raise ValueError("Resume is currently supported only for V8_80")

        if not temporary.is_file():
            raise FileNotFoundError(temporary)

        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        if temporary.stat().st_size != expected_bytes:
            raise RuntimeError("Temporary file size does not match metadata")

        output = np.memmap(
            temporary, mode="r+", dtype=dtype, shape=shape
        )

        bins = [10, 30, 60, 90, 130, 170, 210, 240]
        finished = np.any(
            output[:, bins, -1] != 0, axis=1
        )

        missing = np.flatnonzero(~finished)
        resume_from = int(missing[0]) if len(missing) else len(mapping)

        if finished[resume_from:].any():
            raise RuntimeError(
                "Non-contiguous completed windows: unsafe to resume"
            )

        print(
            f"RESUMING {args.condition} {args.split}: "
            f"{resume_from}/{len(mapping)} windows completed",
            flush=True,
        )
    else:
        if temporary.exists():
            temporary.unlink()

        output = np.memmap(
            temporary, mode="w+", dtype=dtype, shape=shape
        )

    model = PhysioTokenizerV3().cuda().eval()
    ckpt = torch.load(
        AD / "outputs/checkpoints/timetok_v3_eeg_ae/frozen_for_agent.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    norm = ckpt["normalization"]

    agents = {}
    match = re.fullmatch(r"(V4|V5|V7)_(64|80|96)", args.condition)

    if match:
        agent, _ = load_agent(
            str(AGENTS[match.group(1)]),
            torch.device("cuda"),
        )
        agents[match.group(1)] = agent.eval()

    total_tokens = float(resume_from * 512 * 80)
    total_epochs = resume_from * 512

    with torch.inference_mode():
        for i, (start, sid, first_epoch) in enumerate(mapping):
            if i < resume_from:
                continue
            for begin in range(0, 512, args.microbatch):
                end = min(begin + args.microbatch, 512)

                raw = np.array(
                    eeg[start+begin:start+end],
                    dtype=np.float32,
                    copy=True,
                )

                x = torch.from_numpy(raw).cuda().unsqueeze(1)
                x = (
                    (x - float(norm["mean"])) / float(norm["std"])
                ).clamp(-8, 8)

                codes, _ = model.tokenize(x)

                sample_ids = [
                    f"{sid}:{first_epoch+j}"
                    for j in range(begin, end)
                ]

                rec, used = reconstruct(
                    args.condition, model, agents, codes, sample_ids
                )

                waves = (
                    rec.squeeze(1).float()
                    * float(norm["std"])
                    + float(norm["mean"])
                ).cpu().numpy()

                output[i, :, begin:end] = spectrum(
                    waves, proc, low, high
                ).astype(output.dtype)

                if torch.is_tensor(used):
                    total_tokens += used.float().sum().item()
                else:
                    total_tokens += float(used) * (end - begin)

                total_epochs += end - begin

            if (i+1) % 25 == 0 or i+1 == len(mapping):
                print(
                    f"BUILD {args.condition} {args.split}: "
                    f"{i+1}/{len(mapping)} windows",
                    flush=True,
                )

    output.flush()
    del output
    os.replace(temporary, final)

    complete.write_text(json.dumps({
        "condition": args.condition,
        "split": args.split,
        "windows": len(mapping),
        "mean_tokens": total_tokens / total_epochs,
    }, indent=2))

    print(
        "COMPLETE:", final,
        "mean tokens:", total_tokens / total_epochs,
        flush=True,
    )


if __name__ == "__main__":
    main()
