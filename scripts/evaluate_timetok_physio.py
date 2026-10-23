"""Evaluate generated SHHS1 EEG at 16/64/128 tokens (no training)."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from adaptive_physio.data.preprocessed_dataset import PreprocessedPSGDataset
from adaptive_physio.models.timetok_physio import LEVELS, TimeTokPhysio, gaussian_target
from train_timetok_physio import OneModality


def statistics(pred, reference, hz=100):
    p = pred.float().flatten(1)
    r = reference.float().flatten(1)
    delta = p - r

    mae = delta.abs().mean(1)
    rmse = delta.square().mean(1).sqrt()

    pc = p - p.mean(1, keepdim=True)
    rc = r - r.mean(1, keepdim=True)
    corr = (pc * rc).mean(1) / (
        (pc.square().mean(1) * rc.square().mean(1)).sqrt().clamp_min(1e-8)
    )

    hann = torch.hann_window(p.shape[1], periodic=True, device=p.device)
    freqs = torch.fft.rfftfreq(
        p.shape[1], d=1.0 / hz, device=p.device
    )
    pp = torch.fft.rfft(p * hann, dim=1).abs().square()
    rr = torch.fft.rfft(r * hann, dim=1).abs().square()

    band_errors = []
    for low, high in ((0.5, 4), (4, 8), (8, 13), (13, 30)):
        select = (freqs >= low) & (freqs < high)
        pb = pp[:, select].sum(1).clamp_min(1e-8)
        rb = rr[:, select].sum(1).clamp_min(1e-8)
        band_errors.append((10 * torch.log10(pb / rb)).abs())

    band_db = torch.stack(band_errors, 0).mean(0)

    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "band_db": band_db,
    }


@torch.inference_mode()
def sample_flow(model, codes, keep, initial_noise, steps):
    # Flow was trained with:
    # x_t = t * noise + (1-t) * clean
    # velocity = noise - clean
    # Integrate backwards from t=1 to t=0.
    x = initial_noise.clone().float()
    batch = x.shape[0]
    dt = 1.0 / steps

    for i in range(steps):
        time = torch.full(
            (batch,),
            1.0 - (i + 0.5) * dt,
            device=x.device,
        )
        velocity = model.decoder(x, time, codes, keep)
        x = x - dt * velocity.float()

        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"Nonfinite generated signal at solver step {i + 1}"
            )

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/timetok_epochs_eeg/best.pt",
    )
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/timetok_eeg",
    )
    args = parser.parse_args()

    if min(args.val_windows, args.batch_size, args.ode_steps) < 1:
        parser.error("All evaluation counts must be positive")

    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required")

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if ckpt.get("modality") != "eeg":
        raise ValueError("Expected an EEG checkpoint")

    model = TimeTokPhysio().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean = ckpt["normalization"]["mean"]
    std = ckpt["normalization"]["std"]

    print("Checkpoint:", args.checkpoint)
    print("Training step:", ckpt.get("step", "unknown"))

    source = PreprocessedPSGDataset(split="val")
    dataset = OneModality(source, "eeg")

    count = min(args.val_windows, len(dataset))
    indices = torch.linspace(
        0, len(dataset) - 1, steps=count
    ).long().tolist()

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=0,
    )

    collected = defaultdict(lambda: defaultdict(float))
    samples = defaultdict(int)
    ids_seen = {k: set() for k in LEVELS}
    saved_plot = None

    rng = torch.Generator(device=device).manual_seed(args.seed)

    for raw in loader:
        x = ((raw.to(device) - mean) / std).clamp(-8, 8)
        batch = x.shape[0]

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            codes, ids = model.fsq(model.encoder(x))
            noise = torch.randn(
                x.shape,
                device=device,
                dtype=x.dtype,
                generator=rng,
            )

            plot_data = (
                {"original": x[0, 0].float().cpu()}
                if saved_plot is None else None
            )

            for k in LEVELS:
                keep = torch.full(
                    (batch,), k,
                    device=device,
                    dtype=torch.long,
                )

                target = gaussian_target(x, keep)
                generated = sample_flow(
                    model,
                    codes,
                    keep,
                    noise,
                    args.ode_steps,
                )

                ids_seen[k].update(
                    ids[:, :k].reshape(-1).cpu().tolist()
                )

                raw_stats = statistics(generated, x)
                target_stats = statistics(generated, target)

                for name, stats in (
                    ("original", raw_stats),
                    ("target", target_stats),
                ):
                    for metric, vals in stats.items():
                        collected[k][
                            f"{metric}_{name}"
                        ] += vals.sum().item()

                samples[k] += batch

                if plot_data is not None:
                    plot_data[str(k)] = (
                        generated[0, 0].float().cpu()
                    )

            if plot_data is not None:
                saved_plot = plot_data

        print(
            f"Evaluated "
            f"{sum(samples.values()) // len(LEVELS)}/{count} "
            f"windows",
            flush=True,
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    columns = [
        "tokens", "n_windows", "unique_prefix_ids",
        "mae_original", "rmse_original",
        "corr_original", "band_db_original",
        "mae_target", "rmse_target",
        "corr_target", "band_db_target",
    ]

    rows = []

    for k in LEVELS:
        row = {
            "tokens": k,
            "n_windows": samples[k],
            "unique_prefix_ids": len(ids_seen[k]),
        }
        row.update({
            name: value / samples[k]
            for name, value in collected[k].items()
        })
        rows.append(row)

    csv_path = output / "granularity_metrics.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== GENERATED EEG RECONSTRUCTION ===")
    print("Metrics in normalized EEG units")
    print(
        "tokens | unique | MAE-real | RMSE-real | "
        "Corr-real | BandErr-dB | MAE-target | Corr-target"
    )

    for r in rows:
        print(
            f"{r['tokens']:6d} | "
            f"{r['unique_prefix_ids']:6d} | "
            f"{r['mae_original']:.4f} | "
            f"{r['rmse_original']:.4f} | "
            f"{r['corr_original']:.4f} | "
            f"{r['band_db_original']:.2f} | "
            f"{r['mae_target']:.4f} | "
            f"{r['corr_target']:.4f}"
        )

    print("Saved CSV:", csv_path)

    if saved_plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            4, 1, figsize=(13, 9), sharex=True
        )

        curves = (
            ("original", "Original EEG"),
            ("16", "Coarse - 16 tokens"),
            ("64", "Mid - 64 tokens"),
            ("128", "Fine - 128 tokens"),
        )

        for ax, (name, title) in zip(axes, curves):
            y = saved_plot[name].numpy()[:1000]
            t = torch.arange(len(y)).numpy() / 100.0
            ax.plot(t, y, linewidth=0.7)
            ax.set_title(title)
            ax.set_ylabel("EEG (norm.)")
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("Time (seconds)")
        fig.tight_layout()

        plot_path = output / "waveforms_10s.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        print("Saved plot:", plot_path)


if __name__ == "__main__":
    main()
