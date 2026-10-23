"""V5 EEG agent with a frozen pretrained SingLEM teacher."""

import argparse
import csv
import hashlib
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_physio.models.temporal_agent_v3 import (
    TemporalRefinementAgent,
)
from adaptive_physio.models.timetok_physio_v3 import (
    reconstruction_stats,
)
from evaluate_timetok_v3_adaptive import decode_mixed
from train_temporal_agent_v3 import (
    data_loader,
    load_backbone,
    normalize,
)
from train_temporal_agent_v4 import (
    quality_error,
    random_states,
    load_agent,
    rollout_lambdas,
)

SCALE = 100.0


class FrozenSingLEM(nn.Module):
    """Single-channel frozen EEG representation teacher."""

    def __init__(self, repo):
        super().__init__()
        root = Path(repo).expanduser().resolve()
        model_file = root / "SingLEM" / "model.py"
        checkpoint = (
            root / "SingLEM" / "checkpoints"
            / "singlem_downstream_excluded.pt"
        )

        if not model_file.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing SingLEM files: {model_file}, {checkpoint}"
            )

        spec = importlib.util.spec_from_file_location(
            "external_singlem_model", model_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        expected = {
            "sample_rate_hz": 128.0,
            "token_samples": 128,
            "input_unit": "microvolt",
            "input_scale": 0.01,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Unexpected SingLEM {key}: {payload.get(key)}"
                )

        config = module.Config()
        for key, value in payload["model_config"].items():
            setattr(config, key, value)

        config.mask_prob = 0.0

        if config.max_seq_len < 13:
            raise ValueError("SingLEM sequence limit too short")

        self.encoder = module.EEGEncoder(config)
        self.encoder.load_state_dict(
            payload["encoder_state_dict"],
            strict=True,
        )
        self.encoder.requires_grad_(False).eval()

        self.sha256 = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()

        self.checkpoint = str(checkpoint)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x, norm):
        """
        x: standardized EEG [B,1,3000], 100 Hz.
        Output: 39 SingLEM embeddings per 30-second epoch.
        """
        if x.ndim != 3 or x.shape[1:] != (1, 3000):
            raise ValueError("Expected [B,1,3000] EEG")

        # Restore physical microvolt scale.
        eeg_uv = (
            x.float() * float(norm["std"])
            + float(norm["mean"])
        )

        # Robust range for pretrained EEG teacher.
        eeg_uv = eeg_uv.clamp(-100.0, 100.0)

        # 100 Hz -> 128 Hz, 30 seconds.
        eeg = F.interpolate(
            eeg_uv,
            size=3840,
            mode="linear",
            align_corners=False,
        )

        # SingLEM scales microvolts by 0.01.
        eeg = eeg * 0.01

        # Three 10-second chunks to respect max_seq_len=16.
        eeg = eeg.reshape(-1, 1, 1280)

        # 1-second tokens, 0.75-second hop.
        tokens = eeg[:, 0].unfold(
            -1, 128, 96
        ).contiguous()

        features, _, _ = self.encoder(tokens)

        return features.reshape(
            -1, 39, features.shape[-1]
        ).float()


def physio_error(features, reference):
    """Local cosine distance between pretrained EEG embeddings."""
    a = F.normalize(
        features.float(), dim=-1, eps=1e-6
    )
    b = F.normalize(
        reference.float(), dim=-1, eps=1e-6
    )

    return (
        1.0 - (a * b).sum(-1)
    ).clamp_min(0).mean(-1)


def get_codes(backbone, x):
    with torch.no_grad():
        with torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            codes, _ = backbone.tokenize(x)
    return codes.float()


@torch.no_grad()
def make_targets(
    backbone, teacher, codes, x, norm, weight
):
    b = len(x)

    # Legal mixed granularity states.
    states = random_states(b, x.device)

    eligible = (states < 2).float()
    actions = torch.multinomial(
        eligible,
        num_samples=2,
        replacement=True,
    )

    kinds = states.gather(1, actions)
    costs = torch.where(kinds == 0, 3, 4).float()

    # Apply one candidate refinement at a time.
    after_states = (
        states[:, None].expand(-1, 2, -1).clone()
    )
    after_states.scatter_(
        2,
        actions[..., None],
        kinds[..., None] + 1,
    )

    before = decode_mixed(
        backbone.reconstructor,
        codes,
        states,
    ).float()

    after = decode_mixed(
        backbone.reconstructor,
        codes.repeat_interleave(2, 0),
        after_states.reshape(b * 2, 16),
    ).float()

    x2 = x.repeat_interleave(2, 0)

    # Classical reconstruction-quality gain.
    wave_before = quality_error(before, x)
    wave_after = quality_error(
        after, x2
    ).reshape(b, 2)

    # NEW: pretrained physiological feature gain.
    reference_features = teacher(x, norm)
    before_features = teacher(before, norm)
    after_features = teacher(after, norm)

    phys_before = physio_error(
        before_features,
        reference_features,
    )
    phys_after = physio_error(
        after_features,
        reference_features.repeat_interleave(2, 0),
    ).reshape(b, 2)

    wave_gain = (
        wave_before[:, None] - wave_after
    ) / costs

    phys_gain = (
        phys_before[:, None] - phys_after
    ) / costs

    # Target: reduction of mixed reconstruction distortion
    # per newly revealed token.
    target = SCALE * (
        wave_gain + weight * phys_gain
    )

    return (
        codes,
        states,
        actions,
        kinds,
        target,
        wave_gain,
        phys_gain,
    )


def controller_loss(agent, data):
    codes, states, actions, kinds, target, _, _ = data

    scores = agent(codes, states)

    chosen = scores.gather(
        1,
        actions[..., None].expand(-1, -1, 2),
    )

    predicted = chosen.gather(
        2, kinds[..., None]
    ).squeeze(-1)

    regression = F.smooth_l1_loss(
        predicted, target, beta=0.15
    )

    ranking = F.smooth_l1_loss(
        predicted[:, 0] - predicted[:, 1],
        target[:, 0] - target[:, 1],
        beta=0.15,
    )

    loss = regression + 0.2 * ranking

    error = (predicted - target).abs().mean()

    return loss, error


@torch.no_grad()
def validate(
    agent, backbone, teacher,
    loader, norm, weight, device
):
    agent.eval()
    total_loss = 0.0
    total_error = 0.0
    count = 0

    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()]
    ):
        torch.manual_seed(1729)

        for raw in loader:
            x = normalize(raw, norm, device)

            data = make_targets(
                backbone,
                teacher,
                get_codes(backbone, x),
                x,
                norm,
                weight,
            )

            loss, error = controller_loss(
                agent, data
            )

            total_loss += loss.item() * len(x)
            total_error += error.item() * len(x)
            count += len(x)

    agent.train()

    return (
        total_loss / count,
        total_error / count,
    )


def setup(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")

    backbone, bc = load_backbone(
        args.backbone, device
    )

    teacher = FrozenSingLEM(
        args.singlem
    ).to(device).eval()

    print(
        "V3 backbone step:", bc["step"],
        flush=True
    )
    print(
        "SingLEM pretrained:", teacher.checkpoint,
        flush=True
    )
    print(
        "SingLEM SHA256:", teacher.sha256,
        flush=True
    )
    print(
        "Normalization:", bc["normalization"],
        flush=True
    )

    return device, backbone, bc, teacher


def smoke(args, device, backbone, bc, teacher):
    loader, _ = data_loader(
        "train", "eeg", 4, 4, 0, args.seed
    )

    x = normalize(
        next(iter(loader)),
        bc["normalization"],
        device,
    )

    codes = get_codes(backbone, x)

    with torch.no_grad():
        original_features = teacher(
            x, bc["normalization"]
        )

        print(
            "SingLEM features:",
            tuple(original_features.shape),
            flush=True,
        )

        for keep in (16, 64, 128):
            reconstructed = backbone.reconstructor(
                codes, keep
            ).float()

            features = teacher(
                reconstructed,
                bc["normalization"],
            )

            distance = physio_error(
                features,
                original_features,
            )

            print(
                f"Uniform {keep:3d} | "
                f"Physio distance={distance.mean().item():.6f}",
                flush=True,
            )

        shuffled = x.roll(1, 0)

        distance_shuffled = physio_error(
            teacher(
                shuffled, bc["normalization"]
            ),
            original_features,
        ).mean()

        print(
            "Shuffled EEG distance:",
            distance_shuffled.item(),
            flush=True,
        )

        data = make_targets(
            backbone,
            teacher,
            codes,
            x,
            bc["normalization"],
            args.physio_weight,
        )

    agent = TemporalRefinementAgent().to(device)

    loss, error = controller_loss(agent, data)
    loss.backward()

    grads = [
        p.grad for p in agent.parameters()
        if p.grad is not None
    ]

    assert grads
    assert all(
        bool(torch.isfinite(g).all())
        for g in grads
    )

    wave = data[-2].abs().mean()
    phys = data[-1].abs().mean()

    fraction = (
        args.physio_weight * phys
        / (
            wave
            + args.physio_weight * phys
            + 1e-9
        )
    )

    print(
        "Wave gain/token:",
        wave.item(),
        flush=True,
    )
    print(
        "Physio gain/token:",
        phys.item(),
        flush=True,
    )
    print(
        "Weighted physiological fraction:",
        fraction.item(),
        flush=True,
    )
    print(
        "SMOKE TEST PASSED",
        f"loss={loss.item():.5f}",
        f"error={error.item():.5f}",
        flush=True,
    )


def train(args, device, backbone, bc, teacher):
    agent = TemporalRefinementAgent().to(device)

    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    batches = math.ceil(
        args.train_windows / args.batch_size
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=batches * args.epochs,
            eta_min=args.lr * 0.05,
        )
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True, exist_ok=True
    )

    validation, _ = data_loader(
        "val",
        "eeg",
        args.val_windows,
        args.batch_size,
        min(args.workers, 2),
        args.seed,
    )

    best = float("inf")
    initial_epoch = 0
    step = 0

    if args.resume:
        ck = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        required = (
            ("teacher_sha256", teacher.sha256),
            ("backbone_step", bc["step"]),
            ("physio_weight", args.physio_weight),
            ("train_windows", args.train_windows),
            ("batch_size", args.batch_size),
            ("epochs", args.epochs),
            ("seed", args.seed),
        )

        for key, expected in required:
            if ck[key] != expected:
                raise ValueError(
                    f"Resume mismatch: {key}"
                )

        agent.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])

        initial_epoch = ck["epoch"]
        step = ck["step"]
        best = ck["best"]

        torch.set_rng_state(ck["cpu_rng"])
        torch.cuda.set_rng_state(ck["gpu_rng"])

        print(
            "Resumed from epoch", initial_epoch,
            flush=True,
        )

    print(
        f"Training: {args.epochs} epochs, "
        f"{args.train_windows} windows/epoch, "
        f"batch={args.batch_size}, "
        f"physio_weight={args.physio_weight}",
        flush=True,
    )

    for epoch in range(initial_epoch, args.epochs):
        agent.train()

        loader, _ = data_loader(
            "train",
            "eeg",
            args.train_windows,
            args.batch_size,
            args.workers,
            args.seed,
            epoch=epoch,
        )

        for index, raw in enumerate(loader, 1):
            x = normalize(
                raw,
                bc["normalization"],
                device,
            )

            codes = get_codes(backbone, x)

            data = make_targets(
                backbone,
                teacher,
                codes,
                x,
                bc["normalization"],
                args.physio_weight,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, error = controller_loss(
                agent, data
            )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), 1.0
            )

            optimizer.step()
            scheduler.step()
            step += 1

            if (
                index == 1
                or index % args.log_every == 0
            ):
                print(
                    f"epoch={epoch+1}/{args.epochs} "
                    f"step={step} "
                    f"loss={loss.item():.5f} "
                    f"gain_error={error.item():.5f}",
                    flush=True,
                )

        val_loss, val_error = validate(
            agent,
            backbone,
            teacher,
            validation,
            bc["normalization"],
            args.physio_weight,
            device,
        )

        improved = val_loss < best
        best = min(best, val_loss)

        ck = {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": step,
            "best": best,
            "backbone_step": bc["step"],
            "teacher_sha256": teacher.sha256,
            "teacher_checkpoint": teacher.checkpoint,
            "physio_weight": args.physio_weight,
            "gain_scale": SCALE,
            "train_windows": args.train_windows,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "cpu_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state(),
        }

        temporary = outdir / "last.pt.tmp"

        torch.save(ck, temporary)
        os.replace(
            temporary,
            outdir / "last.pt",
        )

        if improved:
            temporary = outdir / "best.pt.tmp"

            torch.save(ck, temporary)
            os.replace(
                temporary,
                outdir / "best.pt",
            )

        print(
            f"VAL epoch={epoch+1} "
            f"loss={val_loss:.6f} "
            f"error={val_error:.6f} "
            f"best={best:.6f}",
            flush=True,
        )

    print(
        "PHYSIO AGENT TRAINING FINISHED",
        flush=True,
    )


@torch.no_grad()
def evaluate(args, device, backbone, bc, teacher):
    agent, ck = load_agent(
        args.agent_checkpoint, device
    )

    if (
        ck.get("teacher_sha256") != teacher.sha256
        or ck.get("backbone_step") != bc["step"]
    ):
        raise ValueError(
            "Teacher/backbone checkpoint mismatch"
        )

    agent.eval()

    loader, n = data_loader(
        "val",
        "eeg",
        args.eval_windows,
        args.batch_size,
        min(args.workers, 2),
        args.seed,
    )

    lambdas = args.lambdas
    accum = np.zeros(
        (len(lambdas), 4),
        dtype=float,
    )

    seen = 0

    for raw in loader:
        x = normalize(
            raw, bc["normalization"], device
        )

        codes = get_codes(backbone, x)

        original_features = teacher(
            x, bc["normalization"]
        )

        repeated, states, used = rollout_lambdas(
            agent,
            codes,
            lambdas,
            maximum=128,
        )

        reconstructed = decode_mixed(
            backbone.reconstructor,
            repeated,
            states,
        ).float()

        for j in range(len(lambdas)):
            idx = torch.arange(
                j,
                len(reconstructed),
                len(lambdas),
                device=device,
            )

            y = reconstructed.index_select(
                0, idx
            )

            mae, _, corr = reconstruction_stats(
                y, x
            )

            physio = physio_error(
                teacher(y, bc["normalization"]),
                original_features,
            )

            accum[j] += [
                mae.sum().item(),
                corr.sum().item(),
                used.index_select(
                    0, idx
                ).sum().item(),
                physio.sum().item(),
            ]

        seen += len(x)

        if seen % 128 == 0 or seen == n:
            print(
                f"Evaluated {seen}/{n}",
                flush=True,
            )

    results = []

    print(
        "\nLambda | Tokens | MAE | Corr | Physio distance"
    )

    for i, lam in enumerate(lambdas):
        mae, corr, tokens, physio = (
            accum[i] / seen
        )

        row = {
            "lambda_cost": lam,
            "avg_used": tokens,
            "mae": mae,
            "corr": corr,
            "physio_cosine_distance": physio,
        }
        results.append(row)

        print(
            f"{lam:.5f} | "
            f"{tokens:.1f} | "
            f"{mae:.4f} | "
            f"{corr:.4f} | "
            f"{physio:.5f}"
        )

    path = Path(args.output)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print("Saved:", path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=("smoke", "train", "eval"),
        default="smoke",
    )
    p.add_argument(
        "--backbone",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
    )
    p.add_argument(
        "--singlem",
        default="third_party/SingLEM",
    )
    p.add_argument(
        "--agent-checkpoint",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_agent_physio_eeg/best.pt"
        ),
    )
    p.add_argument(
        "--physio-weight",
        type=float, default=1.0,
    )
    p.add_argument(
        "--train-windows",
        type=int, default=12000,
    )
    p.add_argument(
        "--val-windows",
        type=int, default=256,
    )
    p.add_argument(
        "--eval-windows",
        type=int, default=512,
    )
    p.add_argument(
        "--batch-size",
        type=int, default=8,
    )
    p.add_argument(
        "--epochs", type=int, default=3
    )
    p.add_argument(
        "--workers", type=int, default=2
    )
    p.add_argument(
        "--lr", type=float, default=3e-4
    )
    p.add_argument(
        "--log-every", type=int, default=100
    )
    p.add_argument(
        "--seed", type=int, default=42
    )
    p.add_argument("--resume")
    p.add_argument(
        "--output-dir",
        default="outputs/checkpoints/timetok_v3_agent_physio_eeg",
    )
    p.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[
            0.0, 0.0002, 0.0004, 0.0008,
            0.0012, 0.002, 0.004,
        ],
    )
    p.add_argument(
        "--output",
        default=(
            "outputs/evaluation/timetok_v3/"
            "physio_v5/validation_sweep.csv"
        ),
    )

    args = p.parse_args()

    if (
        args.physio_weight < 0
        or args.batch_size <= 0
        or args.train_windows <= 0
    ):
        p.error("Invalid training parameters")

    device, backbone, bc, teacher = setup(args)

    if args.mode == "smoke":
        smoke(
            args, device, backbone, bc, teacher
        )
    elif args.mode == "train":
        train(
            args, device, backbone, bc, teacher
        )
    else:
        evaluate(
            args, device, backbone, bc, teacher
        )


if __name__ == "__main__":
    main()
