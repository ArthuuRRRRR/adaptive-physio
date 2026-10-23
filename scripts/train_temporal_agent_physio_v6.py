"""V6: Local SingLEM supervision dominates waveform quality.

Only the temporal agent is trained.
V3 tokenizer and SingLEM remain frozen.
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from adaptive_physio.models.temporal_agent_v3 import (
    TemporalRefinementAgent,
)
from evaluate_timetok_v3_adaptive import decode_mixed
from train_temporal_agent_v4 import (
    quality_error,
    random_states,
)
from train_temporal_agent_physio_v5 import (
    controller_loss,
    data_loader,
    get_codes,
    normalize,
    setup,
)


def region_weights(device, sigma=1.25):
    # SingLEM: three 10-second chunks.
    # Each chunk produces 13 embeddings.
    # A token covers 1 second, with 0.75-second stride.
    times = torch.tensor(
        [
            10 * (i // 13) + 0.75 * (i % 13) + 0.5
            for i in range(39)
        ],
        dtype=torch.float32,
        device=device,
    )

    centers = (
        torch.arange(16, device=device) + 0.5
    ) * (30.0 / 16.0)

    weights = torch.exp(
        -0.5 * (
            (centers[:, None] - times[None])
            / sigma
        ) ** 2
    )

    return weights / weights.sum(
        dim=1, keepdim=True
    )


def feature_distances(features, reference, weights):
    a = F.normalize(
        features.float(), dim=-1, eps=1e-6
    )
    b = F.normalize(
        reference.float(), dim=-1, eps=1e-6
    )

    d = (
        1.0 - (a * b).sum(-1)
    ).clamp_min(0)

    global_distance = d.mean(dim=1)
    regional_distances = d @ weights.T

    return global_distance, regional_distances


@torch.no_grad()
def raw_gains(
    backbone, teacher, codes, x,
    norm, weights, local_weight,
):
    b = len(x)

    states = random_states(
        b, x.device
    )

    eligible = (states < 2).float()

    actions = torch.multinomial(
        eligible,
        num_samples=2,
        replacement=True,
    )

    kinds = states.gather(1, actions)

    costs = torch.where(
        kinds == 0, 3.0, 4.0
    )

    after_states = (
        states[:, None]
        .expand(-1, 2, -1)
        .clone()
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

    # Frozen pretrained physiological teacher.
    reference = teacher(x, norm)
    f_before = teacher(before, norm)
    f_after = teacher(after, norm)

    global_before, local_before_all = (
        feature_distances(
            f_before, reference, weights
        )
    )

    global_after, local_after_all = (
        feature_distances(
            f_after,
            reference.repeat_interleave(2, 0),
            weights,
        )
    )

    # Only evaluate local preservation around
    # the region modified by each candidate action.
    local_before = local_before_all.gather(
        1, actions
    )

    local_after = (
        local_after_all
        .reshape(b, 2, 16)
        .gather(2, actions[..., None])
        .squeeze(-1)
    )

    physiological_gain = (
        local_weight
        * (local_before - local_after)
        +
        (1.0 - local_weight)
        * (
            global_before[:, None]
            - global_after.reshape(b, 2)
        )
    ) / costs

    waveform_gain = (
        quality_error(before, x)[:, None]
        -
        quality_error(
            after,
            x.repeat_interleave(2, 0),
        ).reshape(b, 2)
    ) / costs

    return (
        codes,
        states,
        actions,
        kinds,
        physiological_gain,
        waveform_gain,
    )


def supervision(raw, calibration, args):
    codes, states, actions, kinds, pg, wg = raw

    physiological_term = (
        args.physio_ratio
        * pg
        / calibration["phys_scale"]
    )

    waveform_term = (
        (1.0 - args.physio_ratio)
        * wg
        / calibration["wave_scale"]
    )

    target = args.target_scale * (
        physiological_term + waveform_term
    )

    # Existing agent regression/ranking loss.
    return (
        codes,
        states,
        actions,
        kinds,
        target,
        waveform_term,
        physiological_term,
    )


@torch.no_grad()
def calibrate(
    backbone, teacher, args, norm,
    device, weights,
):
    loader, _ = data_loader(
        "train", "eeg",
        args.calibration_windows,
        args.batch_size,
        0,
        args.seed,
    )

    phys_sum = 0.0
    wave_sum = 0.0
    count = 0

    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()]
    ):
        torch.manual_seed(args.seed + 5101)

        for raw in loader:
            x = normalize(raw, norm, device)

            results = raw_gains(
                backbone,
                teacher,
                get_codes(backbone, x),
                x,
                norm,
                weights,
                args.local_weight,
            )

            pg, wg = results[-2:]

            phys_sum += pg.abs().sum().item()
            wave_sum += wg.abs().sum().item()
            count += pg.numel()

    phys = phys_sum / count
    wave = wave_sum / count

    if phys < 1e-9 or wave < 1e-9:
        raise RuntimeError(
            f"Degenerate gains: "
            f"phys={phys}, wave={wave}"
        )

    result = {
        "phys_scale": phys,
        "wave_scale": wave,
    }

    print(
        "Frozen calibration:",
        result,
        flush=True,
    )

    return result


@torch.no_grad()
def validate(
    agent, backbone, teacher, loader,
    norm, device, weights, calibration, args,
):
    agent.eval()

    loss_sum = 0.0
    error_sum = 0.0
    count = 0

    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()]
    ):
        torch.manual_seed(1729)

        for raw in loader:
            x = normalize(raw, norm, device)

            targets = supervision(
                raw_gains(
                    backbone,
                    teacher,
                    get_codes(backbone, x),
                    x,
                    norm,
                    weights,
                    args.local_weight,
                ),
                calibration,
                args,
            )

            loss, error = controller_loss(
                agent, targets
            )

            loss_sum += loss.item() * len(x)
            error_sum += error.item() * len(x)
            count += len(x)

    agent.train()

    return (
        loss_sum / count,
        error_sum / count,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=["smoke", "train"],
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
        "--output-dir",
        default=(
            "outputs/checkpoints/"
            "timetok_v3_agent_physio_local_v6"
        ),
    )

    p.add_argument(
        "--physio-ratio",
        type=float,
        default=0.85,
    )

    p.add_argument(
        "--local-weight",
        type=float,
        default=0.80,
    )

    p.add_argument(
        "--sigma",
        type=float,
        default=1.25,
    )

    p.add_argument(
        "--target-scale",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--calibration-windows",
        type=int,
        default=128,
    )

    p.add_argument(
        "--train-windows",
        type=int,
        default=30000,
    )

    p.add_argument(
        "--val-windows",
        type=int,
        default=512,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument("--resume")

    args = p.parse_args()

    if not (
        0 <= args.physio_ratio <= 1
        and 0 <= args.local_weight <= 1
    ):
        p.error(
            "Loss mixing coefficients must be in [0,1]"
        )

    if args.sigma <= 0 or args.batch_size <= 0:
        p.error("Sigma and batch size must be positive")

    device, backbone, bc, teacher = setup(args)

    weights = region_weights(
        device, args.sigma
    )

    agent = TemporalRefinementAgent().to(device)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    params = dict(
        physio_ratio=args.physio_ratio,
        local_weight=args.local_weight,
        sigma=args.sigma,
        target_scale=args.target_scale,
        train_windows=args.train_windows,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    if args.resume:
        ck = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        if (
            ck["teacher_sha256"] != teacher.sha256
            or ck["backbone_step"] != bc["step"]
            or ck["hparams"] != params
        ):
            raise ValueError(
                "Resume checkpoint mismatch"
            )

        calibration = ck["calibration"]
        agent.load_state_dict(ck["model"])

    else:
        calibration = calibrate(
            backbone,
            teacher,
            args,
            bc["normalization"],
            device,
            weights,
        )

    if args.mode == "smoke":
        loader, _ = data_loader(
            "train", "eeg", 4, 4, 0, args.seed
        )

        x = normalize(
            next(iter(loader)),
            bc["normalization"],
            device,
        )

        targets = supervision(
            raw_gains(
                backbone,
                teacher,
                get_codes(backbone, x),
                x,
                bc["normalization"],
                weights,
                args.local_weight,
            ),
            calibration,
            args,
        )

        loss, error = controller_loss(
            agent, targets
        )

        loss.backward()

        assert bool(torch.isfinite(loss))
        assert any(
            p.grad is not None
            for p in agent.parameters()
        )

        phys = targets[-1].abs().mean()
        wave = targets[-2].abs().mean()

        print(
            "Local supervision: "
            "16 regions / 39 embeddings"
        )

        print(
            "Physiological contribution:",
            float(phys / (phys + wave + 1e-9))
        )

        print(
            f"SMOKE PASSED "
            f"loss={loss.item():.6f} "
            f"error={error.item():.6f}"
        )
        return

    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(
        args.train_windows / args.batch_size
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps_per_epoch * args.epochs,
            eta_min=args.lr * 0.05,
        )
    )

    start_epoch = 0
    global_step = 0
    best = float("inf")

    if args.resume:
        optimizer.load_state_dict(
            ck["optimizer"]
        )
        scheduler.load_state_dict(
            ck["scheduler"]
        )

        start_epoch = ck["epoch"]
        global_step = ck["step"]
        best = ck["best"]

        torch.set_rng_state(ck["cpu_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])

    val_loader, _ = data_loader(
        "val", "eeg",
        args.val_windows,
        args.batch_size,
        min(args.workers, 2),
        args.seed,
    )

    print(
        "V6 training:",
        params,
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        loader, _ = data_loader(
            "train",
            "eeg",
            args.train_windows,
            args.batch_size,
            args.workers,
            args.seed,
            epoch=epoch,
        )

        agent.train()

        for i, raw in enumerate(loader, 1):
            x = normalize(
                raw, bc["normalization"], device
            )

            targets = supervision(
                raw_gains(
                    backbone,
                    teacher,
                    get_codes(backbone, x),
                    x,
                    bc["normalization"],
                    weights,
                    args.local_weight,
                ),
                calibration,
                args,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, error = controller_loss(
                agent, targets
            )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "Non-finite agent loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), 1.0
            )

            optimizer.step()
            scheduler.step()
            global_step += 1

            if (
                i == 1
                or i % args.log_every == 0
            ):
                print(
                    f"epoch={epoch+1}/{args.epochs} "
                    f"step={global_step} "
                    f"loss={loss.item():.5f} "
                    f"gain_error={error.item():.5f}",
                    flush=True,
                )

        val_loss, val_error = validate(
            agent,
            backbone,
            teacher,
            val_loader,
            bc["normalization"],
            device,
            weights,
            calibration,
            args,
        )

        improved = val_loss < best
        best = min(best, val_loss)

        checkpoint = {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
            "best": best,
            "backbone_step": bc["step"],
            "teacher_sha256": teacher.sha256,
            "teacher_checkpoint": teacher.checkpoint,
            "calibration": calibration,
            "hparams": params,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "version": "v6-local-physio-dominant",
        }

        temporary = output / "last.pt.tmp"

        torch.save(checkpoint, temporary)
        os.replace(
            temporary,
            output / "last.pt",
        )

        if improved:
            temporary = output / "best.pt.tmp"

            torch.save(checkpoint, temporary)
            os.replace(
                temporary,
                output / "best.pt",
            )

        print(
            f"VAL epoch={epoch+1} "
            f"loss={val_loss:.6f} "
            f"error={val_error:.6f} "
            f"best={best:.6f}",
            flush=True,
        )

    print("V6 FINISHED", flush=True)


if __name__ == "__main__":
    main()
