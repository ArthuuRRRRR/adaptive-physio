
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.encoders.register_encoder import RegisterEncoder
from adaptive_physio.tokenizers.fsq import FSQQuantizer
from adaptive_physio.decoders.conditional_flow_decoder import (
    ConditionalFlowDecoder,
)
from adaptive_physio.training.granularity_targets import (
    GaussianGranularityTargets,
)
from adaptive_physio.training.granularity_config import (
    TOKEN_BUDGETS,
    SMOOTHING,
)
from adaptive_physio.training.flow_matching import (
    flow_matching_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)
    print("Loading preprocessed SHHS1...", flush=True)

    dataset = PreprocessedPSGDataset(
        split="train",
        max_records=1,
    )

    if args.windows > len(dataset):
        raise ValueError("Not enough windows")

    # Small EEG-only dataset for controlled overfitting.
    x = torch.stack(
        [dataset[i]["eeg"] for i in range(args.windows)]
    ).to(device)

    # Fixed normalization on the overfitting subset.
    # This is NOT our final training normalization.
    mean = x.mean().detach()
    std = x.std(unbiased=False).clamp_min(1e-6).detach()

    x = (x - mean) / std

    print("Input:", tuple(x.shape))
    print("Raw mean:", round(mean.item(), 5))
    print("Raw std:", round(std.item(), 5))

    # Build coarse/mid/fine targets once.
    targets = GaussianGranularityTargets(
        sample_rate=100,
        sigmas_seconds=SMOOTHING["eeg"],
    )(x)

    encoder = RegisterEncoder(
        sample_rate=100,
        patch_seconds=0.2,
        embedding_dim=128,
        num_registers=128,
    ).to(device)

    quantizer = FSQQuantizer(
        input_dim=128,
        levels=(4, 4, 4, 4, 4, 4),
    ).to(device)

    decoder = ConditionalFlowDecoder(
        code_dim=6,
        hidden_dim=64,
        downsample=5,
    ).to(device)

    parameters = (
        list(encoder.parameters())
        + list(quantizer.parameters())
        + list(decoder.parameters())
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=1e-4,
    )

    levels = ("coarse", "mid", "fine")

    print("\nTRAINING STARTED", flush=True)

    recent_losses = []

    encoder.train()
    quantizer.train()
    decoder.train()

    for step in range(1, args.steps + 1):
        batch_ids = torch.randint(
            len(x),
            (args.batch_size,),
            device=device,
        )

        level = levels[(step - 1) % len(levels)]
        prefix = TOKEN_BUDGETS[level]

        batch = x[batch_ids]
        target = targets[level][batch_ids]

        registers = encoder(batch)
        codes, indices = quantizer(registers)

        loss = flow_matching_loss(
            decoder=decoder,
            target=target,
            codes=codes,
            prefix_length=prefix,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=1.0
        )

        optimizer.step()

        recent_losses.append(loss.item())

        if step % 50 == 0 or step == 1:
            avg = sum(recent_losses) / len(recent_losses)

            print(
                f"Step {step:4d}/{args.steps} "
                f"| level={level:6s} "
                f"| loss={loss.item():.4f} "
                f"| recent_avg={avg:.4f}",
                flush=True,
            )

            recent_losses.clear()

    # Simple conditioning diagnostic with matched noise.
    encoder.eval()
    quantizer.eval()
    decoder.eval()

    print("\nCONDITIONING DIAGNOSTIC")

    with torch.no_grad():
        subset = x[:min(4, len(x))]
        codes, _ = quantizer(encoder(subset))

        for level in levels:
            n = TOKEN_BUDGETS[level]
            target = targets[level][:len(subset)]

            torch.manual_seed(123)
            eps = torch.randn_like(target)
            tau = torch.full(
                (len(subset),),
                0.5,
                device=device,
            )

            x_tau = 0.5 * eps + 0.5 * target
            velocity = target - eps

            correct = decoder(
                x_tau, tau, codes[:, :n]
            )

            # Condition on another window's tokens.
            shuffled = decoder(
                x_tau,
                tau,
                torch.roll(codes[:, :n], 1, 0),
            )

            correct_loss = F.mse_loss(
                correct, velocity
            ).item()

            shuffled_loss = F.mse_loss(
                shuffled, velocity
            ).item()

            print(
                f"{level:6s} | "
                f"correct={correct_loss:.4f} | "
                f"shuffled={shuffled_loss:.4f}"
            )

    output = Path("outputs/checkpoints")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = output / (
        f"overfit_eeg_{args.steps}steps.pt"
    )

    torch.save({
        "encoder": encoder.state_dict(),
        "quantizer": quantizer.state_dict(),
        "decoder": decoder.state_dict(),
        "modality": "eeg",
        "sampling_rate": 100,
        "normalization": {
            "mean": mean.item(),
            "std": std.item(),
        },
        "token_budgets": TOKEN_BUDGETS,
        "smoothing": SMOOTHING["eeg"],
        "steps": args.steps,
    }, checkpoint)

    print("\nCheckpoint:", checkpoint)
    print("TRAINING COMPLETED")


if __name__ == "__main__":
    main()
