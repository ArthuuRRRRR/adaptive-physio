import torch
import torch.nn.functional as F

from adaptive_physio.data.preprocessed_dataset import (
    PreprocessedPSGDataset,
)
from adaptive_physio.encoders.register_encoder import (
    RegisterEncoder,
)
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


def main():
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = (
        "outputs/checkpoints/overfit_eeg_600steps.pt"
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    dataset = PreprocessedPSGDataset(
        split="train",
        max_records=1,
    )

    # Different windows from the training subset.
    indices = [0, 5, 10, 15]

    x = torch.stack(
        [dataset[i]["eeg"] for i in indices]
    ).to(device)

    # Reuse training normalization.
    mean = ckpt["normalization"]["mean"]
    std = ckpt["normalization"]["std"]
    x = (x - mean) / std

    encoder = RegisterEncoder(
        sample_rate=100,
        patch_seconds=0.2,
    ).to(device)

    quantizer = FSQQuantizer().to(device)

    decoder = ConditionalFlowDecoder(
        downsample=5,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    quantizer.load_state_dict(ckpt["quantizer"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    quantizer.eval()
    decoder.eval()

    targets = GaussianGranularityTargets(
        sample_rate=100,
        sigmas_seconds=SMOOTHING["eeg"],
    )(x)

    with torch.no_grad():
        codes, ids = quantizer(encoder(x))

        print("\n=== TOKEN DIAGNOSTICS ===")
        print("Codes:", tuple(codes.shape))
        print("IDs:", tuple(ids.shape))
        print("Unique token IDs:", torch.unique(ids).numel())

        difference = F.mse_loss(
            codes,
            torch.roll(codes, 1, 0),
        ).item()

        print(f"Between-window code MSE: {difference:.6f}")

        print("\n=== CONDITIONING DIAGNOSTICS ===")

        for tau_value in [0.05, 0.50, 0.95]:
            print(f"\nFlow time: {tau_value}")

            for level, n in TOKEN_BUDGETS.items():
                target = targets[level]
                noise = torch.randn_like(target)

                tau = torch.full(
                    (len(x),),
                    tau_value,
                    device=device,
                )

                t = tau[:, None, None]

                x_tau = (1 - t) * noise + t * target
                velocity = target - noise

                correct_codes = codes[:, :n]

                shuffled_codes = torch.roll(
                    correct_codes,
                    shifts=1,
                    dims=0,
                )

                correct_pred = decoder(
                    x_tau,
                    tau,
                    correct_codes,
                )

                shuffled_pred = decoder(
                    x_tau,
                    tau,
                    shuffled_codes,
                )

                correct_loss = F.mse_loss(
                    correct_pred,
                    velocity,
                ).item()

                shuffled_loss = F.mse_loss(
                    shuffled_pred,
                    velocity,
                ).item()

                prediction_change = F.mse_loss(
                    correct_pred,
                    shuffled_pred,
                ).item()

                print(
                    f"{level:6s} | "
                    f"correct={correct_loss:.4f} | "
                    f"shuffled={shuffled_loss:.4f} | "
                    f"change={prediction_change:.6f}"
                )

    print("\nDIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()
