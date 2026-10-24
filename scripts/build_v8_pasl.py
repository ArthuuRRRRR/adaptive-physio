"""Export V8_80 using existing validated PaSL pipeline."""

from pathlib import Path

import torch

from adaptive_physio.models.belief_agent_v8 import (
    choose_levels,
)

from train_belief_agent_v8 import load

import build_adaptive_pasl as existing

from evaluate_timetok_v3_adaptive import (
    decode_mixed,
)


original_reconstruct = existing.reconstruct

MODEL = None


@torch.no_grad()
def reconstruct_v8(
    name,
    tokenizer,
    agents,
    codes,
    sample_ids,
):
    global MODEL

    if name != "V8_80":
        return original_reconstruct(
            name,
            tokenizer,
            agents,
            codes,
            sample_ids,
        )

    if MODEL is None:

        path = Path(
            "outputs/checkpoints/"
            "timetok_v3_agent_belief_v8/best.pt"
        )

        MODEL, ck = load(
            path, codes.device
        )

        if ck["phase"] != "ppo":
            raise RuntimeError(
                "Expected PPO-trained V8 checkpoint"
            )

        backbone_ck = torch.load(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt",
            map_location="cpu",
            weights_only=False,
        )

        if (
            ck["backbone_step"] != backbone_ck["step"]
            or ck["normalization"]
            != backbone_ck["normalization"]
        ):
            raise RuntimeError(
                "Tokenizer V3 / V8 checkpoint mismatch"
            )

        MODEL.eval()

    levels, used = choose_levels(
        MODEL,
        codes.float(),
        budget=80,
    )

    reconstruction = decode_mixed(
        tokenizer.reconstructor,
        codes,
        levels,
    ).float()

    return reconstruction, used


def main():

    # Reuse all existing alignment and audit checks.
    existing.reconstruct = reconstruct_v8

    existing.main()


if __name__ == "__main__":
    main()
