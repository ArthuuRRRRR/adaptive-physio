import torch
from types import SimpleNamespace

from train_temporal_agent_v4 import load_agent
from train_temporal_agent_physio_v5 import (
    setup, data_loader, normalize, get_codes,
)


def visible_mask(states):
    b = states.shape[0]
    mask = torch.zeros(
        b, 128, dtype=torch.bool,
        device=states.device
    )

    # Coarse tokens are always available.
    mask[:, :16] = True

    for r in range(16):
        mid = states[:, r] >= 1
        fine = states[:, r] >= 2

        mask[mid, 16 + 3*r:16 + 3*(r+1)] = True
        mask[fine, 64 + 4*r:64 + 4*(r+1)] = True

    return mask


@torch.no_grad()
def check(agent, codes, states, label):
    visible = visible_mask(states)
    shuffled = codes.roll(1, dims=0)

    # Only alter information not yet revealed.
    hidden_changed = torch.where(
        visible.unsqueeze(-1),
        codes,
        shuffled,
    )

    # Control experiment: alter visible information.
    visible_changed = torch.where(
        visible.unsqueeze(-1),
        shuffled,
        codes,
    )

    original = agent(codes, states).float()
    hidden_output = agent(
        hidden_changed, states
    ).float()
    visible_output = agent(
        visible_changed, states
    ).float()

    valid = torch.stack(
        (states == 0, states == 1),
        dim=-1,
    )

    hidden_diff = (
        original - hidden_output
    ).abs()[valid]

    visible_diff = (
        original - visible_output
    ).abs()[valid]

    print(f"\n{label}")
    print(
        "Hidden-token change:",
        f"max={hidden_diff.max().item():.8f}",
        f"mean={hidden_diff.mean().item():.8f}"
    )
    print(
        "Visible-token change:",
        f"max={visible_diff.max().item():.8f}",
        f"mean={visible_diff.mean().item():.8f}"
    )

    if hidden_diff.max().item() > 1e-6:
        print("RESULT: HIDDEN INFORMATION AFFECTS DECISIONS")
    else:
        print("RESULT: NO HIDDEN-TOKEN DEPENDENCE DETECTED")


def main():
    cfg = SimpleNamespace(
        seed=42,
        backbone=(
            "outputs/checkpoints/"
            "timetok_v3_eeg_ae/frozen_for_agent.pt"
        ),
        singlem="third_party/SingLEM",
        physio_weight=5.0,
    )

    device, backbone, ck, teacher = setup(cfg)

    loader, _ = data_loader(
        "val", "eeg", 8, 8, 0, 42
    )

    raw = next(iter(loader))
    x = normalize(raw, ck["normalization"], device)

    with torch.no_grad():
        codes = get_codes(backbone, x)

    b = len(x)

    states_coarse = torch.zeros(
        b, 16,
        dtype=torch.long,
        device=device,
    )

    states_mixed = torch.tensor(
        [
            [(r + j) % 3 for r in range(16)]
            for j in range(b)
        ],
        dtype=torch.long,
        device=device,
    )

    checkpoints = {
        "V4": (
            "outputs/checkpoints/"
            "timetok_v3_agent_cost_eeg/best.pt"
        ),
        "V5": (
            "outputs/checkpoints/"
            "timetok_v3_agent_physio_beta5/best.pt"
        ),
        "V6": (
            "outputs/checkpoints/"
            "timetok_v3_agent_physio_local_v6/best.pt"
        ),
    }

    for name, path in checkpoints.items():
        agent, _ = load_agent(path, device)
        agent.eval()

        print(f"\n========== {name} ==========")

        check(
            agent, codes,
            states_coarse,
            "All coarse"
        )

        check(
            agent, codes,
            states_mixed,
            "Mixed coarse/mid/fine"
        )


if __name__ == "__main__":
    main()
