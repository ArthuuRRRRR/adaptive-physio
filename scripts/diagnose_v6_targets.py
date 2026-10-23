import numpy as np
import torch
from types import SimpleNamespace

from train_temporal_agent_v4 import load_agent
from train_temporal_agent_physio_v5 import (
    setup, data_loader, normalize, get_codes,
)
from train_temporal_agent_physio_v6 import (
    raw_gains, region_weights,
)

device_name = "cuda" if torch.cuda.is_available() else "cpu"

cfg = SimpleNamespace(
    seed=42,
    backbone="outputs/checkpoints/timetok_v3_eeg_ae/frozen_for_agent.pt",
    singlem="third_party/SingLEM",
    physio_weight=5.0,
)

device, backbone, bc, teacher = setup(cfg)

v5path = "outputs/checkpoints/timetok_v3_agent_physio_beta5/best.pt"
v6path = "outputs/checkpoints/timetok_v3_agent_physio_local_v6/best.pt"

v5, _ = load_agent(v5path, device)
v6, _ = load_agent(v6path, device)
v5.eval()
v6.eval()

checkpoint = torch.load(
    v6path, map_location="cpu", weights_only=False
)
cal = checkpoint["calibration"]

weights = region_weights(device, sigma=1.25)

loader, _ = data_loader(
    "val", "eeg", 256, 8, 0, 42
)

records = {
    k: [] for k in (
        "local", "global", "wave",
        "target", "pred_v5", "pred_v6",
        "distinct"
    )
}

with torch.inference_mode():
    for step, raw in enumerate(loader):
        x = normalize(raw, bc["normalization"], device)
        codes = get_codes(backbone, x)

        # Reset RNG: same states and actions in both calls.
        seed = 7000 + step

        torch.manual_seed(seed)
        loc = raw_gains(
            backbone, teacher, codes, x,
            bc["normalization"], weights, 1.0
        )

        torch.manual_seed(seed)
        glob = raw_gains(
            backbone, teacher, codes, x,
            bc["normalization"], weights, 0.0
        )

        states = loc[1]
        actions = loc[2]
        kinds = loc[3]

        assert torch.equal(states, glob[1])
        assert torch.equal(actions, glob[2])
        assert torch.equal(kinds, glob[3])

        local_gain = loc[4]
        global_gain = glob[4]
        wave_gain = loc[5]

        target = 0.20 * (
            0.85 * (
                0.80 * local_gain
                + 0.20 * global_gain
            ) / cal["phys_scale"]
            + 0.15 * wave_gain / cal["wave_scale"]
        )

        rows = torch.arange(len(x), device=device)[:, None]

        pred5 = v5(codes, states)[rows, actions, kinds]
        pred6 = v6(codes, states)[rows, actions, kinds]

        for key, value in (
            ("local", local_gain),
            ("global", global_gain),
            ("wave", wave_gain),
            ("target", target),
            ("pred_v5", pred5),
            ("pred_v6", pred6),
            ("distinct", actions[:, 0] != actions[:, 1]),
        ):
            records[key].append(value.cpu().numpy())

data = {
    k: np.concatenate(v, axis=0)
    for k, v in records.items()
}

mask = data["distinct"].astype(bool)

def agreement(a, b):
    da = data[a][:, 0] - data[a][:, 1]
    db = data[b][:, 0] - data[b][:, 1]

    valid = mask & (np.abs(da) > 1e-10) & (np.abs(db) > 1e-10)

    return (
        np.mean(np.sign(da[valid]) == np.sign(db[valid])),
        int(valid.sum())
    )

def report(label, a, b):
    value, n = agreement(a, b)
    print(f"{label}: {value:.1%} (n={n})")

print("\n=== V6 TARGET DIAGNOSTIC ===")
print("Distinct action pairs:", mask.sum())

r = np.corrcoef(
    data["local"].reshape(-1),
    data["global"].reshape(-1),
)[0, 1]

print(f"Local/global gain correlation: {r:.4f}")

report(
    "Local vs global preferences",
    "local", "global"
)
report(
    "V6 vs its training target",
    "pred_v6", "target"
)
report(
    "V6 vs local SingLEM",
    "pred_v6", "local"
)
report(
    "V6 vs global SingLEM",
    "pred_v6", "global"
)
report(
    "V5 vs global SingLEM",
    "pred_v5", "global"
)

for name in ("local", "global", "wave"):
    values = data[name].reshape(-1)
    print(
        f"{name}: mean={values.mean():.8f} "
        f"std={values.std():.8f} "
        f"negative={(values < 0).mean():.1%}"
    )
