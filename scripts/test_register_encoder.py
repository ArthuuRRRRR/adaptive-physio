
import torch
from adaptive_physio.encoders.register_encoder import RegisterEncoder

torch.manual_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

modalities = {
    "EEG": 128,
    "ECG": 128,
    "RESP": 32,
}

print("Device:", device)

for name, fs in modalities.items():

    model = RegisterEncoder(
        sample_rate=fs,
        patch_seconds=0.25,
        embedding_dim=128,
        num_registers=128,
    ).to(device)

    # Synthetic 30-second signal
    x = torch.randn(2, 1, 30 * fs, device=device)

    tokens = model(x)

    assert tokens.shape == (2, 128, 128)

    # Check gradient flow
    loss = tokens.square().mean()
    loss.backward()

    assert model.patch_projection.weight.grad is not None

    print(f"{name}: {tuple(x.shape)} -> {tuple(tokens.shape)}")

print("ALL TESTS PASSED")

