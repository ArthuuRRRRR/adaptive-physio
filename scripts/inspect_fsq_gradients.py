import torch

from adaptive_physio.data.preprocessed_dataset import PreprocessedPSGDataset
from adaptive_physio.encoders.register_encoder import RegisterEncoder
from adaptive_physio.tokenizers.fsq import FSQQuantizer
from adaptive_physio.decoders.conditional_flow_decoder import ConditionalFlowDecoder
from adaptive_physio.training.flow_matching import flow_matching_loss

torch.manual_seed(42)
device = "cuda:0"

ckpt = torch.load(
    "outputs/checkpoints/overfit_eeg_600steps.pt",
    map_location=device,
    weights_only=True,
)

ds = PreprocessedPSGDataset(split="train", max_records=1)

x = torch.stack([ds[i]["eeg"] for i in range(16)]).to(device)
x = (
    x - ckpt["normalization"]["mean"]
) / ckpt["normalization"]["std"]

encoder = RegisterEncoder(
    sample_rate=100, patch_seconds=0.2
).to(device)
fsq = FSQQuantizer().to(device)
decoder = ConditionalFlowDecoder(
    downsample=5
).to(device)

encoder.load_state_dict(ckpt["encoder"])
fsq.load_state_dict(ckpt["quantizer"])
decoder.load_state_dict(ckpt["decoder"])

encoder.eval()
fsq.eval()
decoder.eval()

registers = encoder(x)
z = fsq.to_code(registers)
codes, ids = fsq(registers)

print("\n=== FSQ DISTRIBUTION ===")
for d in range(6):
    values, counts = torch.unique(
        codes[..., d].detach(),
        return_counts=True,
    )
    print(
        f"Dim {d}: "
        f"z_min={z[..., d].min().item():.3f} "
        f"z_max={z[..., d].max().item():.3f} "
        f"codes={list(zip(values.tolist(), counts.tolist()))}"
    )

print("\n=== GRADIENT TEST ===")
loss = flow_matching_loss(
    decoder=decoder,
    target=x,
    codes=codes,
    prefix_length=128,
)
loss.backward()

for name, parameter in [
    ("FSQ projection", fsq.to_code.weight),
    ("Encoder registers", encoder.register_tokens),
    ("Decoder attention", decoder.cross_attention.in_proj_weight),
]:
    grad = parameter.grad
    print(
        f"{name}: "
        f"{grad.norm().item():.8f}" if grad is not None
        else f"{name}: NO GRADIENT"
    )

print(f"Flow loss: {loss.item():.5f}")
