
import torch

from adaptive_physio.encoders.register_encoder import RegisterEncoder
from adaptive_physio.tokenizers.fsq import FSQQuantizer


torch.manual_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Step 1: Encode a synthetic 30-second EEG signal.
encoder = RegisterEncoder(
    sample_rate=128,
    embedding_dim=128,
    num_registers=128,
).to(device)

x = torch.randn(2, 1, 3840, device=device)

registers = encoder(x)

# Step 2: Quantize continuous embeddings.
quantizer = FSQQuantizer(
    input_dim=128,
    levels=(4, 4, 4, 4, 4, 4),
).to(device)

codes, indices = quantizer(registers)

print("Registers:", registers.shape)
print("Quantized codes:", codes.shape)
print("Discrete IDs:", indices.shape)
print("Vocabulary size:", quantizer.codebook_size)

# Step 3: Check valid token IDs.
assert indices.shape == (2, 128)
assert indices.min().item() >= 0
assert indices.max().item() < 4096

# Step 4: Verify exact code recovery from IDs.
recovered = quantizer.indices_to_codes(indices)

assert torch.allclose(codes.detach(), recovered)

# Step 5: Verify hierarchical prefixes.
for name, n in {
    "coarse": 16,
    "mid": 64,
    "fine": 128,
}.items():
    prefix = indices[:, :n]
    print(f"{name}: {prefix.shape}")
    assert prefix.shape == (2, n)

# Step 6: Verify gradients pass through quantization.
loss = codes.square().mean()
loss.backward()

assert quantizer.to_code.weight.grad is not None
assert encoder.patch_projection.weight.grad is not None

print("Gradient flow: OK")
print("ALL TESTS PASSED")

