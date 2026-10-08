
import torch

from adaptive_physio.encoders.register_encoder import RegisterEncoder
from adaptive_physio.tokenizers.fsq import FSQQuantizer
from adaptive_physio.training.granularity_targets import (
    GaussianGranularityTargets,
)
from adaptive_physio.decoders.conditional_flow_decoder import (
    ConditionalFlowDecoder,
)
from adaptive_physio.training.flow_matching import (
    flow_matching_loss,
)


torch.manual_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Use respiration-like sampling rate for smoke test
fs = 32
x = torch.randn(2, 1, 30 * fs, device=device)

encoder = RegisterEncoder(sample_rate=fs).to(device)
quantizer = FSQQuantizer().to(device)
decoder = ConditionalFlowDecoder().to(device)

# Construct targets
targets = GaussianGranularityTargets(sample_rate=fs)(x)

# Encode and quantize
codes, ids = quantizer(encoder(x))

# Compute losses for all three granularities
loss = 0

for level, n in [
    ("coarse", 16),
    ("mid", 64),
    ("fine", 128),
]:
    l = flow_matching_loss(
        decoder,
        targets[level],
        codes,
        n,
    )

    print(level, "loss:", round(l.item(), 5))
    loss = loss + l / 3

# Backpropagation
loss.backward()

assert encoder.patch_projection.weight.grad is not None
assert encoder.patch_projection.weight.grad.abs().sum() > 0

assert quantizer.to_code.weight.grad is not None
assert quantizer.to_code.weight.grad.abs().sum() > 0

assert decoder.cross_attention.in_proj_weight.grad is not None
assert decoder.cross_attention.in_proj_weight.grad.abs().sum() > 0

assert torch.isfinite(loss)

print("Loss:", round(loss.item(), 5))
print("Gradients: OK")
print("ALL TESTS PASSED")
