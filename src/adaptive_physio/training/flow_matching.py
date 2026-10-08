
import torch
import torch.nn.functional as F


def flow_matching_loss(
    decoder,
    target,
    codes,
    prefix_length,
):
    """
    Linear conditional flow matching.

    target: [B, 1, T]
    codes:  [B, 128, 6]
    """

    if target.ndim != 3:
        raise ValueError("target must be [B, C, T]")

    if not 1 <= prefix_length <= codes.shape[1]:
        raise ValueError("Invalid prefix length")

    # Gaussian source
    eps = torch.randn_like(target)

    # Sample flow time
    tau = torch.rand(
        target.shape[0],
        device=target.device,
        dtype=target.dtype,
    )

    t = tau[:, None, None]

    # Intermediate state
    x_tau = (1 - t) * eps + t * target

    # Ground-truth velocity
    velocity_target = target - eps

    # Predicted velocity conditioned on token prefix
    velocity_pred = decoder(
        x_tau,
        tau,
        codes[:, :prefix_length],
    )

    return F.mse_loss(
        velocity_pred,
        velocity_target,
    )
