"""Exact Boolean finite differences, including original-neuron signed sums.

No derivative approximation, gate linearization, fitting, thresholding of
contributions, or truncation. 'Exact' means the finite algebra, to FP tolerance.
"""

from itertools import product

import torch

from .coordinates import semantic_forward


def corners_and_signs(contexts, indices):
    """Return (context, corner, feature) and (-1)^(order-sum(bits))."""
    ids = tuple(indices)
    if len(ids) not in (2, 3) or len(set(ids)) != len(ids):
        raise ValueError("Require two or three distinct intervention indices")
    if contexts.ndim != 2 or min(ids) < 0 or max(ids) >= contexts.shape[1]:
        raise ValueError("Invalid contexts or input indices")
    if not torch.all((contexts == 0) | (contexts == 1)):
        raise ValueError("Contexts must be binary")
    patterns = list(product((0, 1), repeat=len(ids)))
    bits = torch.tensor(patterns, dtype=contexts.dtype, device=contexts.device)
    a = contexts[:, None, :].expand(-1, len(bits), -1).clone()
    a[:, :, list(ids)] = bits[None, :, :]
    signs = torch.tensor([(-1) ** (len(ids) - sum(row)) for row in patterns], dtype=contexts.dtype, device=contexts.device)
    return a, signs


@torch.no_grad()
def finite_interaction(contexts, indices, coordinates):
    """Every context/output/neuron contribution, preserving complete signs.

    hidden_delta: (C,H), contributions: (C,G,H), direct/value: (C,G).
    Zero-order output bias cancels exactly in a positive-order difference.
    """
    a, signs = corners_and_signs(contexts, indices)
    return decompose_corners(a, signs, coordinates)


@torch.no_grad()
def decompose_corners(a, signs, coordinates):
    """Also accepts batches: a (...,corners,features), signs (corners,)."""
    signed_shape = [1] * (a.ndim - 2) + [len(signs), 1]
    hidden = torch.relu(a @ coordinates["C1"].T + coordinates["b1"])
    hidden_delta = (hidden * signs.reshape(signed_shape)).sum(-2)
    contributions = hidden_delta[..., None, :] * coordinates["D2_eff"]
    value = contributions.sum(-1)
    direct = (semantic_forward(a, coordinates) * signs.reshape(signed_shape)).sum(-2)
    positive = contributions.clamp_min(0).sum(-1)
    negative = contributions.clamp_max(0).sum(-1)
    absolute = positive - negative
    cancellation = torch.where(absolute > 0, 1 - value.abs() / absolute.clamp_min(torch.finfo(value.dtype).tiny), 0)
    return {"corners": a, "signs": signs, "hidden_delta": hidden_delta,
            "contributions": contributions, "value": value, "direct": direct,
            "positive_sum": positive, "negative_sum": negative,
            "absolute_sum": absolute, "cancellation_fraction": cancellation}
