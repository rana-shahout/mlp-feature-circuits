"""Exact semantic re-expression of ORIGINAL neurons; no fitted probes."""

import torch


@torch.no_grad()
def original_coordinates(model, p, q):
    """C1 (H,m), D2_eff (g,H), b1 (H,), semantic_bias (g,).

    D2_eff is Q.T @ W2 (the user's D2), following SPEC's naming caveat.
    The caller chooses device/dtype; diagnostics use float64 for cancellation.
    """
    w1, w2 = model.W1.weight, model.W2.weight
    if p.ndim != 2 or q.ndim != 2 or p.shape[0] != w1.shape[1] or q.shape[0] != w2.shape[0]:
        raise ValueError("Incompatible P/Q/model dimensions")
    if p.device != w1.device or q.device != w1.device or p.dtype != w1.dtype or q.dtype != w1.dtype:
        raise ValueError("P/Q must share the model's device and dtype")
    return {"C1": w1 @ p, "D2_eff": q.T @ w2,
            "b1": model.W1.bias.detach(), "semantic_bias": q.T @ model.W2.bias}


def semantic_forward(a, coordinates):
    return torch.relu(a @ coordinates["C1"].T + coordinates["b1"]) @ coordinates["D2_eff"].T + coordinates["semantic_bias"]


@torch.no_grad()
def transcoder_coordinates(model, p, q):
    """K=EP (K,m), A=Q.T B (G,K); aliases reuse exact gate algebra only."""
    k = model.encoder.weight @ p
    a = q.T @ model.decoder.weight
    return {"C1": k, "D2_eff": a, "b1": model.encoder.bias.detach(),
            "semantic_bias": q.T @ model.decoder.bias}
