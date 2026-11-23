"""Teacher fidelity and ground-truth behavior are deliberately separate."""

import numpy as np
import torch

from .data import RULE_IDS
from .evaluation import prediction_metrics


def contribution_statistics(contributions):
    """Absolute-mass participation ratio; no signed cancellation hidden."""
    c = np.asarray(contributions, dtype=np.float64)
    absolute = np.abs(c)
    mass = absolute.sum(-1)
    squares = np.square(c).sum(-1)
    safe = np.where(mass > 0, mass, 1)
    effective = np.divide(mass**2, squares, out=np.zeros_like(mass), where=squares > 0)
    ordered = np.sort(absolute, axis=-1)[..., ::-1]
    return {"value": c.sum(-1), "positive_sum": np.maximum(c, 0).sum(-1),
            "negative_sum": np.minimum(c, 0).sum(-1), "absolute_mass": mass,
            "effective_units": effective, "effective_fraction": effective/c.shape[-1],
            "top1_mass_fraction": ordered[..., :1].sum(-1)/safe,
            "top5_mass_fraction": ordered[..., :5].sum(-1)/safe,
            "top10_mass_fraction": ordered[..., :10].sum(-1)/safe,
            "cancellation_fraction": np.where(mass > 0, 1-np.abs(c.sum(-1))/safe, 0)}


def fidelity_metrics(student, teacher, g, q, threshold):
    s, t, q = [np.asarray(x, dtype=np.float64) for x in (student, teacher, q)]
    ss, ts = s @ q, t @ q
    sp, tp, truth = ss >= threshold, ts >= threshold, g.astype(bool)
    teacher_row_correct = np.all(tp == truth, axis=1)
    result = {"n": len(g), "full_output_mse": float(np.mean((s-t)**2)),
              "semantic_mse": float(np.mean((ss-ts)**2)),
              "exact_vector_agreement": float(np.all(sp == tp, axis=1).mean()),
              "mean_feature_agreement": float((sp == tp).mean()), "per_output": []}
    for name, mask in [("teacher_success_rows", teacher_row_correct), ("teacher_failure_rows", ~teacher_row_correct)]:
        result[name] = {"n": int(mask.sum()),
                       "full_output_mse": float(((s[mask]-t[mask])**2).mean()) if mask.any() else None,
                       "exact_vector_agreement": float(np.all(sp[mask] == tp[mask], axis=1).mean()) if mask.any() else None}
    for j, name in enumerate(RULE_IDS):
        wrong = tp[:, j] != truth[:, j]
        result["per_output"].append({"rule": name, "semantic_mse": float(((ss[:, j]-ts[:, j])**2).mean()),
            "agreement": float((sp[:, j] == tp[:, j]).mean()), "teacher_error_count": int(wrong.sum()),
            "teacher_errors_preserved_fraction": float((sp[wrong,j] == tp[wrong,j]).mean()) if wrong.any() else None,
            "teacher_errors_corrected_fraction": float((sp[wrong,j] == truth[wrong,j]).mean()) if wrong.any() else None})
    return {"fidelity": result, "teacher_task": prediction_metrics(t, g, q, threshold),
            "student_task": prediction_metrics(s, g, q, threshold)}


def activation_statistics(z):
    z = np.asarray(z)
    active = z > 0  # Strict ReLU L0, not an arbitrary epsilon cutoff.
    return {"mean_l0": float(active.sum(1).mean()), "mean_l0_fraction": float(active.mean()),
            "mean_l1_sum": float(z.sum(1).mean()), "dead_units": int((~active.any(0)).sum()),
            "prevalence": active.mean(0).tolist(), "mean": z.mean(0).tolist(),
            "std": z.std(0).tolist(), "quantile_levels": [0, .5, .9, .99, 1],
            "activation_quantiles": np.quantile(z, [0,.5,.9,.99,1], axis=0).tolist()}


@torch.no_grad()
def student_predict(model, a, p, device, batch_size):
    p = torch.as_tensor(p, dtype=next(model.parameters()).dtype, device=device)
    outputs, latents = [], []
    for start in range(0, len(a), batch_size):
        inputs = torch.as_tensor(a[start:start+batch_size], dtype=p.dtype, device=device) @ p.T
        z = model.latents(inputs)
        y = model.decoder(z)
        if not torch.isfinite(y).all() or not torch.isfinite(z).all():
            raise RuntimeError("Non-finite transcoder prediction")
        outputs.append(y.cpu().numpy()); latents.append(z.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(latents)
