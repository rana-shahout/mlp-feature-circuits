"""Population metrics, distractor associations, and observational truth tables."""

from itertools import product

import numpy as np

from .data import ANTECEDENTS, RULE_IDS
from .evaluation import prediction_metrics


def diagnostic_metrics(prediction, g, q, threshold):
    result = prediction_metrics(prediction, g, q, threshold)
    if not len(g):
        return result
    scores = prediction.astype(np.float64) @ q.astype(np.float64)
    for j, row in enumerate(result["per_rule"]):
        row["false_positive_rate"] = row["fp"] / row["n_negative"] if row["n_negative"] else None
        row["false_negative_rate"] = row["fn"] / row["n_positive"] if row["n_positive"] else None
        row["mean_absolute_error"] = float(np.abs(scores[:, j] - g[:, j]).mean())
        row["signed_bias"] = float((scores[:, j] - g[:, j]).mean())
    return result


def grouped_rows(a, scores, g, threshold, codes, group_name, label_fn=str):
    """Vectorized count/identity groups; no quadratic scan over input masks."""
    values, inverse = np.unique(codes, return_inverse=True)
    n = np.bincount(inverse, minlength=len(values))
    predicted, truth = scores >= threshold, g.astype(bool)
    for j, rule in enumerate(RULE_IDS):
        t, p = truth[:, j], predicted[:, j]
        fields = {"positive": t, "tp": t & p, "tn": ~t & ~p, "fp": ~t & p, "fn": t & ~p,
                  "squared_error": (scores[:, j] - g[:, j])**2,
                  "absolute_error": np.abs(scores[:, j] - g[:, j])}
        counts = {key: np.bincount(inverse, weights=value, minlength=len(values)) for key, value in fields.items()}
        for b, value in enumerate(values):
            pos, neg = int(counts["positive"][b]), int(n[b] - counts["positive"][b])
            tp, tn, fp, fn = [int(counts[key][b]) for key in ("tp", "tn", "fp", "fn")]
            yield {"group": group_name, "value": label_fn(int(value)), "rule": rule,
                   "n": int(n[b]), "n_positive": pos, "n_negative": neg, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                   "precision": tp / (tp + fp) if tp + fp else None,
                   "recall": tp / pos if pos else None,
                   "false_positive_rate": fp / neg if neg else None,
                   "false_negative_rate": fn / pos if pos else None,
                   "error_rate": (fp + fn) / int(n[b]),
                   "semantic_mse": float(counts["squared_error"][b] / n[b]),
                   "mean_absolute_error": float(counts["absolute_error"][b] / n[b])}


def distractor_groups(a):
    nuisance = a[:, 18:].astype(np.int64)
    count = nuisance.sum(1)
    yield "total_active_count", a.sum(1).astype(int), str
    yield "distractor_count", count, str
    mask = nuisance @ (1 << np.arange(14))
    yield "distractor_identity_set", mask, lambda key: " ".join(str(18+i) for i in range(14) if key & (1 << i)) or "none"
    for i in range(18, 32):
        yield f"distractor_{i}_presence", a[:, i], str
        yield f"distractor_{i}_presence_given_count", 2*count+a[:, i], lambda key: f"count={key//2};present={key%2}"


def observational_tables(a, scores, g, threshold):
    for j, ids in enumerate(ANTECEDENTS):
        for bits in product((0, 1), repeat=len(ids)):
            mask = np.all(a[:, ids] == bits, axis=1)
            predicted, truth = scores[mask, j] >= threshold, g[mask, j].astype(bool)
            yield {"rule": RULE_IDS[j], "antecedents": ",".join(map(str, ids)), "pattern": "".join(map(str, bits)),
                   "n": int(mask.sum()), "positive_count": int(truth.sum()),
                   "mean_score": float(scores[mask, j].mean()) if mask.any() else None,
                   "error_rate": float((predicted != truth).mean()) if mask.any() else None,
                   "note": "Observed rows; other semantic inputs and distractors are not held fixed."}


def make_contexts(settings, seed):
    rng = np.random.default_rng(seed)
    distractor_order = rng.permutation(np.arange(18, 32))
    a = np.zeros((len(settings), 32), dtype=np.uint8)
    names = []
    for i, context in enumerate(settings):
        semantic = context["semantic_active"]
        if any(not 0 <= bit < 18 for bit in semantic) or not 0 <= context["distractor_count"] <= 14:
            raise ValueError("Invalid fixed context")
        a[i, semantic] = 1
        # Same nuisance identities at a given count across semantic gate states;
        # count-2/4/6 contexts are nested prefixes of one fixed permutation.
        a[i, distractor_order[:context["distractor_count"]]] = 1
        names.append(context["name"])
    if len(set(names)) != len(names):
        raise ValueError("Context names must be unique")
    return a, names
