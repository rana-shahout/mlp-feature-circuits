"""Known semantic inputs/targets; the original model receives only x = P a.

P: (d_in,32), Q: (d_out,8). Stored row batches use X = a @ P.T,
Y = g @ Q.T. P is overcomplete with unit columns, never orthogonalized.
"""

from itertools import combinations, product
import warnings

import numpy as np


RULE_IDS = ("and_01", "or_23", "inhibition_45", "xor_67", "and_8910",
            "overlapping_dnf", "mixed_inhibition", "mixed_xor")
ANTECEDENTS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9, 10),
               (0, 11, 12, 13), (2, 14, 15), (4, 16, 17))
MODES = ("background", "target_positive", "hard_negative")


def targets(a):
    """Single source of truth for SPEC §4, returning (...,8) float32."""
    a = np.asarray(a)
    if a.ndim < 1 or a.shape[-1] != 32 or not np.all((a == 0) | (a == 1)):
        raise ValueError("Expected binary semantic inputs of shape (...,32)")
    b = a.astype(bool)
    return np.stack([
        b[..., 0] & b[..., 1], b[..., 2] | b[..., 3],
        b[..., 4] & ~b[..., 5], b[..., 6] ^ b[..., 7],
        b[..., 8] & b[..., 9] & b[..., 10],
        (b[..., 0] & b[..., 11]) | (b[..., 12] & b[..., 13]),
        (b[..., 2] | b[..., 14]) & ~b[..., 15],
        (b[..., 4] & b[..., 16]) ^ b[..., 17],
    ], axis=-1).astype(np.float32)


def embeddings(d_in, d_out, input_seed, output_seed):
    if not 0 < d_in < 32 or d_out < 8:
        raise ValueError("Require an overcomplete P and at least 8 output dimensions")
    p = np.random.default_rng(input_seed).normal(size=(d_in, 32))
    p /= np.linalg.norm(p, axis=0, keepdims=True)
    q, r = np.linalg.qr(np.random.default_rng(output_seed).normal(size=(d_out, 8)), mode="reduced")
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)[None, :]
    p, q = p.astype(np.float32), q.astype(np.float32)
    np.testing.assert_allclose(np.linalg.norm(p, axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-6)
    return p, q


def vector_keys(a):
    """Injective key for binary vectors of length 32 (not a hash)."""
    return np.asarray(a, dtype=np.uint64) @ (np.uint64(1) << np.arange(32, dtype=np.uint64))


def pattern_pools():
    pools = []
    for j, ids in enumerate(ANTECEDENTS):
        patterns = np.asarray(list(product((0, 1), repeat=len(ids))), dtype=np.uint8)
        a = np.zeros((len(patterns), 32), dtype=np.uint8)
        a[:, ids] = patterns
        label = targets(a)[:, j].astype(bool)
        # Hard negative = Hamming distance one from some satisfying local pattern.
        # XOR negatives include both 00 and 11; inhibition includes inhibitor-on.
        distance = np.abs(patterns[:, None, :].astype(int) - patterns[label][None, :, :]).sum(-1)
        near = (~label) & (distance.min(axis=1) == 1)
        pools.append((patterns[label], patterns[near]))
    return pools


def sample_dataset(config, seed):
    rng = np.random.default_rng(seed)
    pools = pattern_pools()
    rows, modes, requested = [], [], []
    seen = set()
    attempted_modes = np.zeros(3, dtype=int)
    rejected_modes = np.zeros(3, dtype=int)
    counts = np.asarray(config["active_counts"])
    probabilities = np.asarray(config["active_probabilities"], dtype=float)
    attempts = 0
    while len(rows) < config["n_samples"] and attempts < config["max_attempts"]:
        attempts += 1
        mode = int(rng.choice(3, p=config["mode_probabilities"]))
        attempted_modes[mode] += 1
        row = np.zeros(32, dtype=np.uint8)
        rule = -1
        if mode == 0:
            k = int(rng.choice(counts, p=probabilities))
            row[rng.choice(32, size=k, replace=False)] = 1
        else:
            rule = int(rng.integers(8))
            ids = ANTECEDENTS[rule]
            pool = pools[rule][0 if mode == 1 else 1]
            row[list(ids)] = pool[int(rng.integers(len(pool)))]
            active = int(row.sum())
            valid = (counts >= active) & (counts - active <= 32 - len(ids))
            if not valid.any():
                rejected_modes[mode] += 1
                continue
            k = int(rng.choice(counts[valid], p=probabilities[valid] / probabilities[valid].sum()))
            available = np.setdiff1d(np.arange(32), ids)
            row[rng.choice(available, size=k-active, replace=False)] = 1
            if targets(row)[rule] != (mode == 1):
                raise AssertionError("Constructed label does not match requested sampling mode")
        key = int(vector_keys(row))
        if key in seen:
            rejected_modes[mode] += 1
            continue
        seen.add(key)
        rows.append(row)
        modes.append(mode)
        requested.append(rule)
    if len(rows) < config["min_samples"]:
        raise RuntimeError(f"Only {len(rows)} unique samples after {attempts} attempts; below min_samples")
    if len(rows) < config["n_samples"]:
        warnings.warn(f"Requested {config['n_samples']} unique inputs; obtained {len(rows)} in {attempts} attempts")
    # Shuffle away construction order; split uses an independent seed.
    order = rng.permutation(len(rows))
    return (np.asarray(rows)[order], np.asarray(modes, dtype=np.int8)[order],
            np.asarray(requested, dtype=np.int8)[order],
            {"requested_count": config["n_samples"], "actual_count": len(rows),
             "attempts": attempts, "attempted_modes": attempted_modes.tolist(),
             "rejected_modes": rejected_modes.tolist(),
             "note": "Uniqueness/rejection changes the accepted mixture; inspect realized counts."})


def split_indices(n, fractions, seed):
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) != 3 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive split fractions must sum to one")
    sizes = np.floor(n * fractions).astype(int)
    for i in np.argsort(-(n * fractions - sizes), kind="stable")[:n - sizes.sum()]:
        sizes[i] += 1
    if np.any(sizes == 0):
        raise ValueError("Dataset too small for nonempty train/validation/test")
    order = np.random.default_rng(seed).permutation(n)
    return dict(zip(("train", "validation", "test"), np.split(order, np.cumsum(sizes)[:2])))


def contrast_set(config, seed):
    """Complete antecedent tables with fixed non-antecedent backgrounds per group.

    Background density uses the same k distribution, removing antecedents before
    varying them. Complete tables can exceed the training density; record this.
    Rows are unique; memberships retain every (rule,context,pattern) occurrence.
    """
    rng = np.random.default_rng(seed)
    rows, lookup, memberships = [], {}, []
    for rule, ids in enumerate(ANTECEDENTS):
        contexts = set()
        attempts = 0
        while len(contexts) < config["contrast_contexts_per_rule"]:
            attempts += 1
            if attempts > 10000 * config["contrast_contexts_per_rule"]:
                raise RuntimeError("Unable to generate requested distinct contrast contexts")
            base = np.zeros(32, dtype=np.uint8)
            k = int(rng.choice(config["active_counts"], p=config["active_probabilities"]))
            base[rng.choice(32, k, replace=False)] = 1
            base[list(ids)] = 0
            key = int(vector_keys(base))
            if key in contexts:
                continue
            context = len(contexts)
            contexts.add(key)
            for pattern, bits in enumerate(product((0, 1), repeat=len(ids))):
                row = base.copy()
                row[list(ids)] = bits
                row_key = int(vector_keys(row))
                if row_key not in lookup:
                    lookup[row_key] = len(rows)
                    rows.append(row)
                memberships.append((lookup[row_key], rule, context, pattern))
    return np.asarray(rows, dtype=np.uint8), np.asarray(memberships, dtype=np.int64)


def overlap_partitions(contrast, a, split):
    result = np.full(len(contrast), -1, dtype=np.int8)
    contrast_keys = vector_keys(contrast)
    for label, name in enumerate(("train", "validation", "test")):
        result[np.isin(contrast_keys, vector_keys(a[split[name]]))] = label
    return result


def audit_samples(a, mode=None):
    if not len(a):
        return {"n": 0}
    g = targets(a)
    report = {"n": len(a), "unique_count": len(np.unique(vector_keys(a))),
              "prevalence": g.mean(0).tolist(),
              "active_count_histogram": {str(int(k)): int((a.sum(1) == k).sum()) for k in np.unique(a.sum(1))}}
    report["antecedent_pair_counts_00_01_10_11"] = {
        f"{RULE_IDS[r]}:{i},{j}": np.bincount(2*a[:, i]+a[:, j], minlength=4).tolist()
        for r, ids in enumerate(ANTECEDENTS) for i, j in combinations(ids, 2)}
    report["prevalence_by_density"] = {
        str(int(k)): g[a.sum(1) == k].mean(0).tolist() for k in np.unique(a.sum(1))}
    if mode is not None:
        report["by_mode"] = {name: {"n": int((mode == i).sum()),
                                  "prevalence": g[mode == i].mean(0).tolist() if (mode == i).any() else None}
                             for i, name in enumerate(MODES)}
    # Phi/Pearson correlations flag marginal nuisance shortcuts, not independence.
    nuisance = a[:, 18:].astype(float)
    centered_n, centered_g = nuisance - nuisance.mean(0), g - g.mean(0)
    denom = np.linalg.norm(centered_n, axis=0)[:, None] * np.linalg.norm(centered_g, axis=0)[None, :]
    corr = np.divide(centered_n.T @ centered_g, denom, out=np.zeros((14, 8)), where=denom > 0)
    report["nuisance_phi"] = [[float(corr[i, j]) if denom[i, j] > 0 else None for j in range(8)] for i in range(14)]
    report["nuisance_prevalence_by_label"] = {
        RULE_IDS[j]: {str(label): nuisance[g[:, j] == label].mean(0).tolist() if (g[:, j] == label).any() else None
                      for label in (0, 1)} for j in range(8)}
    return report


def geometry_diagnostics(p, q, a, probe_samples):
    gram = p.T @ p
    n = min(probe_samples, len(a))
    probe = a[:n].astype(np.float64)
    x = probe @ p.astype(np.float64).T
    distances = np.full(n, np.inf)
    neighbor = np.full(n, -1, dtype=int)
    for start in range(0, n, 128):
        squared = ((x[start:start+128, None, :] - x[None, :, :]) ** 2).sum(-1)
        squared[np.arange(len(squared)), np.arange(start, start + len(squared))] = np.inf
        neighbor[start:start+len(squared)] = squared.argmin(1)
        distances[start:start+len(squared)] = np.sqrt(squared.min(1))
    reconstruction = x @ np.linalg.pinv(p.astype(np.float64)).T
    return {"P_shape": list(p.shape), "Q_shape": list(q.shape),
            "P_singular_values": np.linalg.svd(p, compute_uv=False).tolist(),
            "P_column_norms": np.linalg.norm(p, axis=0).tolist(),
            "P_column_gram": gram.tolist(),
            "P_max_absolute_off_diagonal_similarity": float(np.max(np.abs(gram - np.diag(np.diag(gram))))),
            "Q_orthogonality_max_error": float(np.abs(q.T @ q - np.eye(8)).max()),
            "probe_n": n, "probe_min_input_distance": float(distances.min()) if n > 1 else None,
            "probe_median_nearest_input_distance": float(np.median(distances)) if n > 1 else None,
            "probe_nearest_different_target_fraction": float(np.any(targets(a[:n]) != targets(a[neighbor]), axis=1).mean()) if n > 1 else None,
            "minimum_norm_latent_reconstruction_mse": float(np.mean((reconstruction - probe)**2)),
            "interpretation": "Sampled proximity and pseudoinverse diagnostics only; not exhaustive sparse decoding or an injectivity proof."}


def create_dataset(config, path):
    d, m, seeds = config["data"], config["model"], config["seeds"]
    p, q = embeddings(m["d_in"], m["d_out"], seeds["input_embedding"], seeds["output_embedding"])
    a, mode, requested, generation = sample_dataset(d, seeds["data"])
    split = split_indices(len(a), d["split_fractions"], seeds["split"])
    contrast, memberships = contrast_set(d, seeds["contrast"])
    overlap = overlap_partitions(contrast, a, split)
    np.savez_compressed(path, a=a, g=targets(a), P=p, Q=q, mode=mode, requested_rule=requested,
                        contrast_a=contrast, contrast_g=targets(contrast), contrast_memberships=memberships,
                        contrast_overlap_partition=overlap, **split)
    return {"generation": generation, "full": audit_samples(a, mode),
            "splits": {key: audit_samples(a[idx], mode[idx]) for key, idx in split.items()},
            "contrast": {**audit_samples(contrast),
                         "overlap_counts": {name: int((overlap == i).sum()) for i, name in [(-1, "unseen"), (0, "train"), (1, "validation"), (2, "test")]},
                         "truly_unseen_fraction": float((overlap == -1).mean())},
            "geometry": geometry_diagnostics(p, q, a, d["geometry_probe_samples"])}
