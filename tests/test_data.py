from itertools import product

import numpy as np
import pytest

from src.data import (ANTECEDENTS, audit_samples, contrast_set, create_dataset, embeddings,
                      overlap_partitions, pattern_pools, sample_dataset, split_indices,
                      targets, vector_keys)
from src.runtime import load_config


@pytest.fixture
def config():
    c = load_config("configs/sparse_superposition_v1.yaml")
    c["data"].update(n_samples=256, min_samples=32, max_attempts=10000,
                      contrast_contexts_per_rule=3, geometry_probe_samples=32)
    return c


def reference(bits):
    # Independent scalar Boolean implementation of the eight SPEC formulas.
    b = list(map(bool, bits))
    return [b[0] and b[1], b[2] or b[3], b[4] and not b[5], b[6] != b[7],
            all(b[i] for i in (8, 9, 10)), (b[0] and b[11]) or (b[12] and b[13]),
            (b[2] or b[14]) and not b[15], (b[4] and b[16]) != b[17]]


def test_rules_exhaustive_local_tables_and_nuisance():
    rng = np.random.default_rng(42)
    for rule, ids in enumerate(ANTECEDENTS):
        for pattern in product((0, 1), repeat=len(ids)):
            a = rng.integers(0, 2, size=32, dtype=np.uint8)
            a[list(ids)] = pattern
            np.testing.assert_array_equal(targets(a), reference(a))
            flipped = a.copy()
            flipped[18:] = 1 - flipped[18:]
            np.testing.assert_array_equal(targets(a), targets(flipped))
    with pytest.raises(ValueError):
        targets(np.full(32, 0.2))


def test_p_is_overcomplete_q_is_orthonormal():
    p, q = embeddings(16, 12, 1, 2)
    assert p.shape == (16, 32) and q.shape == (12, 8)
    np.testing.assert_allclose(np.linalg.norm(p, axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-6)
    assert np.linalg.matrix_rank(p) == 16
    assert not np.allclose(p.T @ p, np.eye(32))
    pp, qq = embeddings(16, 12, 1, 2)
    np.testing.assert_array_equal(p, pp)
    np.testing.assert_array_equal(q, qq)


def test_near_miss_pools():
    for j, (positive, negative) in enumerate(pattern_pools()):
        for patterns, expected in [(positive, 1), (negative, 0)]:
            a = np.zeros((len(patterns), 32), dtype=np.uint8)
            a[:, ANTECEDENTS[j]] = patterns
            assert np.all(targets(a)[:, j] == expected)
        distance = np.abs(negative[:, None].astype(int) - positive[None].astype(int)).sum(-1)
        assert np.all(distance.min(1) == 1)
    np.testing.assert_array_equal(pattern_pools()[3][1], [[0, 0], [1, 1]])


def test_unique_reproducible_sampling_and_splits(config):
    a, mode, requested, report = sample_dataset(config["data"], 7)
    again = sample_dataset(config["data"], 7)
    np.testing.assert_array_equal(a, again[0])
    assert len(a) == report["actual_count"] == len(np.unique(vector_keys(a))) == 256
    assert np.all((a.sum(1) >= 2) & (a.sum(1) <= 6))
    g = targets(a)
    for i in np.flatnonzero(mode != 0):
        assert g[i, requested[i]] == (mode[i] == 1)
    split = split_indices(len(a), [0.7, 0.15, 0.15], 9)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(a)))
    for name, idx in split.items():
        np.testing.assert_array_equal(idx, split_indices(len(a), [0.7, 0.15, 0.15], 9)[name])
    key_sets = [set(vector_keys(a[idx])) for idx in split.values()]
    assert not key_sets[0] & key_sets[1] and not key_sets[0] & key_sets[2] and not key_sets[1] & key_sets[2]


def test_generation_has_a_bound(config):
    config["data"].update(max_attempts=40, n_samples=256, min_samples=3)
    with pytest.warns(UserWarning, match="obtained"):
        a, _, _, report = sample_dataset(config["data"], 3)
    assert len(a) <= 40 and report["attempts"] == 40
    config["data"]["min_samples"] = 100
    with pytest.raises(RuntimeError, match="below min_samples"):
        sample_dataset(config["data"], 3)


def test_contrast_complete_tables_fixed_context_and_overlap(config):
    a, membership = contrast_set(config["data"], 9)
    assert len(np.unique(vector_keys(a))) == len(a)
    for rule, ids in enumerate(ANTECEDENTS):
        for context in range(3):
            rows = a[membership[(membership[:, 1] == rule) & (membership[:, 2] == context), 0]]
            np.testing.assert_array_equal(rows[:, ids], list(product((0, 1), repeat=len(ids))))
            others = np.setdiff1d(np.arange(32), ids)
            assert np.all(rows[:, others] == rows[0, others])
    split = {"train": np.array([0]), "validation": np.array([1]), "test": np.array([2])}
    overlap = overlap_partitions(a, a[:3], split)
    np.testing.assert_array_equal(overlap[:3], [0, 1, 2])
    assert np.all(overlap[3:] == -1)


def test_saved_dataset_and_audit(config, tmp_path):
    path = tmp_path / "dataset.npz"
    report = create_dataset(config, path)
    with np.load(path, allow_pickle=False) as dataset:
        np.testing.assert_array_equal(dataset["g"], targets(dataset["a"]))
        np.testing.assert_array_equal(dataset["contrast_g"], targets(dataset["contrast_a"]))
        assert report["full"]["n"] == len(dataset["a"])
        assert report["geometry"]["probe_n"] == 32
        assert set(np.unique(dataset["contrast_overlap_partition"])) <= {-1, 0, 1, 2}
    assert audit_samples(np.empty((0, 32)))["n"] == 0


def test_model_seed_does_not_change_dataset(config, tmp_path):
    create_dataset(config, tmp_path / "one.npz")
    config["seeds"]["model"] += 1
    create_dataset(config, tmp_path / "two.npz")
    with np.load(tmp_path / "one.npz") as first, np.load(tmp_path / "two.npz") as second:
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
