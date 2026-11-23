import numpy as np
import pytest

from scripts.tune_stage1 import fresh_sets, selection_key
from src.data import vector_keys
from src.runtime import load_config


def test_candidate_selection_uses_worst_rule_before_average_or_mse():
    def metrics(values, mse):
        return {"per_rule": [{"balanced_accuracy": v} for v in values], "output_mse": mse}
    assert selection_key(metrics([0.9, 0.9], 0.2)) > selection_key(metrics([0.8, 1.0], 0.01))
    assert selection_key(metrics([0.9, 1.0], 0.2)) > selection_key(metrics([0.9, 0.95], 0.01))
    assert selection_key(metrics([0.9, 1.0], 0.01)) > selection_key(metrics([0.9, 1.0], 0.02))
    with pytest.raises(ValueError, match="lacks"):
        selection_key(metrics([None], 0.01))


def test_fresh_sets_are_disjoint_and_memberships_are_valid():
    c = load_config("configs/sparse_superposition_v1.yaml")
    c["data"].update(min_samples=8, max_attempts=1000)
    original = {"a": np.eye(32, dtype=np.uint8), "contrast_a": np.zeros((1, 32), dtype=np.uint8)}
    plan = {"fresh_proposal_samples": 64, "fresh_sample_limit": 32,
            "fresh_data_seed": 30, "fresh_contrast_seed": 31, "fresh_contexts_per_rule": 2}
    fresh, audit = fresh_sets(original, c, plan)
    old = set(vector_keys(original["a"])) | set(vector_keys(original["contrast_a"]))
    sample, contrast = set(vector_keys(fresh["sample_a"])), set(vector_keys(fresh["contrast_a"]))
    assert not old & sample and not old & contrast and not sample & contrast
    assert len(sample) == len(fresh["sample_a"]) == audit["sample"]["n"]
    assert len(contrast) == len(fresh["contrast_a"])
    assert np.all(fresh["contrast_memberships"][:, 0] < len(contrast))
    assert np.all(fresh["contrast_memberships"][:, 0] >= 0)
