import numpy as np
import pytest
import torch

from scripts.prepare_coverage import matched_rows
from src.data import targets, vector_keys
from src.train_mlp import balanced_sampling_weights, epoch_learning_rate


def test_balanced_weights_match_class_mixture_and_use_no_external_labels():
    labels = np.zeros((10, 2), dtype=np.float32)
    labels[0, 0] = 1
    labels[::2, 1] = 1
    weights = balanced_sampling_weights(labels).numpy()
    reference = np.zeros(10)
    for j in range(2):
        for label in (0, 1):
            members = labels[:, j] == label
            reference[members] += 1 / (2 * 2 * members.sum())
    np.testing.assert_allclose(weights, reference)
    assert weights @ labels[:, 0] > labels[:, 0].mean()
    with pytest.raises(ValueError, match="both classes"):
        balanced_sampling_weights(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="binary"):
        balanced_sampling_weights(np.full((10, 2), 0.3))


def test_weighted_shuffle_rng_restarts_exactly():
    weights = balanced_sampling_weights(np.array([[0, 0], [1, 1], [0, 1]]))
    generator = torch.Generator().manual_seed(2)
    torch.multinomial(weights, 20, replacement=True, generator=generator)
    state = generator.get_state()
    expected = torch.multinomial(weights, 20, replacement=True, generator=generator)
    resumed = torch.Generator()
    resumed.set_state(state)
    torch.testing.assert_close(expected, torch.multinomial(weights, 20, replacement=True, generator=resumed))


def test_epoch_schedule_has_exact_endpoints_and_no_hidden_resume_state():
    t = {"epochs": 101, "learning_rate": 0.001, "minimum_learning_rate": 0.00001,
         "lr_schedule": "cosine"}
    assert epoch_learning_rate(t, 1) == pytest.approx(0.001)
    assert epoch_learning_rate(t, 101) == pytest.approx(0.00001)
    rates = [epoch_learning_rate(t, i) for i in range(1, 102)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    t.pop("lr_schedule")
    assert epoch_learning_rate(t, 101) == t["learning_rate"]


def test_coverage_has_disjoint_rows_and_contexts_and_preserves_labels():
    settings = {"context_active_counts": [2, 3, 4, 5, 6],
                "context_probabilities": [0.15, 0.35, 0.30, 0.15, 0.05]}
    reserved = np.eye(32, dtype=np.uint8)
    a, contexts, members = matched_rows(8, 11, settings, vector_keys(reserved))
    b, val_contexts, val_members = matched_rows(4, 12, settings,
        np.concatenate([vector_keys(reserved), vector_keys(a)]), contexts)
    assert len(np.unique(vector_keys(a))) == len(a)
    assert not set(vector_keys(a)) & set(vector_keys(b))
    assert not set(map(tuple, contexts)) & set(map(tuple, val_contexts))
    assert np.all((a.sum(1) >= 2) & (a.sum(1) <= 10))
    assert a.sum(1).max() > 6
    assert np.all(members[:, 0] < len(a)) and np.all(val_members[:, 0] < len(b))
    assert targets(a).shape == (len(a), 8)
