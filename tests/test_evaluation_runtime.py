import numpy as np
import pytest
import torch

from src.data import embeddings
from src.evaluation import prediction_metrics, predict
from src.models import OriginalMLP
from src.runtime import (load_checkpoint, load_config, save_checkpoint, select_device,
                         validate_config, write_json)
from src.train_mlp import config_digest, file_digest, verify_checkpoint


def test_perfect_metrics_and_output_error_decomposition():
    _, q = embeddings(16, 12, 8, 9)
    g = np.tile(np.array([[0]*8, [1]*8], dtype=float), (4, 1))
    perfect = prediction_metrics(g @ q.T, g, q)
    assert perfect["exact_vector_accuracy"] == 1
    assert all(row["balanced_accuracy"] == 1 for row in perfect["per_rule"])
    rng = np.random.default_rng(12)
    y = rng.normal(size=(len(g), 12))
    result = prediction_metrics(y, g, q)
    assert result["output_mse"] == pytest.approx(
        result["semantic_error_output_normalization"] + result["orthogonal_output_residual_mse"], abs=1e-6)


def test_confusion_counts_and_missing_class_are_explicit():
    _, q = embeddings(16, 12, 1, 2)
    g = np.tile(np.array([[0]*8, [0]*8, [1]*8, [1]*8]), (1, 1))
    scores = np.tile(np.array([[0.], [1.], [0.], [1.]]), (1, 8))
    r = prediction_metrics(scores @ q.T, g, q)["per_rule"][0]
    assert (r["tp"], r["tn"], r["fp"], r["fn"]) == (1, 1, 1, 1)
    assert r["balanced_accuracy"] == 0.5
    missing = prediction_metrics(np.zeros((4, 12)), np.zeros((4, 8)), q)["per_rule"][0]
    assert missing["recall"] is None and missing["balanced_accuracy"] is None and missing["precision"] is None
    assert prediction_metrics(np.empty((0, 12)), np.empty((0, 8)), q)["n"] == 0


def test_cuda_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        select_device()
    assert select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="Only cuda"):
        select_device("mps")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_checkpoint_round_trip_and_batched_inference(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA round-trip requires CUDA")
    model = OriginalMLP().to(device)
    p, _ = embeddings(16, 12, 1, 2)
    a = np.eye(32, dtype=np.uint8)
    before = predict(model, a, p, torch.device(device), 7, capture_hidden=True)
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"model": model.state_dict(), "epoch": 1})
    other = OriginalMLP().to(device)
    other.load_state_dict(load_checkpoint(path)["model"])
    after = predict(other, a, p, torch.device(device), 7, capture_hidden=True)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert not path.with_suffix(".pt.tmp").exists()


def test_checkpoint_integrity_and_strict_json(tmp_path):
    config = load_config("configs/sparse_superposition_v1.yaml")
    path = tmp_path / "data.npz"
    path.write_bytes(b"fixture")
    checkpoint = {"config_sha256": config_digest(config), "dataset_sha256": file_digest(path)}
    verify_checkpoint(checkpoint, config, path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Dataset changed"):
        verify_checkpoint(checkpoint, config, path)
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"x": float("nan")})


def test_configuration_rejects_invalid_geometry_and_rules():
    c = load_config("configs/sparse_superposition_v1.yaml")
    assert c["device"] == "cuda" and c["model"] == {"d_in": 16, "d_hidden": 48, "d_out": 12}
    c["model"]["d_in"] = 32
    with pytest.raises(ValueError, match="overcomplete"):
        validate_config(c)


def test_evaluation_artifacts_agree_with_metrics(tmp_path):
    import csv
    import json

    from src.data import create_dataset
    from src.evaluation import evaluate_run

    c = load_config("configs/sparse_superposition_v1.yaml")
    c["data"].update(n_samples=64, min_samples=16, max_attempts=1000,
                      contrast_contexts_per_rule=1, geometry_probe_samples=16)
    create_dataset(c, tmp_path / "dataset.npz")
    with np.load(tmp_path / "dataset.npz", allow_pickle=False) as dataset:
        result = evaluate_run(OriginalMLP(), dataset, c, torch.device("cpu"), tmp_path / "evaluation", 0)
        with np.load(tmp_path / "evaluation" / "predictions_activations.npz") as cache:
            assert cache["sample_preactivation"].shape == (64, 48)
            assert cache["contrast_output"].shape == (len(dataset["contrast_a"]), 12)
    saved = json.loads((tmp_path / "evaluation" / "metrics.json").read_text())
    assert saved["partitions"]["test"]["n"] == result["partitions"]["test"]["n"]
    with open(tmp_path / "evaluation" / "classification_errors.csv") as handle:
        errors = list(csv.DictReader(handle))
    expected = sum(row["errors"] for name in ("full_sample_includes_training", "contrast_all")
                   for row in result["partitions"][name]["per_rule"])
    assert len(errors) == expected
    assert (tmp_path / "evaluation" / "report.md").is_file()
    assert (tmp_path / "evaluation" / "baseline.png").is_file()
