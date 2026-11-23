"""Device, reproducibility, atomic artifacts and configuration validation."""

import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml


def select_device(name="cuda"):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cuda[:index] and explicit cpu are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested but unavailable. Python: {sys.executable}; "
                f"PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}. "
                "Check nvidia-smi in the host environment and ensure this PyTorch CUDA "
                "build is compatible with the installed NVIDIA driver. CUDA-enabled "
                "PyTorch alone is insufficient if its runtime requires a newer driver. "
                "No CPU fallback is performed.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        torch.cuda.set_device(index)
        print(f"Device: cuda:{index} — {torch.cuda.get_device_name(index)}", flush=True)
        return torch.device("cuda", index)
    print("Device: cpu (explicit manual override)", flush=True)
    return device


def seed_everything(seed, deterministic):
    # Must precede the first CUDA computation. Deterministic unsupported operations fail.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path):
    # Our local checkpoints contain Python/NumPy RNG state. Never load untrusted files.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_config(path):
    with open(path) as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(c):
    from .data import RULE_IDS

    d, m, t, e = c["data"], c["model"], c["training"], c["evaluation"]
    if d["n_features"] != 32 or list(d["rule_ids"]) != list(RULE_IDS):
        raise ValueError("v1 requires all 32 inputs and all eight SPEC rules in order")
    if any(not isinstance(m[key], int) for key in ("d_in", "d_hidden", "d_out")):
        raise ValueError("Model dimensions must be integers")
    if not (0 < m["d_in"] < 32 and m["d_out"] >= 8 and m["d_hidden"] > 0):
        raise ValueError("Require overcomplete P, positive hidden width, and d_out >= 8")
    for key, count in [("mode_probabilities", 3), ("split_fractions", 3),
                       ("active_probabilities", len(d["active_counts"]))]:
        values = np.asarray(d[key], dtype=float)
        if len(values) != count or not np.all(np.isfinite(values)) or np.any(values <= 0) or not np.isclose(values.sum(), 1):
            raise ValueError(f"{key} must be positive probabilities summing to one")
    counts = d["active_counts"]
    if len(set(counts)) != len(counts) or any(not isinstance(k, int) or not 2 <= k <= 6 for k in counts):
        raise ValueError("v1 active_counts must be distinct integers between 2 and 6")
    if not (3 in counts and 4 in counts):
        raise ValueError("Include active counts 3 and 4 to support all v1 contrast patterns")
    for key in ["n_samples", "max_attempts", "min_samples", "contrast_contexts_per_rule", "geometry_probe_samples"]:
        if not isinstance(d[key], int) or d[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 3 <= d["min_samples"] <= d["n_samples"]:
        raise ValueError("Require 3 <= min_samples <= n_samples")
    for key in ["epochs", "batch_size"]:
        if not isinstance(t[key], int) or t[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not np.all(np.isfinite([t["learning_rate"], t["weight_decay"]])):
        raise ValueError("Learning rate and weight decay must be finite")
    if t["optimizer"] != "adam" or t["learning_rate"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Use adam with positive learning rate and nonnegative weight decay")
    if t.get("sampling", "uniform") not in {"uniform", "balanced_labels"}:
        raise ValueError("Unknown training sampling strategy")
    if t.get("lr_schedule", "constant") not in {"constant", "cosine"}:
        raise ValueError("Unknown learning-rate schedule")
    if t.get("lr_schedule") == "cosine":
        minimum = t.get("minimum_learning_rate", -1)
        if not np.isfinite(minimum) or not 0 < minimum <= t["learning_rate"]:
            raise ValueError("Cosine minimum learning rate must be positive and <= initial rate")
    if not isinstance(e["batch_size"], int) or e["batch_size"] <= 0 or not 0 < e["classification_threshold"] < 1:
        raise ValueError("Invalid evaluation batch size or classification threshold")
    if not np.all(np.isfinite([e["review_min_balanced_accuracy"], e["review_max_semantic_mse"]])):
        raise ValueError("Review thresholds must be finite")
    if not 0 <= e["review_min_balanced_accuracy"] <= 1 or e["review_max_semantic_mse"] < 0:
        raise ValueError("Invalid review flag thresholds")
    if not isinstance(t["deterministic"], bool):
        raise ValueError("deterministic must be a Boolean")
    for name in ["input_embedding", "output_embedding", "data", "split", "contrast", "model"]:
        seed = c["seeds"][name]
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"Invalid {name} seed")
