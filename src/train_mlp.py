"""Validation-selected teacher training, with resumable epoch-boundary state."""

import hashlib
import json
import math
import platform
import random
import time

import numpy as np
import torch

from .models import OriginalMLP
from .runtime import load_checkpoint, save_checkpoint, seed_everything, write_json


def balanced_sampling_weights(labels):
    """Mixture: choose output uniformly, choose its class uniformly, choose a row.

    Uses training labels only. This changes the sampled training population,
    not the per-example mixed-output MSE or the model's input coordinates.
    """
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2 or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected a binary training label matrix")
    prevalence = labels.mean(0)
    if np.any((prevalence == 0) | (prevalence == 1)):
        raise ValueError("Balanced sampling requires both classes for every training output")
    weights = (labels / (2 * prevalence) + (1 - labels) / (2 * (1 - prevalence))).mean(1)
    return torch.from_numpy(weights / weights.sum())


def epoch_learning_rate(training, epoch):
    if training.get("lr_schedule", "constant") == "constant":
        return training["learning_rate"]
    fraction = (epoch - 1) / max(training["epochs"] - 1, 1)
    minimum = training["minimum_learning_rate"]
    return minimum + (training["learning_rate"] - minimum) * (1 + math.cos(math.pi * fraction)) / 2


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def model_from_config(config):
    m = config["model"]
    return OriginalMLP(m["d_in"], m["d_hidden"], m["d_out"])


def environment_record(device):
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "torch_cuda_build": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "note": "Determinism is best effort across software, hardware and device changes; exact cross-platform reproducibility is not promised."}


def verify_checkpoint(checkpoint, config, dataset_path):
    if checkpoint["config_sha256"] != config_digest(config):
        raise ValueError("Checkpoint configuration differs from saved run configuration")
    if checkpoint["dataset_sha256"] != file_digest(dataset_path):
        raise ValueError("Dataset changed since checkpoint creation")


@torch.no_grad()
def validation_mse(model, a, g, p, q, indices, device, batch_size):
    model.eval()
    total, elements = 0.0, 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start+batch_size]
        x = torch.as_tensor(a[idx], dtype=torch.float32, device=device) @ p.T
        y = torch.as_tensor(g[idx], dtype=torch.float32, device=device) @ q.T
        error = (model(x) - y).square()
        if not torch.isfinite(error).all():
            raise RuntimeError("Non-finite validation error")
        total += error.sum().item()
        elements += error.numel()
    return total / elements


def train(config, dataset, run_dir, device):
    t = config["training"]
    seed_everything(config["seeds"]["model"], t["deterministic"])
    model = model_from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(config["seeds"]["model"])
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history, best_loss, best_epoch, first_epoch = [], float("inf"), 0, 1
    dataset_hash = file_digest(run_dir / "dataset.npz")
    if last_path.exists():
        checkpoint = load_checkpoint(last_path)
        verify_checkpoint(checkpoint, config, run_dir / "dataset.npz")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_loss, best_epoch = checkpoint["history"], checkpoint["best_validation_mse"], checkpoint["best_epoch"]
        first_epoch = checkpoint["epoch"] + 1
        generator.set_state(checkpoint["shuffle_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            if len(checkpoint["cuda_rng"]) != torch.cuda.device_count():
                raise RuntimeError("CUDA device count changed; exact RNG restoration is unavailable")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"Resuming after epoch {checkpoint['epoch']}; best epoch={best_epoch}", flush=True)
    environment = environment_record(device)
    write_json(run_dir / "environment.json", environment)
    print(f"Environment: {json.dumps(environment)}", flush=True)
    p, q = (torch.as_tensor(dataset[k], device=device) for k in ("P", "Q"))
    a, g, indices = dataset["a"], dataset["g"], dataset["train"]
    sampling_weights = balanced_sampling_weights(g[indices]) if t.get("sampling", "uniform") == "balanced_labels" else None
    if sampling_weights is not None:
        write_json(run_dir / "sampling_diagnostics.json", {
            "original_train_prevalence": g[indices].mean(0).tolist(),
            "expected_sampled_prevalence": (sampling_weights.numpy() @ g[indices]).tolist(),
            "effective_sample_size": float(1 / sampling_weights.square().sum()),
            "note": "With-replacement training resampling only; validation/test are never reweighted."})
    print(f"Training {len(indices)} samples; validation {len(dataset['validation'])}; config={config}", flush=True)
    for epoch in range(first_epoch, t["epochs"] + 1):
        started = time.monotonic()
        model.train()
        lr = epoch_learning_rate(t, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampled = (torch.randperm(len(indices), generator=generator) if sampling_weights is None else
                   torch.multinomial(sampling_weights, len(indices), replacement=True, generator=generator))
        order = indices[sampled.numpy()]
        total, elements = 0.0, 0
        for start in range(0, len(order), t["batch_size"]):
            idx = order[start:start+t["batch_size"]]
            # Only x is passed to model. a/g/Q are used to construct input/target.
            x = torch.as_tensor(a[idx], dtype=torch.float32, device=device) @ p.T
            y = torch.as_tensor(g[idx], dtype=torch.float32, device=device) @ q.T
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y, reduction="mean")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}")
            optimizer.step()
            total += loss.item() * y.numel()
            elements += y.numel()
        val = validation_mse(model, a, g, p, q, dataset["validation"], device, config["evaluation"]["batch_size"])
        improved = val < best_loss
        if improved:
            best_loss, best_epoch = val, epoch
        row = {"epoch": epoch, "train_online_mse": total / elements, "validation_mse": val,
               "best_epoch": best_epoch, "seconds": time.monotonic() - started, "learning_rate": lr}
        history.append(row)
        state = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_epoch": best_epoch, "best_validation_mse": best_loss,
                 "history": history, "config": config, "config_sha256": config_digest(config),
                 "dataset_sha256": dataset_hash, "shuffle_rng": generator.get_state(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        if improved:
            save_checkpoint(best_path, state)
        save_checkpoint(last_path, state)
        write_json(run_dir / "history.json", history)
        print(json.dumps(row), flush=True)
    print(f"Training complete. Selected epoch {best_epoch}; validation MSE {best_loss:.8g}. "
          "Run the separate evaluation command; Stage 1 is not automatically approved.", flush=True)
