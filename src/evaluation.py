"""Teacher task evaluation only. No surrogate or original-neuron circuit analysis."""

import csv
from pathlib import Path

import numpy as np
import torch

from .data import RULE_IDS
from .runtime import write_json


def prediction_metrics(prediction, g, q, threshold=0.5):
    if len(g) == 0:
        return {"n": 0, "note": "Empty subset; metrics undefined"}
    prediction, g, q = (np.asarray(v, dtype=np.float64) for v in (prediction, g, q))
    if prediction.shape != (len(g), q.shape[0]) or g.shape[1] != 8 or q.shape[1] != 8:
        raise ValueError("Invalid prediction, target or Q shapes")
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions")
    semantic = prediction @ q  # Q.T @ F, not probabilities or a fitted probe.
    labels = semantic >= threshold
    truth = g.astype(bool)
    residual = prediction - semantic @ q.T
    per_rule = []
    for j, name in enumerate(RULE_IDS):
        t, p = truth[:, j], labels[:, j]
        tp, tn = int((t & p).sum()), int((~t & ~p).sum())
        fp, fn = int((~t & p).sum()), int((t & ~p).sum())
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        per_rule.append({"rule": name, "n_positive": int(t.sum()), "n_negative": int((~t).sum()),
                         "prevalence": float(t.mean()), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                         "errors": fp + fn, "accuracy": float((t == p).mean()),
                         "precision": tp / (tp + fp) if tp + fp else None,
                         "recall": recall, "specificity": specificity,
                         "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
                         "semantic_mse": float(((semantic[:, j] - g[:, j])**2).mean()),
                         "positive_mse": float(((semantic[t, j] - 1)**2).mean()) if t.any() else None,
                         "negative_mse": float((semantic[~t, j]**2).mean()) if (~t).any() else None,
                         "prediction_min": float(semantic[:, j].min()), "prediction_max": float(semantic[:, j].max())})
    return {"n": len(g), "classification_threshold": threshold,
            "output_mse": float(((prediction - g @ q.T)**2).mean()),
            "semantic_mse": float(((semantic - g)**2).mean()),
            "orthogonal_output_residual_mse": float((residual**2).mean()),
            "semantic_error_output_normalization": float(((semantic - g)**2).sum(1).mean() / q.shape[0]),
            "exact_vector_accuracy": float(np.all(labels == truth, axis=1).mean()),
            "per_rule": per_rule,
            "note": "Decoded values are real regression scores. Thresholding does not make them calibrated probabilities."}


@torch.no_grad()
def predict(model, a, p, device, batch_size, capture_hidden=False):
    model.eval()
    p_device = torch.as_tensor(p, device=device)
    outputs, preactivations, activations = [], [], []
    for start in range(0, len(a), batch_size):
        semantic_inputs = torch.as_tensor(a[start:start+batch_size], dtype=torch.float32, device=device)
        x = semantic_inputs @ p_device.T
        pre, hidden = model.hidden(x)
        out = model.W2(hidden)
        if not torch.isfinite(out).all() or not torch.isfinite(pre).all():
            raise RuntimeError("Non-finite model output/hidden state during evaluation")
        outputs.append(out.cpu().numpy())
        if capture_hidden:
            preactivations.append(pre.cpu().numpy())
            activations.append(hidden.cpu().numpy())
    result = {"output": np.concatenate(outputs)}
    if capture_hidden:
        result.update(preactivation=np.concatenate(preactivations), hidden=np.concatenate(activations))
    return result


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError("Empty CSV requires explicit fieldnames")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(model, dataset, config, device, output_dir, checkpoint_epoch):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    a, g, p, q = (dataset[k] for k in ("a", "g", "P", "Q"))
    batch = config["evaluation"]["batch_size"]
    threshold = config["evaluation"]["classification_threshold"]
    ordinary = predict(model, a, p, device, batch, capture_hidden=True)
    contrasts = predict(model, dataset["contrast_a"], p, device, batch, capture_hidden=True)
    metrics = {"checkpoint_epoch": checkpoint_epoch, "partitions": {}}
    for name in ("train", "validation", "test"):
        idx = dataset[name]
        metrics["partitions"][name] = prediction_metrics(ordinary["output"][idx], g[idx], q, threshold)
    metrics["partitions"]["full_sample_includes_training"] = prediction_metrics(ordinary["output"], g, q, threshold)
    # A train-only constant comparator makes the sparse-label imbalance explicit.
    mean_target = g[dataset["train"]].mean(0) @ q.T
    metrics["train_mean_baseline"] = {}
    for name in ("train", "validation", "test"):
        idx = dataset[name]
        metrics["train_mean_baseline"][name] = prediction_metrics(
            np.broadcast_to(mean_target, (len(idx), len(mean_target))), g[idx], q, threshold)
    overlap = dataset["contrast_overlap_partition"]
    subsets = {"contrast_all": np.ones(len(overlap), dtype=bool), "contrast_unseen": overlap == -1,
               "contrast_seen_train": overlap == 0, "contrast_seen_validation": overlap == 1,
               "contrast_seen_test": overlap == 2}
    for name, mask in subsets.items():
        metrics["partitions"][name] = prediction_metrics(contrasts["output"][mask], dataset["contrast_g"][mask], q, threshold)
    metrics["contrast_unseen_fraction"] = float((overlap == -1).mean())
    # Each rule's own complete antecedent tables, with unseen subsets separate.
    members = dataset["contrast_memberships"]
    metrics["contrast_by_designed_rule"] = {}
    for j, name in enumerate(RULE_IDS):
        ids = np.unique(members[members[:, 1] == j, 0])
        unseen = ids[overlap[ids] == -1]
        metrics["contrast_by_designed_rule"][name] = {
            "all": prediction_metrics(contrasts["output"][ids], dataset["contrast_g"][ids], q, threshold),
            "unseen": prediction_metrics(contrasts["output"][unseen], dataset["contrast_g"][unseen], q, threshold),
            "complete_contexts_all_unseen": sum(
                bool(np.all(overlap[members[(members[:, 1] == j) & (members[:, 2] == context), 0]] == -1))
                for context in np.unique(members[members[:, 1] == j, 2]))}
    metrics["hidden_diagnostics"] = {
        "activation_prevalence": (ordinary["hidden"] > 0).mean(0).tolist(),
        "mean_activation": ordinary["hidden"].mean(0).tolist(),
        "preactivation_min": ordinary["preactivation"].min(0).tolist(),
        "preactivation_max": ordinary["preactivation"].max(0).tolist()}
    metrics["parameter_diagnostics"] = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().numpy()
        metrics["parameter_diagnostics"][name] = {
            "shape": list(value.shape), "frobenius_or_l2_norm": float(np.linalg.norm(value)),
            "minimum": float(value.min()), "maximum": float(value.max()),
            "singular_values": np.linalg.svd(value, compute_uv=False).tolist() if value.ndim == 2 else None}
    rows, failures = [], []
    for partition, values in metrics["partitions"].items():
        rows.extend({"partition": partition, **row} for row in values.get("per_rule", []))
    for source, inputs, target, prediction in [
        ("sample", a, g, ordinary["output"]),
        ("contrast", dataset["contrast_a"], dataset["contrast_g"], contrasts["output"])]:
        # Match the float64 projection used by prediction_metrics at the threshold.
        decoded = prediction.astype(np.float64) @ q.astype(np.float64)
        bad = (decoded >= threshold) != target.astype(bool)
        partition_names = np.full(len(inputs), "unseen", dtype=object)
        if source == "sample":
            for name in ("train", "validation", "test"):
                partition_names[dataset[name]] = name
        else:
            for i, name in enumerate(("train", "validation", "test")):
                partition_names[overlap == i] = name
        for i, j in np.argwhere(bad):
            failures.append({"source": source, "row": int(i), "partition": str(partition_names[i]),
                             "rule": RULE_IDS[j], "active_inputs": " ".join(map(str, np.flatnonzero(inputs[i]))),
                             "target": int(target[i, j]), "score": float(decoded[i, j])})
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "per_rule.csv", rows)
    write_csv(output_dir / "classification_errors.csv", failures,
              ["source", "row", "partition", "rule", "active_inputs", "target", "score"])
    np.savez_compressed(output_dir / "predictions_activations.npz",
                        **{f"sample_{k}": v for k, v in ordinary.items()},
                        **{f"contrast_{k}": v for k, v in contrasts.items()})
    review = []
    for partition in ("test", "contrast_unseen"):
        if metrics["partitions"][partition]["n"] == 0:
            review.append(f"{partition}: empty; performance cannot be assessed")
        for row in metrics["partitions"][partition].get("per_rule", []):
            accuracy = row["balanced_accuracy"]
            if accuracy is None or accuracy < config["evaluation"]["review_min_balanced_accuracy"] or row["semantic_mse"] > config["evaluation"]["review_max_semantic_mse"]:
                review.append(f"{partition}/{row['rule']}: balanced_accuracy={accuracy}, semantic_mse={row['semantic_mse']:.6g}")
    report = ["# Original MLP baseline", "", f"Selected checkpoint epoch: {checkpoint_epoch}.",
              "Validation MSE selected the checkpoint; this report does not establish mechanism recovery.",
              f"Truly unseen contrast fraction: {metrics['contrast_unseen_fraction']:.3f}.", "",
              "Review flags (configurable heuristics, not a scientific go/no-go decision):"]
    report += [f"- {line}" for line in review] or ["- No configured numerical flags; scientific review is still required."]
    report += ["", "All classification errors are retained in classification_errors.csv.",
               "Regression scores are unbounded; contrast density can differ from training.",
               "No transcoder fidelity or original-neuron causal claims are evaluated."]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    plot_baseline(output_dir, metrics, p, q)
    return metrics


def plot_baseline(output_dir, metrics, p, q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(p.T @ p, vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_title("P column similarities")
    axes[1].imshow(q.T @ q, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title("Q orthogonality")
    rules = metrics["partitions"]["test"]["per_rule"]
    axes[2].bar(np.arange(8), [r["balanced_accuracy"] if r["balanced_accuracy"] is not None else np.nan for r in rules])
    for j, row in enumerate(rules):
        if row["balanced_accuracy"] is None:
            axes[2].text(j, 0.05, "NA", ha="center")
    axes[2].set_xticks(np.arange(8), [f"g{j}" for j in range(8)])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Test balanced accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline.png", dpi=160)
    plt.close(fig)
