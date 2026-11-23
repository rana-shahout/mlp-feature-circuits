#!/usr/bin/env python3
"""Frozen-protocol initialization replication; every seed is reported."""

import argparse
import csv
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.evaluation import predict, prediction_metrics
from src.runtime import load_checkpoint, load_config, select_device, write_json
from src.train_mlp import config_digest, file_digest, model_from_config, train, verify_checkpoint


def summarize(records, seeds):
    by_seed = {row["seed"]: row for row in records}
    if len(by_seed) != len(records) or set(by_seed) != set(seeds):
        raise ValueError("Need exactly one result for every prespecified seed")
    summary = {"seeds": seeds, "per_seed": [by_seed[seed] for seed in seeds]}
    for population in ("sample", "contrast"):
        values = [by_seed[seed][population]["exact_vector_accuracy"] for seed in seeds]
        summary[population] = {"exact_accuracy_mean": float(np.mean(values)),
                               "exact_accuracy_std_across_training_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               "exact_accuracy_min": min(values), "exact_accuracy_max": max(values),
                               "per_rule_recall_min": np.min([[r["recall"] for r in by_seed[seed][population]["per_rule"]] for seed in seeds], axis=0).tolist(),
                               "per_rule_balanced_accuracy_min": np.min([[r["balanced_accuracy"] for r in by_seed[seed][population]["per_rule"]] for seed in seeds], axis=0).tolist()}
    summary["note"] = "Exploratory training-seed robustness (initialization and row-resampling randomness) on a fixed held-out population. No best-seed selection. Across-seed standard deviation is not a confidence interval over new contexts or P/Q."
    return summary


def write_evaluation_artifacts(root, result):
    """Figures and complete error rows; no new model selection or inference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = json.loads((root / "plan.json").read_text())["plan"]
    baseline = json.loads(Path(plan["reference_metrics"]).read_text())["baseline"]
    reference_seed = load_config(Path(plan["source_run"]) / "config.yaml")["seeds"]["model"]
    threshold = load_config(Path(plan["source_run"]) / "config.yaml")["evaluation"]["classification_threshold"]
    labels = ["prior 256"] + [str(row["seed"]) for row in result["per_seed"]]
    records = [baseline] + result["per_seed"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, population, title, key in [(axes[0], "sample", "Fresh sample: all 8 outputs", "exact_vector_accuracy"),
                                      (axes[1], "contrast", "Fresh contrast: all 8 outputs", "exact_vector_accuracy"),
                                      (axes[2], "contrast", "Three-way AND: contrast recall", None)]:
        values = [r[population][key] if key else r[population]["per_rule"][4]["recall"] for r in records]
        ax.bar(labels, values, color=["gray"] + ["steelblue"] * len(result["per_seed"]))
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Training seed (fixed v2 protocol)")
    fig.suptitle("Same held-out inputs; v2 broadens training density support")
    fig.tight_layout()
    fig.savefig(root / "replication.png", dpi=160)
    plt.close(fig)
    with np.load(plan["holdout"], allow_pickle=False) as fresh, open(root / "classification_errors.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "population", "row", "rule", "target", "score", "active_inputs"])
        writer.writeheader()
        for row in result["per_seed"]:
            seed = row["seed"]
            source = (Path(plan["reference_metrics"]).parent / "fresh_predictions.npz" if seed == reference_seed else root / f"seed{seed}" / "fresh_predictions.npz")
            with np.load(source, allow_pickle=False) as predictions:
                for population in ("sample", "contrast"):
                    key = f"selected_{population}" if seed == reference_seed else population
                    decoded = predictions[key].astype(np.float64) @ fresh["Q"].astype(np.float64)
                    truth = fresh[f"{population}_g"]
                    inputs = fresh[f"{population}_a"]
                    for i, j in np.argwhere((decoded >= threshold) != truth.astype(bool)):
                        writer.writerow({"seed": seed, "population": population, "row": int(i),
                                         "rule": row[population]["per_rule"][j]["rule"], "target": int(truth[i, j]),
                                         "score": float(decoded[i, j]), "active_inputs": " ".join(map(str, np.flatnonzero(inputs[i])))})


def update_summary(root, seeds):
    with open(root / ".summary.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        paths = [root / f"seed{seed}" / "replication_metrics.json" for seed in seeds]
        if not all(path.exists() for path in paths):
            return
        result = summarize([json.loads(path.read_text()) for path in paths], seeds)
        write_json(root / "summary.json", result)
        lines = ["# Stage 1 frozen-protocol replication", "",
                 "All prespecified seeds are reported. During this frozen replication, no hyperparameters or thresholds were changed using holdout results.", "",
                 "| Seed | Fresh sample exact accuracy | Fresh contrast exact accuracy | Three-way AND contrast recall |", "|---|---:|---:|---:|"]
        for row in result["per_seed"]:
            lines.append(f"| {row['seed']} | {row['sample']['exact_vector_accuracy']:.4f} | {row['contrast']['exact_vector_accuracy']:.4f} | {row['contrast']['per_rule'][4]['recall']:.4f} |")
        for population in ("sample", "contrast"):
            stats = result[population]
            lines += ["", f"Fresh {population} exact accuracy: {100*stats['exact_accuracy_mean']:.2f}% mean, "
                      f"{100*stats['exact_accuracy_std_across_training_seeds']:.2f} percentage-point standard deviation across seeds."]
        lines += ["", result["note"], "Training includes the documented density extension and balanced-label resampling.",
                  "Earlier diagnostic holdouts informed the v2 design; this remains an exploratory adaptive follow-up. The v2 fresh sets were first evaluated after validation selection, then reused only for the frozen replication.",
                  "P/Q and dataset remain fixed; this does not establish dictionary-seed robustness or internal mechanisms.",
                  "No transcoder or original-neuron analysis was implemented.", "",
                  "![All prespecified seeds compared on identical holdouts](replication.png)", "",
                  "Every misclassified output is retained in classification_errors.csv."]
        write_evaluation_artifacts(root, result)
        (root / "report.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/stage1_replication_v2.yaml")
    parser.add_argument("--run-dir", default="results/stage1-replication-v2")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text())
    if not set(args.seeds).issubset(plan["seeds"]) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Worker seeds must be distinct prespecified seeds")
    root, source = Path(args.run_dir), Path(plan["source_run"])
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(source / "config.yaml")
    hashes = {"plan": plan, "source_config": config_digest(config),
              "training_data": file_digest(source / "dataset.npz"), "holdout": file_digest(plan["holdout"]),
              "reference_metrics": file_digest(plan["reference_metrics"])}
    with open(root / ".plan.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        frozen = root / "plan.json"
        if frozen.exists() and json.loads(frozen.read_text()) != hashes:
            raise RuntimeError("Replication protocol or data changed; use a new directory")
        write_json(frozen, hashes)
        reference = root / f"seed{config['seeds']['model']}"
        reference.mkdir(exist_ok=True)
        baseline = json.loads(Path(plan["reference_metrics"]).read_text())["selected"]
        write_json(reference / "replication_metrics.json", {"seed": config["seeds"]["model"], **baseline})
    device = select_device("cuda")
    for seed in args.seeds:
        path = root / f"seed{seed}"
        path.mkdir(exist_ok=True)
        with open(path / ".lock", "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if (path / "replication_metrics.json").exists():
                    print(f"Seed {seed} already complete", flush=True)
                    continue
                c = deepcopy(config)
                c["version"] = plan["version"] + f"_seed{seed}"
                c["seeds"]["model"] = seed
                if (path / "config.yaml").exists():
                    if load_config(path / "config.yaml") != c:
                        raise RuntimeError("Existing replication config differs")
                    if file_digest(path / "dataset.npz") != hashes["training_data"]:
                        raise RuntimeError("Replication dataset differs")
                else:
                    (path / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
                    for name in ("dataset.npz", "data_audit.json", "SPEC.snapshot.md"):
                        shutil.copyfile(source / name, path / name)
                    write_json(path / "data_manifest.json", {"config_sha256": config_digest(c), "dataset_sha256": hashes["training_data"]})
                print(f"REPLICATION seed={seed}", flush=True)
                with np.load(path / "dataset.npz", allow_pickle=False) as data:
                    train(c, data, path, device)
                    checkpoint = load_checkpoint(path / "best.pt")
                    verify_checkpoint(checkpoint, c, path / "dataset.npz")
                    model = model_from_config(c).to(device)
                    model.load_state_dict(checkpoint["model"])
                    idx = data["validation"]
                    prediction = predict(model, data["a"][idx], data["P"], device, c["evaluation"]["batch_size"])["output"]
                    validation = prediction_metrics(prediction, data["g"][idx], data["Q"])
                    # Persist each validation-selected checkpoint decision before holdout scoring.
                    write_json(path / "validation_metrics.json", validation)
                    result = {"seed": seed, "epoch": checkpoint["epoch"], "width": c["model"]["d_hidden"],
                              "validation": validation, "checkpoint_sha256": file_digest(path / "best.pt")}
                    cached = {}
                    with np.load(plan["holdout"], allow_pickle=False) as fresh:
                        np.testing.assert_array_equal(data["P"], fresh["P"])
                        np.testing.assert_array_equal(data["Q"], fresh["Q"])
                        for population in ("sample", "contrast"):
                            prediction = predict(model, fresh[f"{population}_a"], data["P"], device, c["evaluation"]["batch_size"])["output"]
                            cached[population] = prediction
                            result[population] = prediction_metrics(prediction, fresh[f"{population}_g"], data["Q"])
                    np.savez_compressed(path / "fresh_predictions.npz", **cached)
                    write_json(path / "replication_metrics.json", result)
                    print(f"REPLICATION COMPLETE seed={seed}; contrast exact={result['contrast']['exact_vector_accuracy']:.4f}; g4 recall={result['contrast']['per_rule'][4]['recall']:.4f}", flush=True)
            except BaseException:
                (path / "failure.log").write_text(traceback.format_exc())
                raise
        update_summary(root, plan["seeds"])
    update_summary(root, plan["seeds"])


if __name__ == "__main__":
    main()
