#!/usr/bin/env python3
"""Bounded Stage 1 tuning using validation only, followed by one fresh evaluation."""

import argparse
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

from src.data import audit_samples, contrast_set, sample_dataset, targets, vector_keys
from src.evaluation import predict, prediction_metrics
from src.runtime import load_checkpoint, load_config, seed_everything, select_device, write_json
from src.train_mlp import (config_digest, environment_record, file_digest, model_from_config,
                           train, verify_checkpoint)


def selection_key(metrics):
    """Predeclared lexicographic ranking; undefined class support cannot win."""
    scores = [r["balanced_accuracy"] for r in metrics["per_rule"]]
    if any(v is None for v in scores):
        raise ValueError("Validation lacks positive/negative support for a rule")
    return min(scores), sum(scores) / len(scores), -metrics["output_mse"]


def fresh_sets(dataset, config, plan):
    d = deepcopy(config["data"])
    d["n_samples"] = plan["fresh_proposal_samples"]
    d["min_samples"] = min(d["min_samples"], d["n_samples"])
    proposed, modes, requested, generation = sample_dataset(d, plan["fresh_data_seed"])
    excluded = np.union1d(vector_keys(dataset["a"]), vector_keys(dataset["contrast_a"]))
    if "exclusion_a" in dataset:
        excluded = np.union1d(excluded, vector_keys(dataset["exclusion_a"]))
    keep = np.flatnonzero(~np.isin(vector_keys(proposed), excluded))[:plan["fresh_sample_limit"]]
    a, modes, requested = proposed[keep], modes[keep], requested[keep]
    if not len(a):
        raise RuntimeError("No fresh sample vectors remain after excluding prior data")
    d["contrast_contexts_per_rule"] = plan["fresh_contexts_per_rule"]
    contrasts, memberships = contrast_set(d, plan["fresh_contrast_seed"])
    # Fresh contrast rows are also disjoint from the new sample evaluation set.
    contrast_keep = ~np.isin(vector_keys(contrasts), np.union1d(excluded, vector_keys(a)))
    if not contrast_keep.any():
        raise RuntimeError("No fresh contrast rows remain after exclusion")
    selected_rows = np.flatnonzero(contrast_keep)
    mapping = np.full(len(contrasts), -1, dtype=int)
    mapping[selected_rows] = np.arange(len(selected_rows))
    retained_members = memberships[contrast_keep[memberships[:, 0]]].copy()
    retained_members[:, 0] = mapping[retained_members[:, 0]]
    complete_groups = []
    for rule in range(8):
        complete_groups.append(sum(bool(np.all(contrast_keep[memberships[(memberships[:, 1] == rule) & (memberships[:, 2] == ctx), 0]]))
                                   for ctx in np.unique(memberships[memberships[:, 1] == rule, 2])))
    return ({"sample_a": a, "sample_g": targets(a), "sample_mode": modes,
             "sample_requested_rule": requested, "contrast_a": contrasts[contrast_keep],
             "contrast_g": targets(contrasts[contrast_keep]), "contrast_memberships": retained_members},
            {"proposal_generation": generation, "requested_fresh_samples": plan["fresh_sample_limit"],
             "sample": audit_samples(a, modes), "contrast": audit_samples(contrasts[contrast_keep]),
             "fully_unseen_contexts_per_rule": complete_groups,
             "note": "Both sets exclude all original sample/contrast vectors. Fresh contrasts also exclude fresh sample vectors. Exclusion conditions the sampling population and can change density, prevalence, and truth-table completeness. Prior test results informed this bounded tuning plan; this is an exploratory follow-up, not a pristine preregistered study."})


def prepare_candidate(base, config, root, width, epochs, version_prefix="stage1_tuning_v1"):
    c = deepcopy(config)
    c["version"] = f"{version_prefix}_width{width}"
    c["model"]["d_hidden"] = width
    c["training"]["epochs"] = epochs
    path = root / f"width{width}"
    if path.exists():
        if load_config(path / "config.yaml") != c:
            raise RuntimeError(f"Existing candidate config differs: {path}")
        manifest = json.loads((path / "data_manifest.json").read_text())
        if manifest != {"config_sha256": config_digest(c), "dataset_sha256": file_digest(base / "dataset.npz")}:
            raise RuntimeError("Candidate manifest differs from fixed baseline dataset/config")
        if file_digest(path / "dataset.npz") != manifest["dataset_sha256"]:
            raise RuntimeError("Candidate dataset was modified")
    else:
        path.mkdir()
        for name in ("dataset.npz", "data_audit.json", "SPEC.snapshot.md"):
            shutil.copyfile(base / name, path / name)
        (path / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        write_json(path / "data_manifest.json", {"config_sha256": config_digest(c),
                                                  "dataset_sha256": file_digest(path / "dataset.npz")})
    return c, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="configs/stage1_tuning_v1.yaml")
    p.add_argument("--run-dir", default="results/stage1-tuning-v1")
    args = p.parse_args()
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan = yaml.safe_load(Path(args.plan).read_text())
            if plan["selection"] != "worst_rule_balanced_accuracy_then_macro_then_mse":
                raise ValueError("Unknown candidate selection rule")
            saved_plan = root / "plan.json"
            base = Path(plan["base_run"])
            config = load_config(base / "config.yaml")
            manifest = json.loads((base / "data_manifest.json").read_text())
            if manifest != {"config_sha256": config_digest(config), "dataset_sha256": file_digest(base / "dataset.npz")}:
                raise RuntimeError("Baseline no longer matches its frozen data manifest")
            record = {"plan": plan, "baseline_manifest": manifest}
            if saved_plan.exists() and json.loads(saved_plan.read_text()) != record:
                raise RuntimeError("Plan or baseline changed; use a new tuning directory")
            write_json(saved_plan, record)
            if (root / "fresh_metrics.json").exists():
                print(f"Tuning already complete: {root}/fresh_metrics.json", flush=True)
                return
            seed_everything(config["seeds"]["model"], config["training"]["deterministic"])
            device = select_device("cuda")
            write_json(root / "environment.json", environment_record(device))
            validations = []
            for width in plan["hidden_widths"]:
                c, path = prepare_candidate(base, config, root, width, plan["epochs"], plan["version"])
                with np.load(path / "dataset.npz", allow_pickle=False) as dataset:
                    print(f"CANDIDATE width={width}; no test/contrast selection", flush=True)
                    train(c, dataset, path, device)
                    checkpoint = load_checkpoint(path / "best.pt")
                    verify_checkpoint(checkpoint, c, path / "dataset.npz")
                    model = model_from_config(c).to(device)
                    model.load_state_dict(checkpoint["model"])
                    idx = dataset["validation"]
                    out = predict(model, dataset["a"][idx], dataset["P"], device, c["evaluation"]["batch_size"])["output"]
                    metrics = prediction_metrics(out, dataset["g"][idx], dataset["Q"], c["evaluation"]["classification_threshold"])
                    row = {"width": width, "best_epoch": checkpoint["epoch"], "run": str(path), "metrics": metrics}
                    validations.append(row)
                    write_json(path / "validation_metrics.json", row)
                    write_json(root / "validation_comparison.json", validations)
                    print(f"VALIDATION width={width} ranking={selection_key(metrics)}", flush=True)
            selected = max(validations, key=lambda row: selection_key(row["metrics"]))
            # Persist the decision before constructing or evaluating fresh holdouts.
            write_json(root / "selection.json", selected)
            if "validation_gate" in plan:
                gate = plan["validation_gate"]
                passed = all(row["balanced_accuracy"] is not None and row["balanced_accuracy"] >= gate["min_balanced_accuracy"]
                             and row["recall"] is not None and row["recall"] >= gate["min_recall"]
                             for row in selected["metrics"]["per_rule"])
                write_json(root / "validation_gate.json", {"passed": passed, "thresholds": gate,
                           "note": "Operational tuning target, not automatic scientific approval."})
                if not passed:
                    print("Validation target not met. Retaining all failures; fresh holdouts remain unevaluated.", flush=True)
                    return
            with np.load(base / "dataset.npz", allow_pickle=False) as dataset:
                fresh, audit = fresh_sets(dataset, config, plan)
                np.savez_compressed(root / "fresh_dataset.npz", P=dataset["P"], Q=dataset["Q"], **fresh)
                write_json(root / "fresh_audit.json", audit)
                results = {}
                predictions = {}
                for name, path in [("baseline", Path(plan.get("comparison_run", str(base)))), ("selected", Path(selected["run"]))]:
                    c = load_config(path / "config.yaml")
                    checkpoint = load_checkpoint(path / "best.pt")
                    verify_checkpoint(checkpoint, c, path / "dataset.npz")
                    model = model_from_config(c).to(device)
                    model.load_state_dict(checkpoint["model"])
                    results[name] = {"width": c["model"]["d_hidden"], "epoch": checkpoint["epoch"],
                                     "checkpoint_sha256": file_digest(path / "best.pt")}
                    for population in ("sample", "contrast"):
                        out = predict(model, fresh[f"{population}_a"], dataset["P"], device, c["evaluation"]["batch_size"])["output"]
                        predictions[f"{name}_{population}"] = out
                        results[name][population] = prediction_metrics(out, fresh[f"{population}_g"], dataset["Q"], c["evaluation"]["classification_threshold"])
                np.savez_compressed(root / "fresh_predictions.npz", **predictions)
                write_json(root / "fresh_metrics.json", results)
                lines = ["# Stage 1 bounded tuning", "", f"Selected hidden width: {selected['width']}; epoch {selected['best_epoch']}.",
                         "Selection used validation only; fresh results were evaluated after the decision.", "",
                         "| Model | Fresh sample exact accuracy | Fresh contrast exact accuracy | Worst contrast balanced accuracy |", "|---|---:|---:|---:|"]
                for name, r in results.items():
                    scores = [x["balanced_accuracy"] for x in r["contrast"]["per_rule"]]
                    worst = min(v for v in scores if v is not None)
                    lines.append(f"| {name} | {r['sample']['exact_vector_accuracy']:.4f} | {r['contrast']['exact_vector_accuracy']:.4f} | {worst:.4f} |")
                lines.extend(["", "Single-seed exploratory follow-up. All candidates retained; see the frozen plan/config for distribution and optimization changes.",
                              "Fresh sets are conditioned on disjointness; their distribution differs from the original test set.",
                              "Higher accuracy does not establish learned mechanisms. No transcoder or circuit analysis was run."])
                (root / "report.md").write_text("\n".join(lines) + "\n")
                print("\n".join(lines), flush=True)
        except BaseException:
            (root / "failure.log").write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
