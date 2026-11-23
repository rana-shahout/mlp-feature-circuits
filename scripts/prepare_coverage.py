#!/usr/bin/env python3
"""Prepare documented Stage 1 coverage extension; no model input/label changes."""

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.data import ANTECEDENTS, audit_samples, targets, vector_keys
from src.runtime import load_config, validate_config, write_json
from src.train_mlp import config_digest, file_digest


def matched_rows(contexts_per_rule, seed, settings, excluded_keys, forbidden_contexts=()):
    """Random non-antecedent contexts, full local tables, then explicit exclusion.

    Context identities are (rule index, binary context key). Duplicate vectors
    and any reserved vector are excluded; retain a row-to-context mapping.
    """
    rng = np.random.default_rng(seed)
    forbidden = set(map(tuple, forbidden_contexts))
    contexts, seen, rows, memberships = [], set(map(int, excluded_keys)), [], []
    for rule, ids in enumerate(ANTECEDENTS):
        available = np.setdiff1d(np.arange(32), ids)
        used, attempts = set(), 0
        while len(used) < contexts_per_rule:
            attempts += 1
            if attempts > contexts_per_rule * 100:
                raise RuntimeError("Coverage context generation exhausted its bounded budget")
            k = int(rng.choice(settings["context_active_counts"], p=settings["context_probabilities"]))
            base = np.zeros(32, dtype=np.uint8)
            base[rng.choice(available, k, replace=False)] = 1
            key = int(vector_keys(base))
            if key in used or (rule, key) in forbidden:
                continue
            used.add(key)
            context_idx = len(contexts)
            contexts.append((rule, key))
            for pattern, bits in enumerate(product((0, 1), repeat=len(ids))):
                row = base.copy()
                row[list(ids)] = bits
                row_key = int(vector_keys(row))
                if row_key in seen:
                    continue
                seen.add(row_key)
                memberships.append((len(rows), context_idx, pattern))
                rows.append(row)
    if not rows:
        raise RuntimeError("Coverage extension yielded no unseen vectors")
    return np.asarray(rows, dtype=np.uint8), np.asarray(contexts, dtype=np.uint64), np.asarray(memberships, dtype=np.int64)


def prepare(plan):
    base = Path(plan["original_run"])
    output = Path(plan["base_run"])
    old_config = load_config(base / "config.yaml")
    expected = {"plan": plan, "original_dataset_sha256": file_digest(base / "dataset.npz"),
                "previous_holdouts_sha256": file_digest(plan["previous_holdouts"])}
    if output.exists():
        if json.loads((output / "preparation.json").read_text()) != expected:
            raise RuntimeError("Coverage plan or source data changed; use a new version")
        frozen = json.loads((output / "data_manifest.json").read_text())
        if frozen != {"config_sha256": config_digest(load_config(output / "config.yaml")),
                      "dataset_sha256": file_digest(output / "dataset.npz")}:
            raise RuntimeError("Existing prepared coverage dataset/config changed")
        print(f"Using already prepared coverage dataset: {output}", flush=True)
        return
    with np.load(base / "dataset.npz", allow_pickle=False) as old, np.load(plan["previous_holdouts"], allow_pickle=False) as prior:
        data = {key: old[key] for key in old.files}
        exclusion_a = np.concatenate([prior["sample_a"], prior["contrast_a"]])
        excluded = vector_keys(np.concatenate([old["a"], old["contrast_a"], exclusion_a]))
        settings = plan["coverage"]
        train_a, train_contexts, train_memberships = matched_rows(
            settings["train_contexts_per_rule"], settings["train_seed"], settings, excluded)
        validation_a, validation_contexts, validation_memberships = matched_rows(
            settings["validation_contexts_per_rule"], settings["validation_seed"], settings,
            np.concatenate([excluded, vector_keys(train_a)]), train_contexts)
        n, n_train = len(old["a"]), len(train_a)
        data["a"] = np.concatenate([old["a"], train_a, validation_a])
        data["g"] = targets(data["a"])
        data["train"] = np.concatenate([old["train"], np.arange(n, n + n_train)])
        data["validation"] = np.concatenate([old["validation"], np.arange(n + n_train, len(data["a"]))])
        data["mode"] = np.concatenate([old["mode"], np.full(n_train, 3, dtype=np.int8), np.full(len(validation_a), 4, dtype=np.int8)])
        data["requested_rule"] = np.concatenate([old["requested_rule"], np.full(n_train + len(validation_a), -1, dtype=np.int8)])
        data.update(exclusion_a=exclusion_a, train_contexts=train_contexts,
                    validation_contexts=validation_contexts, train_coverage_memberships=train_memberships,
                    validation_coverage_memberships=validation_memberships)
        if len(np.unique(vector_keys(data["a"]))) != len(data["a"]):
            raise AssertionError("Coverage data contains duplicate vectors")
        c = deepcopy(old_config)
        c["version"] = plan["version"] + "_prepared"
        c["training"].update(plan["training_changes"])
        c["training"]["epochs"] = plan["epochs"]
        c["coverage"] = settings
        c["data"]["n_samples"] = len(data["a"])
        c["data"]["split_fractions"] = [len(data[k]) / len(data["a"]) for k in ("train", "validation", "test")]
        validate_config(c)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "dataset.npz", **data)
        (output / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        shutil.copyfile(ROOT / "SPEC.md", output / "SPEC.snapshot.md")
        write_json(output / "data_audit.json", {
            "train": audit_samples(data["a"][data["train"]]),
            "validation": audit_samples(data["a"][data["validation"]]),
            "added_train": audit_samples(train_a), "added_validation": audit_samples(validation_a),
            "note": "Original train/validation/test preserved. Added training/validation vectors exclude all original samples/contrasts and previous fresh holdouts. Context identities are disjoint by rule between added train and validation. Row exclusion/deduplication may leave incomplete local tables. New training density support is deliberately broadened to 2..10. Extra mode codes: 3 coverage-train, 4 coverage-validation."})
        write_json(output / "data_manifest.json", {"config_sha256": config_digest(c), "dataset_sha256": file_digest(output / "dataset.npz")})
        write_json(output / "preparation.json", expected)
        print(f"Prepared coverage extension: +{len(train_a)} train, +{len(validation_a)} validation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/stage1_coverage_v2.yaml")
    args = parser.parse_args()
    prepare(yaml.safe_load(Path(args.plan).read_text()))
