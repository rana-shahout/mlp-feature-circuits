#!/usr/bin/env python3
"""Read-only teacher diagnostics and exact original-neuron finite interactions.

No training entry point, optimizer, parameter edits, or transcoder. Executed
only by the user. All source hashes are checked again before completion.
"""

import argparse
import csv
from datetime import datetime, timezone
from itertools import combinations, product
import json
from pathlib import Path
import shutil
import sys
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.coordinates import original_coordinates, semantic_forward
from src.data import ANTECEDENTS, RULE_IDS, targets, vector_keys
from src.diagnostics import diagnostic_metrics, distractor_groups, grouped_rows, make_contexts, observational_tables
from src.evaluation import predict
from src.interactions import decompose_corners
from src.runtime import load_checkpoint, load_config, seed_everything, select_device, write_json
from src.train_mlp import environment_record, file_digest, model_from_config, verify_checkpoint


def write_rows(path, rows):
    with open(path, "w", newline="") as handle:
        writer = None
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
        if writer is None:
            handle.write("# No rows\n")


def finite_corners(contexts, terms):
    if not terms or len(terms[0]) not in (2, 3):
        raise ValueError("Require a nonempty pair or triple batch")
    order = len(terms[0])
    if any(len(term) != order or len(set(term)) != order or min(term) < 0 or max(term) >= 32 for term in terms):
        raise ValueError("All terms must have the same order and valid distinct indices")
    bits = np.asarray(list(product((0, 1), repeat=order)), dtype=np.uint8)
    a = np.broadcast_to(contexts[None, :, None, :], (len(terms), len(contexts), len(bits), 32)).copy()
    for i, term in enumerate(terms):
        for column, bit in enumerate(term):
            a[i, :, :, bit] = bits[None, :, column]
    signs = np.asarray([(-1) ** (order - int(row.sum())) for row in bits], dtype=np.float64)
    return a, signs


def populations(data, fresh):
    a, g = data["a"], data["g"]
    result = {name: (a[data[name]], g[data[name]]) for name in ("train", "validation", "test")}
    if "mode" in data:
        for name, added in (("train", 3), ("validation", 4)):
            idx = data[name]
            for label, mask in (("original", data["mode"][idx] != added), ("coverage", data["mode"][idx] == added)):
                result[f"{name}_{label}"] = (a[idx[mask]], g[idx[mask]])
    result["fresh_sample"] = fresh["sample_a"], fresh["sample_g"]
    result["fresh_contrast"] = fresh["contrast_a"], fresh["contrast_g"]
    return result


def known_flags(data, fresh):
    flags = {}
    groups = [(1, data["a"][data["train"]]), (2, data["a"][data["validation"]]),
              (4, data["a"][data["test"]]), (8, fresh["sample_a"]),
              (16, fresh["contrast_a"]), (32, data["contrast_a"])]
    if "exclusion_a" in data:
        groups.append((64, data["exclusion_a"]))
    for flag, inputs in groups:
        for key in vector_keys(inputs):
            flags[int(key)] = flags.get(int(key), 0) | flag
    return flags


def audit_loaded(data, fresh):
    a = data["a"]
    if len(np.unique(vector_keys(a))) != len(a):
        raise ValueError("Source dataset has duplicate semantic vectors")
    idx = np.concatenate([data[k] for k in ("train", "validation", "test")])
    if not np.array_equal(np.sort(idx), np.arange(len(a))):
        raise ValueError("Source split must be a disjoint complete partition")
    np.testing.assert_array_equal(data["g"], targets(a))
    np.testing.assert_allclose(np.linalg.norm(data["P"], axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(data["Q"].T @ data["Q"], np.eye(8), atol=1e-6)
    for name in ("sample", "contrast"):
        np.testing.assert_array_equal(fresh[f"{name}_g"], targets(fresh[f"{name}_a"]))
        if np.isin(vector_keys(fresh[f"{name}_a"]), vector_keys(a)).any():
            raise ValueError("Saved fresh evaluation overlaps model dataset")
        if len(np.unique(vector_keys(fresh[f"{name}_a"]))) != len(fresh[f"{name}_a"]):
            raise ValueError("Saved fresh population contains duplicate vectors")
    if np.isin(vector_keys(fresh["sample_a"]), vector_keys(fresh["contrast_a"])).any():
        raise ValueError("Fresh sample and contrast sets overlap")
    np.testing.assert_array_equal(data["P"], fresh["P"])
    np.testing.assert_array_equal(data["Q"], fresh["Q"])


def evaluate_populations(model, data, fresh, config, settings, device, output):
    result = {}
    threshold = config["evaluation"]["classification_threshold"]
    errors = open(output / "three_way_and_errors.csv", "w", newline="")
    fields = ["population", "row", "kind", "target", "score", "pattern_8_9_10", "total_active", "distractor_count", "active_distractors", "other_semantic_active"]
    writer = csv.DictWriter(errors, fieldnames=fields)
    writer.writeheader()
    try:
        for name, (a, g) in populations(data, fresh).items():
            if not len(a):
                result[name] = {"n": 0}
                continue
            prediction = predict(model, a, data["P"], device, settings["batch_size"])["output"]
            scores = prediction.astype(np.float64) @ data["Q"].astype(np.float64)
            result[name] = diagnostic_metrics(prediction, g, data["Q"], threshold)
            # Every row retained for cross-seed agreement and reproduction.
            np.savez_compressed(output / f"predictions_{name}.npz", output=prediction, semantic=scores)
            def groups():
                for group, code, label in distractor_groups(a):
                    yield from grouped_rows(a, scores, g, threshold, code, group, label)
            write_rows(output / f"groups_{name}.csv", groups())
            write_rows(output / f"observational_truth_tables_{name}.csv", observational_tables(a, scores, g, threshold))
            # Three-way AND tables include all 8 antecedent patterns, densities,
            # and nuisance masks, not just its false negatives.
            code = 4*a[:, 8] + 2*a[:, 9] + a[:, 10]
            rows = grouped_rows(a, scores, g, threshold, code, "and3_pattern", lambda v: f"{v:03b}")
            write_rows(output / f"and3_patterns_{name}.csv", (r for r in rows if r["rule"] == RULE_IDS[4]))
            mask = (scores[:, 4] >= threshold) != g[:, 4].astype(bool)
            for i in np.flatnonzero(mask):
                writer.writerow({"population": name, "row": int(i), "kind": "FN" if g[i, 4] else "FP",
                                 "target": int(g[i, 4]), "score": float(scores[i, 4]),
                                 "pattern_8_9_10": "".join(map(str, a[i, [8, 9, 10]])),
                                 "total_active": int(a[i].sum()), "distractor_count": int(a[i, 18:].sum()),
                                 "active_distractors": " ".join(map(str, np.flatnonzero(a[i, 18:])+18)),
                                 "other_semantic_active": " ".join(map(str, [v for v in np.flatnonzero(a[i, :18]) if v not in (8, 9, 10)]))})
            print(f"{output.name}/{name}: n={len(a)}, exact={result[name]['exact_vector_accuracy']:.5f}", flush=True)
    finally:
        errors.close()
    write_json(output / "metrics.json", result)
    return result


def controlled_tables(model, data, settings, contexts, names, flags, output, device, threshold):
    def rows():
        for rule, ids in enumerate(ANTECEDENTS):
            patterns = list(product((0, 1), repeat=len(ids)))
            a = np.broadcast_to(contexts[:, None, :], (len(contexts), len(patterns), 32)).copy()
            for k, bits in enumerate(patterns):
                for bit, value in zip(ids, bits):
                    a[:, k, bit] = value
            flat = a.reshape(-1, 32)
            out = predict(model, flat, data["P"], device, settings["batch_size"])["output"]
            scores = (out.astype(np.float64) @ data["Q"].astype(np.float64)).reshape(len(contexts), len(patterns), 8)
            truth = targets(a)
            keys = vector_keys(flat).reshape(len(contexts), len(patterns))
            for c, name in enumerate(names):
                for k, bits in enumerate(patterns):
                    for j in range(8):
                        yield {"designed_rule": RULE_IDS[rule], "output_rule": RULE_IDS[j], "context": name,
                               "antecedents": ",".join(map(str, ids)), "pattern": "".join(map(str, bits)),
                               "input_key": int(keys[c, k]), "population_flags": flags.get(int(keys[c, k]), 0),
                               "total_active": int(a[c, k].sum()), "active_distractors": " ".join(map(str, np.flatnonzero(a[c, k, 18:])+18)),
                               "target": int(truth[c, k, j]), "score": float(scores[c, k, j]),
                               "predicted": int(scores[c, k, j] >= threshold), "constructed_counterfactual": True}
    write_rows(output / "controlled_truth_tables.csv", rows())
    def nuisance_rows():
        for bit in range(18, 32):
            a = np.repeat(contexts[:, None, :], 2, axis=1)
            a[:, 0, bit], a[:, 1, bit] = 0, 1
            flat = a.reshape(-1, 32)
            prediction = predict(model, flat, data["P"], device, settings["batch_size"])["output"]
            scores = (prediction.astype(np.float64) @ data["Q"].astype(np.float64)).reshape(len(contexts), 2, 8)
            truth = targets(a)
            np.testing.assert_array_equal(truth[:, 0], truth[:, 1])
            keys = vector_keys(a)
            for c, context in enumerate(names):
                for j in range(8):
                    yield {"distractor": bit, "context": context, "output_rule": RULE_IDS[j],
                           "score_off": float(scores[c,0,j]), "score_on": float(scores[c,1,j]),
                           "score_difference": float(scores[c,1,j]-scores[c,0,j]), "target_difference": 0,
                           "changes_classification": bool((scores[c,1,j]>=threshold)!=(scores[c,0,j]>=threshold)),
                           "off_population_flags": flags.get(int(keys[c,0]),0), "on_population_flags": flags.get(int(keys[c,1]),0),
                           "constructed_counterfactual": True}
    write_rows(output / "controlled_distractor_effects.csv", nuisance_rows())


@torch.no_grad()
def interaction_archive(model, data, settings, contexts, names, flags, output, device):
    # Promote the stored float32 parameters, never refit or alter checkpoint files.
    model.to(dtype=torch.float64)
    p, q = [torch.as_tensor(data[key], dtype=torch.float64, device=device) for key in ("P", "Q")]
    coord = original_coordinates(model, p, q)
    np.savez_compressed(output / "coordinates.npz", **{key: value.cpu().numpy() for key, value in coord.items()})
    base = torch.as_tensor(contexts, dtype=torch.float64, device=device)
    direct = model(base @ p.T) @ q
    torch.testing.assert_close(direct, semantic_forward(base, coord), atol=settings["interaction_atol"], rtol=settings["interaction_rtol"])
    nc, nh = len(contexts), model.W1.out_features
    global_positive, global_negative = np.zeros((8, nh)), np.zeros((8, nh))
    positive_count, negative_count = np.zeros((8, nh), dtype=np.int64), np.zeros((8, nh), dtype=np.int64)
    summaries, strongest = {}, []
    for order in (2, 3):
        terms = list(combinations(range(32), order))
        folder = output / f"order{order}"
        folder.mkdir()
        np.save(folder / "terms.npy", np.asarray(terms, dtype=np.int16))
        hidden_file = np.lib.format.open_memmap(folder / "hidden_delta.npy", mode="w+", dtype="float64", shape=(len(terms), nc, nh))
        full = (np.lib.format.open_memmap(folder / "contributions.npy", mode="w+", dtype="float64", shape=(len(terms), nc, 8, nh))
                if settings["store_explicit_contributions"] else None)
        population_file = np.lib.format.open_memmap(folder / "corner_population_flags.npy", mode="w+", dtype="uint8", shape=(len(terms), nc, 2**order))
        totals, targets_all = np.zeros((len(terms), nc, 8)), np.zeros((len(terms), nc, 8))
        counters = {name: {"n": np.zeros(8, dtype=int), "sum_absolute_error": np.zeros(8)}
                    for name in ("target_nonzero", "target_zero", "contains_distractor")}
        with open(folder / "signed_sums.csv", "w", newline="") as handle:
            writer = None
            for start in range(0, len(terms), settings["interaction_batch_size"]):
                batch_terms = terms[start:start+settings["interaction_batch_size"]]
                a, signs = finite_corners(contexts, batch_terms)
                tensor = torch.as_tensor(a, dtype=torch.float64, device=device)
                sign_tensor = torch.as_tensor(signs, device=device)
                result = decompose_corners(tensor, sign_tensor, coord)
                # Independent original-input forward, not just the coordinate expression.
                original_x = (tensor @ p.T).reshape(-1, p.shape[0])
                original = (model(original_x) @ q).reshape(*tensor.shape[:-1], 8)
                forward_delta = (original * sign_tensor[None, None, :, None]).sum(-2)
                torch.testing.assert_close(result["value"], forward_delta, atol=settings["interaction_atol"], rtol=settings["interaction_rtol"])
                torch.testing.assert_close(result["direct"], result["value"], atol=settings["interaction_atol"], rtol=settings["interaction_rtol"])
                torch.testing.assert_close(result["positive_sum"] + result["negative_sum"], result["value"], atol=settings["interaction_atol"], rtol=settings["interaction_rtol"])
                arrays = {key: value.cpu().numpy() for key, value in result.items() if key not in ("corners", "signs")}
                arrays["original_forward"] = forward_delta.cpu().numpy()
                arrays["original_forward_residual"] = arrays["value"] - arrays["original_forward"]
                truth = (targets(a).astype(np.float64) * signs[None, None, :, None]).sum(-2)
                stop = start + len(batch_terms)
                hidden_file[start:stop] = arrays["hidden_delta"]
                if full is not None:
                    full[start:stop] = arrays["contributions"]
                totals[start:stop], targets_all[start:stop] = arrays["value"], truth
                keys = vector_keys(a)
                batch_flags = np.asarray([flags.get(int(key), 0) for key in keys.reshape(-1)], dtype=np.uint8).reshape(keys.shape)
                population_file[start:stop] = batch_flags
                contributions = arrays["contributions"]
                global_positive += np.maximum(contributions, 0).sum((0, 1))
                global_negative += np.minimum(contributions, 0).sum((0, 1))
                positive_count += np.count_nonzero(contributions > 0, axis=(0, 1))
                negative_count += np.count_nonzero(contributions < 0, axis=(0, 1))
                err = np.abs(arrays["value"] - truth)
                for name in counters:
                    mask = (truth != 0 if name == "target_nonzero" else truth == 0)
                    if name == "contains_distractor":
                        mask = np.broadcast_to(np.asarray([any(i >= 18 for i in ids) for ids in batch_terms])[:, None, None], truth.shape)
                    counters[name]["n"] += mask.sum((0, 1))
                    counters[name]["sum_absolute_error"] += (err * mask).sum((0, 1))
                for t, ids in enumerate(batch_terms):
                    for c, context in enumerate(names):
                        for j in range(8):
                            vector = contributions[t, c, j]
                            pos_r, neg_r = int(vector.argmax()), int(vector.argmin())
                            row = {"order": order, "term_index": start+t, "inputs": ",".join(map(str, ids)), "context": context,
                                   "output_rule": RULE_IDS[j], "target_interaction": float(truth[t, c, j]),
                                   "constructed_counterfactual": True,
                                   "min_corner_active_count": int(a[t,c].sum(-1).min()), "max_corner_active_count": int(a[t,c].sum(-1).max()),
                                   "train_overlap_fraction": float(((batch_flags[t,c]&1)!=0).mean()),
                                   "original_test_overlap_fraction": float(((batch_flags[t,c]&4)!=0).mean()),
                                   "unseen_by_saved_populations_fraction": float((batch_flags[t,c]==0).mean()),
                                   **{key: float(arrays[key][t, c, j]) for key in ("value", "direct", "original_forward", "original_forward_residual", "positive_sum", "negative_sum", "absolute_sum", "cancellation_fraction")},
                                   "cancelled_absolute_mass": float(2*min(arrays['positive_sum'][t,c,j], -arrays['negative_sum'][t,c,j])),
                                   "absolute_error": float(err[t, c, j]),
                                   "strongest_positive_neuron": pos_r if vector[pos_r] > 0 else None,
                                   "strongest_positive_contribution": max(float(vector[pos_r]), 0),
                                   "strongest_negative_neuron": neg_r if vector[neg_r] < 0 else None,
                                   "strongest_negative_contribution": min(float(vector[neg_r]), 0)}
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row)
                            # Display selection only. All neuron values remain in the archive.
                            if len(strongest) < 20 or abs(row["value"]) > min(abs(r["value"]) for r in strongest):
                                strongest.append(row.copy())
                                strongest = sorted(strongest, key=lambda r: abs(r["value"]), reverse=True)[:20]
                print(f"{output.name}/interactions order={order}: {stop}/{len(terms)}", flush=True)
        hidden_file.flush(); population_file.flush()
        if full is not None:
            full.flush()
        del full, hidden_file, population_file
        np.savez_compressed(folder / "interactions.npz", value=totals, ground_truth=targets_all)
        summaries[str(order)] = {name: [{"rule": RULE_IDS[j], "n": int(values["n"][j]),
                  "mean_absolute_error": float(values["sum_absolute_error"][j]/values["n"][j]) if values["n"][j] else None}
                 for j in range(8)] for name, values in counters.items()}
    write_rows(output / "neuron_totals.csv", ({"output_rule": RULE_IDS[j], "neuron": r,
                "positive_mass": float(global_positive[j,r]), "negative_mass": float(global_negative[j,r]),
                "signed_sum": float(global_positive[j,r]+global_negative[j,r]),
                "absolute_mass": float(global_positive[j,r]-global_negative[j,r]),
                "positive_case_count": int(positive_count[j,r]), "negative_case_count": int(negative_count[j,r]),
                "changes_contribution_sign_across_cases": bool(positive_count[j,r] and negative_count[j,r])}
                for j in range(8) for r in range(nh)))
    write_rows(output / "strongest_interactions_display.csv", strongest)
    write_json(output / "interaction_metrics.json", summaries)
    return summaries, strongest


def validation_rank(metrics):
    rows = metrics["validation"]["per_rule"]
    values = [r["balanced_accuracy"] for r in rows]
    if any(v is None for v in values):
        raise ValueError("Cannot recommend a seed with undefined validation metrics")
    return min(values), sum(values)/len(values), -metrics["validation"]["output_mse"]


def cross_seed_agreement(out, results, settings):
    """Matched-row variation and shared failures, without aligning hidden neurons."""
    seeds = list(results)
    summary = {}
    first = Path(next(iter(settings["models"].values())))
    with np.load(first / "dataset.npz", allow_pickle=False) as data, np.load(settings["holdout"], allow_pickle=False) as fresh:
        for name, (a, g) in populations(data, fresh).items():
            if not len(a):
                continue
            scores = []
            for seed in seeds:
                with np.load(out / f"seed{seed}" / f"predictions_{name}.npz", allow_pickle=False) as cache:
                    scores.append(cache["semantic"])
            scores = np.stack(scores)
            threshold = np.asarray([results[s]["metrics"][name]["classification_threshold"] for s in seeds])[:, None, None]
            wrong = (scores >= threshold) != g.astype(bool)[None, :, :]
            count = wrong.sum(0)
            summary[name] = {"g4_wrong_seed_histogram": np.bincount(count[:, 4], minlength=len(seeds)+1).tolist(),
                             "all_seeds_wrong_per_rule": (count == len(seeds)).sum(0).tolist(),
                             "any_seed_wrong_per_rule": (count > 0).sum(0).tolist()}
            np.savez_compressed(out / f"cross_seed_{name}.npz", wrong_seed_count=count,
                                semantic_mean=scores.mean(0), semantic_std=scores.std(0, ddof=1), input_key=vector_keys(a))
            write_rows(out / f"shared_g4_errors_{name}.csv", ({"row": int(i), "target": int(g[i,4]),
                     "wrong_seed_count": int(count[i,4]), "wrong_seeds": ",".join(str(seeds[k]) for k in range(len(seeds)) if wrong[k,i,4]),
                     "mean_score": float(scores[:,i,4].mean()), "distractor_count": int(a[i,18:].sum()),
                     "active_distractors": " ".join(map(str, np.flatnonzero(a[i,18:])+18))}
                    for i in np.flatnonzero(count[:,4])))
    write_json(out / "cross_seed_agreement.json", summary)
    return summary


def build_report(out, results, settings, agreement):
    seeds = list(results)
    candidate = max(seeds, key=lambda s: validation_rank(results[s]["metrics"]))
    write_rows(out / "validation_seed_ranking.csv", ({"seed": seed,
                "worst_rule_balanced_accuracy": validation_rank(results[seed]["metrics"])[0],
                "macro_balanced_accuracy": validation_rank(results[seed]["metrics"])[1],
                "output_mse": results[seed]["metrics"]["validation"]["output_mse"]}
                for seed in sorted(seeds, key=lambda s: validation_rank(results[s]["metrics"]), reverse=True)))
    lines = ["# Trained MLP diagnostics", "", "Evaluation only: no weights were updated and no transcoder was trained.", "",
             "## Definitions and populations", "", "Exact-vector accuracy requires all eight decoded outputs to be correct. Scores are Q.T F(Pa), classified at each run's saved threshold; they are unbounded regression scores, not probabilities.",
             "FPR=FP/(FP+TN); FNR=FN/(FN+TP); precision=TP/(TP+FP); recall=TP/(TP+FN). Undefined denominators are null. Prediction error includes semantic MSE, MAE, signed bias, and output-space MSE/residual.",
             "Original test and the saved fresh sets have already been inspected during prior exploratory work. They are diagnostic populations, not newly blinded tests. Combined train/validation include coverage additions; original/coverage subsets are separate.", "",
             "## Across seeds", "", "| Seed | Train exact | Validation exact | Original test exact | Saved fresh sample exact | Saved fresh contrast exact | g4 contrast recall |", "|---|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        m = results[seed]["metrics"]
        vals = [m[p]["exact_vector_accuracy"] for p in ("train", "validation", "test", "fresh_sample", "fresh_contrast")]
        lines.append(f"| {seed} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {m['fresh_contrast']['per_rule'][4]['recall']:.4f} |")
    lines += ["", "## Which relationships are reliable on these measured populations?", "",
              "| Rule | Minimum fresh-contrast balanced accuracy | Minimum recall | Maximum FPR | Maximum nonzero pair/triple interaction MAE | Review |", "|---|---:|---:|---:|---:|---|"]
    review = settings["review"]
    behavioral_failures = []
    for j, rule in enumerate(RULE_IDS):
        rows = [results[s]["metrics"]["fresh_contrast"]["per_rule"][j] for s in seeds]
        ba, recall, fpr = min(r["balanced_accuracy"] for r in rows), min(r["recall"] for r in rows), max(r["false_positive_rate"] for r in rows)
        errors = [results[s]["interactions"][order]["target_nonzero"][j]["mean_absolute_error"] for s in seeds for order in ("2", "3")]
        errors = [v for v in errors if v is not None]
        interaction_error = max(errors) if errors else None
        ok = ba >= review["min_balanced_accuracy"] and recall >= review["min_recall"]
        if not ok:
            behavioral_failures.append(rule)
        zero_errors = [results[s]["interactions"][order]["target_zero"][j]["mean_absolute_error"] for s in seeds for order in ("2", "3")]
        zero_errors = [v for v in zero_errors if v is not None]
        probe_ok = interaction_error is not None and interaction_error <= review["max_mean_interaction_absolute_error"] and (not zero_errors or max(zero_errors) <= review["max_mean_interaction_absolute_error"])
        flag = "within configured criteria" if ok and probe_ok else "inspect failures / insufficient probes"
        lines.append(f"| {rule} | {ba:.4f} | {recall:.4f} | {fpr:.4f} | {interaction_error if interaction_error is not None else 'no nonzero probes'} | {flag} |")
    lines += ["", "These are configurable review criteria on finite probes, not proof of global correctness. Zero-target and distractor-involving interactions have separate error metrics; an average dominated by zero targets cannot establish rule recovery.", "",
              "## Distractors and three-way AND", "",
              "| Seed | Train positives | Train FNR | Validation positives | Validation FNR | Original test positives | Test FNR | Contrast positives | Contrast FNR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        fields = []
        for population in ("train", "validation", "test", "fresh_contrast"):
            row = results[seed]["metrics"][population]["per_rule"][4]
            fields.extend([str(row["n_positive"]), str(row["false_negative_rate"])])
        lines.append(f"| {seed} | " + " | ".join(fields) + " |")
    lines += ["", f"On the saved fresh contrasts, g4 errors shared by all five seeds: {agreement['fresh_contrast']['all_seeds_wrong_per_rule'][4]}; errors in at least one seed: {agreement['fresh_contrast']['any_seed_wrong_per_rule'][4]}. Matching rows are in shared_g4_errors_fresh_contrast.csv.", "",
              "See groups_<population>.csv for exact distractor sets, individual presence/absence, presence conditional on distractor count, and total density. Low-support groups and missing classes must not be treated as causal findings. Cross-seed differences use identical rows and contexts.",
              "The following are the largest count-adjusted distractor associations for g4 on the saved fresh contrast set (positive-example FN-rate difference, present minus absent; support shown). They remain observational and may reflect other semantic inputs.", ""]
    # Data-driven observational findings from count-matched, positive-only rates.
    for seed in seeds:
        grouped = {}
        with open(out / f"seed{seed}" / "groups_fresh_contrast.csv") as handle:
            for row in csv.DictReader(handle):
                if row["rule"] == RULE_IDS[4] and row["group"].endswith("_presence_given_count"):
                    count, presence = [int(x.split("=")[1]) for x in row["value"].split(";")]
                    grouped.setdefault((row["group"], count), {})[presence] = row
        effects = []
        for (group, count), pair in grouped.items():
            if set(pair) == {0, 1} and all(int(pair[p]["n_positive"]) >= review["association_min_positive_support"] for p in (0, 1)):
                effects.append((abs(float(pair[1]["false_negative_rate"])-float(pair[0]["false_negative_rate"])), group, count, pair))
        for _, group, count, pair in sorted(effects, key=lambda item: item[0], reverse=True)[:3]:
            delta = float(pair[1]["false_negative_rate"])-float(pair[0]["false_negative_rate"])
            lines.append(f"- Seed {seed}, {group}, nuisance count {count}: FNR difference {delta:+.4f}; positive support absent={pair[0]['n_positive']}, present={pair[1]['n_positive']}.")
        if not effects:
            lines.append(f"- Seed {seed}: no count-matched group pair meets the configured positive-support minimum on both sides; insufficient support for this screen.")
    lines += ["", "All g4 FP/FN rows are in three_way_and_errors.csv; all eight observed antecedent patterns and fixed-context constructed tables are retained. The association screen is exploratory, has many comparisons, and supplies no causal or multiplicity-adjusted significance claim.", "",
              "Largest single-nuisance score changes for g4 in the fixed backgrounds (constructed input flips, not hidden-neuron ablations):"]
    for seed in seeds:
        with open(out / f"seed{seed}" / "controlled_distractor_effects.csv") as handle:
            flips = [row for row in csv.DictReader(handle) if row["output_rule"] == RULE_IDS[4]]
        for row in sorted(flips, key=lambda row: abs(float(row["score_difference"])), reverse=True)[:3]:
            lines.append(f"- Seed {seed}, nuisance {row['distractor']}, context {row['context']}: score change {float(row['score_difference']):+.6g}, while the Boolean target is unchanged.")
    lines += ["",
              "## Interaction-type errors", "", "| Order | Output | Nonzero-target mean absolute error across seeds | Zero-target mean absolute error across seeds |", "|---|---|---:|---:|"]
    for order in ("2", "3"):
        for j, rule in enumerate(RULE_IDS):
            values = {}
            for group in ("target_nonzero", "target_zero"):
                measured = [results[s]["interactions"][order][group][j]["mean_absolute_error"] for s in seeds]
                measured = [x for x in measured if x is not None]
                values[group] = float(np.mean(measured)) if measured else "no nonzero probes"
            lines.append(f"| {order} | {rule} | {values['target_nonzero']} | {values['target_zero']} |")
    lines += ["", "These are finite differences of continuous semantic scores, not thresholded Boolean predictions. Context-specific discrepancies are in order2/order3 signed_sums.csv, including zero ground-truth effects and negative witnesses.", "",
              "## Original neurons and exact nonlinear interactions", "",
              "C1=W1P and D2_eff=Q.T W2 preserve the original biases and gates. For each pair/triple and context, all 4/8 corners are evaluated. Every neuron's contribution is D2_eff[j,r] times its hidden-activation finite difference. Output bias cancels; neuron biases remain inside ReLU. Direct original-forward differences and full signed sums must agree numerically or the run fails.",
              "The table below displays the three strongest measured interactions per seed. Full signed positive/negative/cancelled sums and every neuron are retained, not truncated. Neuron IDs are seed-specific; this table does not align neurons across runs or establish a compact causal mechanism.", "",
              "| Seed | Inputs | Output | Context | Actual interaction | Target | Positive sum | Negative sum | Largest positive neuron | Largest negative neuron |", "|---|---|---|---|---:|---:|---:|---:|---|---|"]
    for seed in seeds:
        for r in results[seed]["strongest"][:3]:
            lines.append(f"| {seed} | {r['inputs']} | {r['output_rule']} | {r['context']} | {r['value']:.6g} | {r['target_interaction']:.6g} | {r['positive_sum']:.6g} | {r['negative_sum']:.6g} | {r['strongest_positive_neuron']} ({r['strongest_positive_contribution']:+.5g}) | {r['strongest_negative_neuron']} ({r['strongest_negative_contribution']:+.5g}) |")
    lines += ["", "## First-transcoder seed and whether more training is needed", "",
              f"Validation-only candidate: **seed {candidate}**, ranking by minimum per-rule balanced accuracy, then macro balanced accuracy, then lower mixed-output MSE. This avoids selecting a seed by test or contrast performance. All seeds remain reported.",
              "This deterministic point-estimate ranking is an operational choice, not evidence of statistically significant superiority. Inspect validation_seed_ranking.csv; small differences should not be overinterpreted.",
              f"Rules outside configured fresh-contrast behavioral review criteria in at least one seed: {', '.join(behavioral_failures) or 'none'}.",
              "The diagnostics distinguish evidence of remaining teacher errors from evidence that more optimization would fix them. Cross-sectional errors and exact decompositions alone cannot prove that additional training is necessary or sufficient. Persistent training errors suggest capacity/optimization limitations; larger held-out gaps suggest coverage/generalization limitations. Inspect train-vs-validation error tables and controlled contexts before changing the experiment.",
              "A transcoder can imitate a demonstrably imperfect teacher if the scientific endpoint is its actual function, with every teacher failure retained. Claims of accurate Boolean computation require accepting the measured per-rule limitations. **Human review is required before choosing retraining or a first transcoder experiment; neither is launched here.**",
              "Finite-difference identities establish an exact decomposition of this trained function. They do not establish necessity/sufficiency of selected neurons; those would require separately authorized original-neuron interventions."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(out / "recommendation.json", {"seed": candidate, "basis": "validation_only", "behavioral_review_failures": behavioral_failures,
                                              "action": "pending_human_review_no_training_authorized"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/diagnostics_v1.yaml")
    parser.add_argument("--output-dir", help="New directory only; existing paths are refused")
    parser.add_argument("--device", default=None, help="Default config: CUDA; explicit cpu for debugging only")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output_dir) if args.output_dir else Path("results") / ("diagnostics-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8])
    # Never allow a diagnostic output path to equal/contain a source checkpoint directory.
    source_dirs = [Path(path).resolve() for path in settings["models"].values()]
    if any(out.resolve() == path or out.resolve() in path.parents or path in out.resolve().parents for path in source_dirs):
        raise ValueError("Diagnostic output must be separate from every model directory")
    out.mkdir(parents=True, exist_ok=False)
    try:
        if settings["batch_size"] <= 0 or settings["interaction_batch_size"] <= 0:
            raise ValueError("Batch sizes must be positive")
        if not settings["contexts"]:
            raise ValueError("At least one fixed context is required")
        if not np.isfinite([settings["interaction_atol"], settings["interaction_rtol"]]).all() or settings["interaction_atol"] <= 0 or settings["interaction_rtol"] < 0:
            raise ValueError("Interaction tolerances must be finite, with atol>0 and rtol>=0")
        if not isinstance(settings["store_explicit_contributions"], bool):
            raise ValueError("store_explicit_contributions must be a Boolean")
        if len(settings["models"]) != 5:
            raise ValueError("This diagnostic protocol requires all five trained models")
        (out / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
        shutil.copyfile(ROOT / "SPEC.md", out / "SPEC.snapshot.md")
        source_paths = {Path(settings["holdout"])}
        for path in settings["models"].values():
            source_paths.update(Path(path)/name for name in ("config.yaml", "dataset.npz", "best.pt", "last.pt"))
        before = {str(p.resolve()): file_digest(p) for p in source_paths}
        write_json(out / "source_hashes_before.json", before)
        seed_everything(settings["context_seed"], True)
        device = select_device(args.device or settings.get("device", "cuda"))
        write_json(out / "environment.json", environment_record(device))
        contexts, names = make_contexts(settings["contexts"], settings["context_seed"])
        np.save(out / "contexts.npy", contexts)
        write_json(out / "context_names.json", names)
        results = {}
        reference_data_hash = None
        for seed, path in settings["models"].items():
            path = Path(path)
            config = load_config(path / "config.yaml")
            if config["seeds"]["model"] != int(seed):
                raise ValueError("Configured seed does not match saved model config")
            checkpoint = load_checkpoint(path / "best.pt")
            verify_checkpoint(checkpoint, config, path / "dataset.npz")
            data_hash = file_digest(path / "dataset.npz")
            if reference_data_hash is not None and data_hash != reference_data_hash:
                raise ValueError("Across-seed comparison requires identical dataset bytes")
            reference_data_hash = data_hash
            estimated = len(settings["models"]) * 5456 * len(contexts) * config["model"]["d_hidden"] * 8 * (9 if settings["store_explicit_contributions"] else 1)
            if not results and shutil.disk_usage(out).free < 1.1*estimated:
                raise RuntimeError(f"Insufficient free disk for full signed archive; estimated {estimated/1e9:.2f} GB. Use a larger output filesystem or explicitly configure lossless factorized storage.")
            folder = out / f"seed{seed}"
            folder.mkdir()
            write_json(folder / "checkpoint_metadata.json", {"seed": int(seed), "epoch": checkpoint["epoch"], "source": str(path),
                        "checkpoint_sha256": file_digest(path / "best.pt"), "dimensions": config["model"],
                        "threshold": config["evaluation"]["classification_threshold"], "config": config})
            model = model_from_config(config).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval().requires_grad_(False)
            with np.load(path / "dataset.npz", allow_pickle=False) as data, np.load(settings["holdout"], allow_pickle=False) as fresh:
                audit_loaded(data, fresh)
                if not results:
                    np.savez_compressed(out / "embeddings.npz", P=data["P"], Q=data["Q"])
                flags = known_flags(data, fresh)
                metrics = evaluate_populations(model, data, fresh, config, settings, device, folder)
                controlled_tables(model, data, settings, contexts, names, flags, folder, device, config["evaluation"]["classification_threshold"])
                interactions, strongest = interaction_archive(model, data, settings, contexts, names, flags, folder, device)
                results[int(seed)] = {"metrics": metrics, "interactions": interactions, "strongest": strongest}
        agreement = cross_seed_agreement(out, results, settings)
        build_report(out, results, settings, agreement)
        after = {str(p.resolve()): file_digest(p) for p in source_paths}
        write_json(out / "source_hashes_after.json", after)
        if before != after:
            raise RuntimeError("A source checkpoint/config/dataset changed during diagnostics")
        write_json(out / "complete.json", {"source_hashes_unchanged": True, "seeds": list(results)})
        print(f"Diagnostics complete: {out}/report.md", flush=True)
    except BaseException:
        (out / "failure.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
