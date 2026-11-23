#!/usr/bin/env python3
"""User-operated Stage 1 CLI. Never starts a transcoder or circuit analysis."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback
from uuid import uuid4

# Direct invocation from repository root, without requiring editable installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.data import create_dataset
from src.evaluation import evaluate_run
from src.runtime import (load_checkpoint, load_config, seed_everything, select_device,
                         validate_config, write_json)
from src.train_mlp import (config_digest, environment_record, file_digest,
                           model_from_config, train, verify_checkpoint)


def unique_name():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/sparse_superposition_v1.yaml")
        p.add_argument("--run-dir", help="New directory; must not already exist")
        p.add_argument("--results-dir", help="Override results root for automatically named runs")
        p.add_argument("--model-seed", type=int, help="Change initialization/shuffling only; P/Q/data stay fixed")
        if name == "train":
            p.add_argument("--device", default=None, help="Default: saved config's cuda. cpu requires explicit selection")
            p.add_argument("--resume", help="Existing prepared/interrupted run; uses its frozen config and dataset")
    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default=None)
    args = parser.parse_args()
    run_dir = None
    try:
        if args.command == "evaluate" or (args.command == "train" and args.resume):
            if args.command == "train" and (args.run_dir or args.results_dir or args.model_seed is not None or args.config != "configs/sparse_superposition_v1.yaml"):
                parser.error("--resume uses frozen config; do not combine with configuration/run overrides")
            run_dir = Path(args.run_dir if args.command == "evaluate" else args.resume)
            config = load_config(run_dir / "config.yaml")
            if not (run_dir / "dataset.npz").is_file() or not (run_dir / "data_audit.json").is_file():
                raise RuntimeError("Prepared dataset/audit missing; use a new run directory")
            manifest = json.loads((run_dir / "data_manifest.json").read_text())
            if manifest != {"config_sha256": config_digest(config),
                            "dataset_sha256": file_digest(run_dir / "dataset.npz")}:
                raise RuntimeError("Prepared config/dataset differs from frozen manifest; use a new run")
            if args.command == "train" and (run_dir / "last.pt").exists() and not (run_dir / "best.pt").exists():
                raise RuntimeError("Incomplete checkpoints: last.pt exists but best.pt is missing")
        else:
            config = load_config(args.config)
            if args.model_seed is not None:
                config["seeds"]["model"] = args.model_seed
                validate_config(config)
            if args.results_dir:
                config["results_dir"] = args.results_dir
            if args.command == "train" and args.device:
                config["device"] = args.device
            requested_dir = Path(args.run_dir) if args.run_dir else Path(config["results_dir"]) / unique_name()
            # Assign run_dir only after exclusive creation: never write into an existing run on failure.
            requested_dir.mkdir(parents=True, exist_ok=False)
            run_dir = requested_dir
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            shutil.copyfile(ROOT / "SPEC.md", run_dir / "SPEC.snapshot.md")
            print(f"Run directory: {run_dir.resolve()}", flush=True)
            if args.command == "train":
                # Fail before preparing data when CUDA is unavailable.
                seed_everything(config["seeds"]["model"], config["training"]["deterministic"])
                device = select_device(config["device"])
            audit = create_dataset(config, run_dir / "dataset.npz")
            write_json(run_dir / "data_audit.json", audit)
            write_json(run_dir / "data_manifest.json", {
                "config_sha256": config_digest(config),
                "dataset_sha256": file_digest(run_dir / "dataset.npz")})
        if args.command == "prepare":
            print(f"Data prepared: {run_dir}. Inspect data_audit.json before training.", flush=True)
            return
        if args.command == "evaluate" or args.resume:
            seed_everything(config["seeds"]["model"], config["training"]["deterministic"])
            device = select_device(args.device or config["device"])
        with np.load(run_dir / "dataset.npz", allow_pickle=False) as dataset:
            if args.command == "train":
                train(config, dataset, run_dir, device)
            else:
                checkpoint = load_checkpoint(run_dir / "best.pt")
                verify_checkpoint(checkpoint, config, run_dir / "dataset.npz")
                model = model_from_config(config).to(device)
                model.load_state_dict(checkpoint["model"])
                out = run_dir / ("evaluation-" + unique_name())
                metrics = evaluate_run(model, dataset, config, device, out, checkpoint["epoch"])
                write_json(out / "evaluation_manifest.json", {
                    "environment": environment_record(device),
                    "checkpoint_sha256": file_digest(run_dir / "best.pt"),
                    "config_sha256": config_digest(config),
                    "dataset_sha256": file_digest(run_dir / "dataset.npz")})
                print(f"Evaluation saved to {out}; test output MSE={metrics['partitions']['test']['output_mse']:.8g}", flush=True)
    except BaseException:
        if run_dir is not None and run_dir.is_dir():
            (run_dir / ("failure-" + unique_name() + ".log")).write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
