#!/usr/bin/env bash
# Run inside tmux on the GPU host after explicit user authorization.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_path=${1:-results/stage1-seed106}
if [[ ! -f "$run_path/data_manifest.json" ]]; then
    printf 'Missing prepared dataset manifest: %s\n' "$run_path" >&2
    exit 1
fi
mkdir -p logs
exec 9>"$run_path/.training.lock"
if ! flock -n 9; then
    printf 'Another verified training process holds the run lock: %s\n' "$run_path" >&2
    exit 1
fi
trap 'run_status=$?; printf "%s\n" "$run_status" > "$run_path/verified_train.exitcode"; printf "Verified training workflow exit status: %s\n" "$run_status"' EXIT
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'Verification started: %s\n' "$(date -Is)"
.venv/bin/python -c 'from src.runtime import select_device; select_device("cuda")' \
    2>&1 | tee -a logs/stage1-seed106-verification.log
.venv/bin/python -m pytest -q \
    2>&1 | tee -a logs/stage1-seed106-verification.log
printf 'Training started: %s\n' "$(date -Is)"
.venv/bin/python scripts/run_sparse_superposition.py train \
    --resume "$run_path" --device cuda \
    2>&1 | tee -a logs/stage1-seed106-train.log
