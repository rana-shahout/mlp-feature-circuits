#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_path=${1:-results/transcoder-v1-seed109}
if [[ -f "$run_path/complete.json" ]]; then
    printf 'Experiment already complete; retained report: %s/report.md\n' "$run_path"
    exit 0
fi
mkdir -p logs
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
trap 'workflow_status=$?; printf "%s\n" "$workflow_status" > logs/transcoder-v1.exitcode' EXIT
.venv/bin/python -c 'from src.runtime import select_device; select_device("cuda")'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_transcoder.py prepare --run-dir "$run_path"
.venv/bin/python scripts/run_transcoder.py smoke --run-dir "$run_path"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_transcoder.py train --width 128 --run-dir "$run_path" 2>&1 | tee -a logs/transcoder-v1-width128.log &
width128_job=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/run_transcoder.py train --width 256 --run-dir "$run_path" 2>&1 | tee -a logs/transcoder-v1-width256.log &
width256_job=$!
set +e
wait "$width128_job"; width128_status=$?
wait "$width256_job"; width256_status=$?
set -e
if [[ $width128_status != 0 || $width256_status != 0 ]]; then
    printf 'Training failed: width128=%s width256=%s\n' "$width128_status" "$width256_status" >&2
    exit 1
fi
.venv/bin/python scripts/run_transcoder.py select --run-dir "$run_path"
.venv/bin/python -u scripts/run_transcoder.py evaluate --scope diagnostic --run-dir "$run_path"
.venv/bin/python -u scripts/run_transcoder.py interactions --run-dir "$run_path"
.venv/bin/python scripts/run_transcoder.py report --run-dir "$run_path"
printf 'Report: %s/report.md\nFinal evaluation population remains reserved and unscored.\n' "$run_path"
