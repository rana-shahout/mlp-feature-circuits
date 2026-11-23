#!/usr/bin/env bash
# USER-STARTED ONLY. No training, detached processes, or tmux creation here.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_tag="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output_path=${1:-results/diagnostics-$run_tag}
if [[ -e "$output_path" ]]; then
    printf 'Refusing existing output path: %s\n' "$output_path" >&2
    exit 1
fi
mkdir -p logs
log_path="logs/diagnostics-$run_tag.log"
status_path="logs/diagnostics-$run_tag.exitcode"
trap 'diagnostic_status=$?; printf "%s\n" "$diagnostic_status" > "$status_path"' EXIT
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'Output: %s\nLog: %s\n' "$output_path" "$log_path" | tee "$log_path"
.venv/bin/python -c 'from src.runtime import select_device; select_device("cuda")' \
    2>&1 | tee -a "$log_path"
if [[ ${RUN_DIAGNOSTIC_TESTS:-0} == 1 ]]; then
    .venv/bin/python -m pytest -q tests/test_interactions.py \
        2>&1 | tee -a "$log_path"
fi
.venv/bin/python -u scripts/diagnose_mlp.py \
    --config configs/diagnostics_v1.yaml --device cuda --output-dir "$output_path" \
    2>&1 | tee -a "$log_path"
printf 'Completed. Review: %s/report.md\n' "$output_path" | tee -a "$log_path"
