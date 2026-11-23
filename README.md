# MLP feature circuits: Stages 0 and 1

The authorized **first sparse transcoder experiment**, its execution commands,
and scientific safeguards are documented in [docs/transcoder.md](docs/transcoder.md).
It freezes seed 109 and all other original teachers; its independently seeded
final population is reserved rather than automatically scored. Historical scope
statements below describe the earlier baseline implementation.

The new **evaluation-only** original-neuron diagnostics and exact pair/triple
interaction decomposition are documented in [docs/diagnostics.md](docs/diagnostics.md).
They preserve the five trained teachers and require user-started execution;
the new diagnostic code/tests have not been executed by Codex. No transcoder is
implemented. The scope statements below describe the earlier baseline workflow.

SPEC.md v0.2 is the scientific source of truth. This implementation provides
analytical tests and an original sparse-superposition teacher baseline only.
It does not implement a transcoder, learned-neuron circuit analysis, finite
interaction analysis, or interventions. Knowing the target functions does not
identify the trained network's internal mechanism.

The initial implementation was delivered without executing tests or experiments.
The user subsequently authorized agent-operated environment repair, verification,
and Stage 1 training in tmux. See the diagnosed-environment note below and the
run's verification/training logs for actual execution status. No later stage is
authorized by that exception.

## Scientific review and explicit choices

The SPEC dimensions are consistent: `P (16,32)`, `Q (12,8)`,
`W1 (48,16)`, `W2 (12,48)`. Source equations use columns; PyTorch batches
use `x = a @ P.T`, `target = g @ Q.T`. The network receives only `x`.

- P has unit columns, not orthogonal columns. There is no linear inverse for
  arbitrary 32-dimensional inputs. Its continuous nullspace alone does not prove
  two distinct sparse binary vectors collide. Random real dictionaries can be
  injective on a finite binary set even when overcomplete. We report singular
  values, the column Gram matrix, sampled nearest-input distances and differing
  labels, and pseudoinverse reconstruction error. These are limited ambiguity
  diagnostics, not exhaustive sparse-decoding guarantees.
- Q is orthonormal after reduced QR with a fixed diagonal-sign convention. For
  `s = Q.T @ prediction`, the output MSE is
  `(8/12) * semantic_MSE + orthogonal_residual_MSE`. Both terms are reported;
  accurate projected outputs alone can hide off-span error.
- Positive/negative sampling makes sampling mode informative about labels by
  construction. Literal mode-label independence conflicts with that design.
  Modes and requested rule indices are diagnostic metadata only. Nuisance
  marginal correlations and conditional prevalences are audited, but cannot
  certify the absence of all shortcuts. Inspect `data_audit.json` before training.
- Hard negatives are local false patterns at Hamming distance one from at least
  one satisfying antecedent pattern. This explicitly includes XOR 00/11,
  inhibitor-on cases, and one-bit breaks of nested rules. Full antecedent contrast
  tables supply additional distinguishing cases. Local patterns are sampled
  uniformly within each positive/negative pool; final density is sampled from the
  configured distribution conditional on containing that local pattern.
- Inputs are deduplicated before a seeded random 70/15/15 split. This estimates
  held-out performance on the accepted, deduplicated sampling population, not the
  original proposal distribution or all `2^32` inputs. Rejection changes mode
  fractions and feature balance; both are reported. No stratification claim is
  made. Counts and missing classes remain visible.
- Contrast tables vary each rule's complete antecedent set in a fixed random
  background. All other relevant inputs and distractors remain fixed per context.
  Overlap is recorded against **all** sample partitions. Evaluation reports all,
  training-seen, validation-seen, test-seen, and entirely unseen rows separately,
  plus per-designed-rule results and counts of fully unseen contexts. These
  contrast distributions may have larger densities than training. Unseen does
  not by itself imply out-of-distribution generalization.
- Capacity and adequate performance are empirical questions. Review flags default
  to balanced accuracy below 0.9 or semantic MSE above 0.05, but are explicitly
  heuristics, not acceptance criteria from SPEC. Undefined metrics use JSON null.
  No stage is automatically passed. Preserve failures and review every rule.
- A decoded regression score threshold of 0.5 defines classification. Scores are
  unbounded and are not probabilities. Test/contrast results never select a
  checkpoint; selection minimizes validation output MSE (earliest epoch on ties).

These choices fill operational gaps without changing SPEC's target functions,
geometry, training loss, or scientific hypotheses. SPEC.md was not modified.

## Files and scope

- `src/data.py`: eight rule definitions, P/Q, bounded sampler, unique-vector
  splits, contrast memberships/overlap, sampling and geometry audits.
- `src/models.py`: original ReLU MLP with exposed hidden states.
- `src/runtime.py`: configuration validation, explicit devices, atomic checkpoint
  and JSON writes, reproducibility setup.
- `src/train_mlp.py`: Adam training against mixed targets, validation selection,
  epoch-boundary restart, dataset/config integrity hashes.
- `src/evaluation.py`: task metrics, hidden activation summaries/caches, individual
  errors, projection residuals, basic plots and review report.
- `scripts/run_sparse_superposition.py`: prepare/train/evaluate entry point.
- `tests/`: Stage 0 explicit Boolean ReLU fixture, original-coordinate identity
  checks, data/metric/checkpoint/device checks. No training tests or experiments.

Python, PyTorch, NumPy, PyYAML, and Matplotlib are runtime requirements; pytest
is optional for verification. Pandas is not needed by these two stages.

## Setup (user executes)

From the repository root, use the existing virtual environment if appropriate:

```bash
cd /home/rana/mlp-feature-circuits
source .venv/bin/activate
python -m pip install -e '.[test]'
```

If you prefer a new environment, use these commands **instead**:

```bash
cd /home/rana/mlp-feature-circuits
python3 -m venv .venv-stage01
source .venv-stage01/bin/activate
python -m pip install -e '.[test]'
```

These commands install dependencies only when you choose to run them. A working
NVIDIA driver and CUDA-enabled PyTorch are required for default training and
evaluation. The default package-index PyTorch build is not guaranteed to suit
your driver/platform; select a compatible build for your machine if necessary.
Check the actual machine rather than assuming GPU/driver compatibility; the
diagnosed host environment below is specific to the current Stage 1 run.

### Diagnosed environment for stage1-seed106

The initial environment installed PyTorch 2.14.1+cu130, but host driver
575.51.03 reports CUDA 12.9 support. The failed training log explicitly reported
an insufficient driver. A CUDA 12.8 PyTorch build is the proposed dependency fix;
installing a CUDA-enabled build without matching driver compatibility is not
sufficient. The user subsequently authorized dependency repair, verification,
and agent-operated tmux training, overriding the earlier execution restriction
for this task. SPEC's scientific assumptions remain unchanged.

The official PyTorch previous-versions page lists this installation command:

```bash
.venv/bin/python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
```

See <https://pytorch.org/get-started/previous-versions/>. The prepared audit has
a readable review in `results/stage1-seed106/audit_review.md`. After dependency
repair, `bash scripts/run_stage1_verified.sh results/stage1-seed106` performs
CUDA preflight, tests, and training sequentially, stopping on any failure. It
uses the project interpreter explicitly, retains logs, holds a run lock, and
records final status in `verified_train.exitcode`. It does not launch evaluation
or any later scientific stage. Start it only once per run, inside host tmux.

## Verification and Stage 0 (user executes)

Check the driver and the actual PyTorch CUDA device first:

```bash
nvidia-smi
python -c 'from src.runtime import select_device; select_device("cuda")'
python -m pytest -q -m 'not cuda'
python -m pytest -q -m cuda
```

The first test command runs CPU analytical/data/unit checks only. The second
runs explicitly marked GPU checks; these skip if CUDA is unavailable, so the
separate device preflight above is required. A skipped check is not a GPU pass.
To run just the Stage 0 formula fixtures:

```bash
python -m pytest -q tests/test_stage0.py -m 'not cuda'
```

The analytical network realizes AND, OR, XOR, inhibition, and three-way AND on
its complete tiny Boolean domain. Coordinate identities additionally use the
main overcomplete P and orthonormal Q and include original biases/ReLU. Those
identities are implementation checks, not scientific discoveries.

## Stage 1: inspect data, train, then evaluate (user executes)

Use a new run directory for each experiment; existing directories are never
overwritten by prepare/new-train. If the name below already exists, choose a new
one and use it consistently. `prepare` is CPU data construction only, with no
training/evaluation or CUDA memory reservation:

```bash
python scripts/run_sparse_superposition.py prepare \
  --config configs/sparse_superposition_v1.yaml \
  --run-dir results/stage1-seed106
less results/stage1-seed106/data_audit.json
python scripts/run_sparse_superposition.py train \
  --resume results/stage1-seed106 --device cuda
python scripts/run_sparse_superposition.py evaluate \
  --run-dir results/stage1-seed106 --device cuda
```

Alternatively, a single new training invocation prepares data and trains:

```bash
python scripts/run_sparse_superposition.py train \
  --config configs/sparse_superposition_v1.yaml --device cuda
```

It prints a unique generated run directory under `results/`. Evaluation remains
a separate invocation. An explicit CPU debug override is `--device cpu`; there
is never automatic fallback. Data construction and report serialization use CPU;
model training/evaluation use the requested device.

## tmux workflow with logs (user executes)

Start a session:

```bash
cd /home/rana/mlp-feature-circuits
tmux new -s mlp-stage1
```

Inside tmux (adjust activation if using `.venv-stage01`):

```bash
cd /home/rana/mlp-feature-circuits
source .venv/bin/activate
set -o pipefail
mkdir -p logs
export PYTHONUNBUFFERED=1
nvidia-smi
python -c 'from src.runtime import select_device; select_device("cuda")'
python -m pytest -q -m 'not cuda' 2>&1 | tee logs/stage0-cpu.log
python -m pytest -q -m cuda 2>&1 | tee logs/stage0-cuda.log
python scripts/run_sparse_superposition.py prepare \
  --config configs/sparse_superposition_v1.yaml \
  --run-dir results/stage1-seed106 2>&1 | tee logs/stage1-seed106-prepare.log
less results/stage1-seed106/data_audit.json
python scripts/run_sparse_superposition.py train \
  --resume results/stage1-seed106 --device cuda \
  2>&1 | tee -a logs/stage1-seed106-train.log
```

Check each command's outcome before moving to the next. Detach while training
with **Ctrl-b**, then **d**. From a later terminal, reattach:

```bash
tmux attach -t mlp-stage1
```

When training finishes, run evaluation inside the session:

```bash
python scripts/run_sparse_superposition.py evaluate \
  --run-dir results/stage1-seed106 --device cuda \
  2>&1 | tee -a logs/stage1-seed106-evaluate.log
```

To resume after an interruption, rerun the same `train --resume ...` command.
Do not start two writers for one run. Completed runs are no-ops on resume;
epochs/config are frozen. To change hyperparameters, create a new versioned
YAML and a new run. Keep comparison decisions independent of held-out results.

For later initialization-seed comparisons, the new-run CLI accepts
`--model-seed 107`. Other seeds remain fixed, so P/Q/data/splits are identical
in value. Begin with one run; no sweep is launched automatically.

## Artifacts and restart limits

Each run contains frozen `config.yaml`, `SPEC.snapshot.md`, `dataset.npz`,
`data_audit.json`, `data_manifest.json`, and after training `environment.json`, `history.json`,
`best.pt` and `last.pt`. The NPZ contains P, Q, a, g, split indices, sampling
modes/requested rules, contrast vectors and complete context memberships,
and overlap partition codes (-1 unseen, 0 train, 1 validation, 2 test).

Best/last checkpoints include all model weights and biases, optimizer, config,
dataset/config hashes, RNG/shuffle states, best epoch, and history. Resume starts
at the next fully saved epoch; interrupted partial epochs are repeated. Each
checkpoint is replaced atomically. best.pt and last.pt are separate files, so
an interruption between their writes can leave best.pt one epoch ahead; resume
repeats that epoch from last.pt. Exact replay requires the same deterministic
software/hardware/device environment. CUDA unsupported deterministic operations
raise errors; reproducibility across platforms/releases is not promised.
Checkpoints contain pickled state: load only trusted local files.

Each evaluation creates a new `evaluation-<UTC timestamp>-<id>/` containing:

- `metrics.json`, `per_rule.csv`, `classification_errors.csv`;
- `predictions_activations.npz` (outputs, original hidden preactivations and
  activations, in dataset/contrast row order);
- `baseline.png` (P/Q geometry and test balanced accuracy), `report.md`.
- `evaluation_manifest.json` (checkpoint/config/dataset hashes and evaluation
  software/device details).

Metrics include per-output MSE, prevalence, confusion counts, precision/recall,
specificity, balanced accuracy, positive/negative MSE, full-vector accuracy, and
out-of-Q-span residual. Empty subsets and absent classes are explicit. The full
sample evaluation includes training and is not a generalization estimate.
Train/validation/test metrics also include a constant predictor formed only from
the training mean target. Parameter shapes, norms, ranges and singular values
are saved as descriptive diagnostics, without assigning learned mechanisms.
No uncertainty intervals are supplied: conditional contrast rows are correlated,
and one training seed cannot establish across-seed uncertainty.

Failures after run creation write timestamped traceback logs in the run;
the terminal/tee captures dependency/import errors too. Logs, configuration,
negative results and individual error rows should be retained for review.

Unresolved scientific decisions: quantitative Stage 1 acceptance criteria,
shortcut-control strength, whether width 48 learns every rule, and robustness
across initialization and P/Q seeds. Review results before authorizing Stage 2.

## Bounded Stage 1 tuning

After observing the first baseline's rare-rule failures, the user requested
further tuning. `configs/stage1_tuning_v1.yaml` fixes three candidates (widths
48/128/256, 1,000 epochs each) with unchanged data, embeddings, model seed, Adam
settings and unweighted mixed-output MSE. This is an exploratory follow-up:
the initial test/contrast results have already informed the design. Larger
widths transparently change capacity as allowed in SPEC Stage 1; they do not
change the single-hidden-layer architecture or supply semantic inputs to it.

`scripts/tune_stage1.py` copies the exact original dataset to each candidate,
selects each checkpoint by validation MSE as before, then ranks candidates
lexicographically by minimum per-rule validation balanced accuracy, mean
balanced accuracy, and negative validation MSE. The selected candidate is
recorded before fresh evaluation. No candidate test/contrast results influence
selection. The fixed threshold remains 0.5. Candidate checkpoints and all
validation results are retained; rerunning resumes saved epochs.

Only after selection, generate fresh sample proposals with seed 303 and fresh
contrast contexts with seed 305. Remove every original sample/contrast vector
from both; also remove fresh sample vectors from the contrast set. Exclusion
can shift the population and break complete truth-table groups; save audits,
membership mappings and complete-group counts. Compare the original and selected
models on these **same** fresh sets, reporting exact-vector and class-specific
performance. This is not a guarantee of improved accuracy or a mechanism claim.
Further tuning must not treat those fresh results as an untouched final test.

Authorized execution, inside GPU-host tmux:

```bash
set -o pipefail
.venv/bin/python -m pytest -q
.venv/bin/python -u scripts/tune_stage1.py \
  --plan configs/stage1_tuning_v1.yaml \
  --run-dir results/stage1-tuning-v1 \
  2>&1 | tee -a logs/stage1-tuning-v1.log
```

Results: `validation_comparison.json`, `selection.json`, candidate subdirectories,
`fresh_dataset.npz`, `fresh_audit.json`, `fresh_predictions.npz`,
`fresh_metrics.json` and `report.md`. No later scientific stage is implemented.

## Stage 1 coverage extension v2

The first tuning round still missed three-way AND positives. Diagnostics on the
already inspected contrast set found poor performance at densities 7–10, outside
the initial training support of 2–6. Stage 1 v2 **changes the training population**:
retain original partitions and add matched antecedent truth tables using random
non-antecedent contexts with 2–6 active coordinates (total density up to 10).
This is a documented extension, not evidence of extrapolation from the original
2–6-bit distribution. SPEC.md and all earlier runs remain unchanged.

`scripts/prepare_coverage.py` generates separate training/validation context
identities, excludes all original sample/contrast vectors and all previous fresh
holdouts from additions, and excludes training additions from validation. Tables
can become incomplete after exclusion/deduplication; saved memberships preserve
the exact retained rows. Original P/Q, targets and single-hidden-layer geometry
are preserved. Source modes and context identifiers are never model inputs.

Training sampling is now a uniform mixture over eight outputs and each output's
two classes, using only training labels. Equivalent row probabilities are
proportional to the mean inverse-class-frequency score across outputs. Rows are
sampled with replacement. This changes the effective training distribution;
the loss on each row is still ordinary unweighted MSE against Qg. No semantic
loss weighting, decoder input, or direct clause input is introduced. Expected
sampled label prevalence and effective sample size are logged. Validation and
evaluation are not reweighted.

The frozen v2 plan compares widths 512 and 1024 for 1,000 epochs each, with
batch size 1024 and cosine learning-rate decay from 0.001 to 0.00001. Uniform
sampling/constant learning rate remain defaults for the original configuration.
The epoch-indexed schedule and stored sampling RNG support restart. Each
checkpoint is still chosen by validation MSE. Candidate ranking prioritizes
worst-rule balanced accuracy, then macro balanced accuracy, then MSE. These
simultaneous changes constitute a remediation experiment, not an attribution
of gains to one isolated factor.

Before evaluating fresh sets, require every validation rule's balanced accuracy
to reach 0.95 and positive recall to reach 0.90. These are operational targets,
not automatic scientific approval. If a round fails, retain results and leave
its fresh evaluation untouched while diagnosing validation. After a passing
selection, use fresh seeds 703/705, excluding all training/validation data and
all earlier evaluation vectors, and compare with the prior 256-neuron teacher.
One successful seed still does not establish robustness or internal mechanisms.

```bash
.venv/bin/python scripts/prepare_coverage.py --plan configs/stage1_coverage_v2.yaml
.venv/bin/python -u scripts/tune_stage1.py \
  --plan configs/stage1_coverage_v2.yaml --run-dir results/stage1-coverage-v2
```

Only Stage 1 is authorized; no transcoder or learned-neuron analysis is started.

### Frozen-protocol replication after the coverage result

After v2's first selected model passed the validation target, evaluate seeds
106–110 under the same frozen width-1024, 1,000-epoch protocol. Seed 106 uses
the already trained selected checkpoint; seeds 107–110 start independently from
scratch. No architecture, dataset, loss, schedule, class weights, threshold or
epoch-budget selection uses the replication holdout results. The same reserved
v2 sets are scored for every seed after validation checkpoint selection, with
no best-seed filtering. This is an exploratory initialization-robustness check
on a fixed conditional evaluation population, not new independent data for each
seed or robustness across P/Q realizations.

The two worker commands run on separate available host GPUs, with each worker
keeping its CUDA visibility setting for checkpoint restart:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/replicate_stage1.py --seeds 107 109
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/replicate_stage1.py --seeds 108 110
```

Use tmux and tee as above. `configs/stage1_replication_v2.yaml` freezes the seed
list and input paths. Per-seed locks prevent duplicate writers. When all seeds
finish, the workers write `results/stage1-replication-v2/summary.json` and
`report.md`, retaining every seed's metrics, failures, checkpoint and predictions.
Across-initialization standard deviations do not quantify uncertainty over
new nuisance contexts. CUDA reproducibility across hardware remains best effort.
