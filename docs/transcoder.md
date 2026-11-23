# First sparse transcoder experiment

Scientific source: SPEC.md and the reviewed teacher diagnostic report at
`results/diagnostics-20261027T132121Z-726003/report.md`. The user authorized
implementation and agent-operated CUDA/tmux execution. No teacher retraining or
original-neuron interventions are authorized. All five original best/last
checkpoints are hashed before preparation and verified unchanged at each phase.

## Frozen experimental decisions

Use original teacher seed **109**, previously selected by validation metrics.
The runtime checks its checkpoint tensor shapes against configuration:
**16 → 1024 → 12**, not the original proposed hidden width 48. Teacher parameters
have `requires_grad=False`; all cached supervision is `F(Pa)` under no-grad.
The original task labels and Q never enter the transcoder training loss.

Students are 16→128→12 and 16→256→12 ReLU networks. For each width use L1
coefficients `[0, 1e-4, 1e-3, 1e-2]`, with 500 epochs of Adam, batch size 1024,
and cosine learning rate 0.001→0.00001. One student seed, 2109, is used for the
initial sweep; results are exploratory, not student-seed robustness conclusions.
Uniformly sample the teacher's saved training partition, with no true-label
balancing or resampling. Ground-truth functions are used only for evaluation
and separately identified sampling of the reserved population.

The exact objective is

```
MSE = mean over batch and all 12 output coordinates of (T(x)-F(x))**2
L1  = mean over examples of sum over latent units z_k(x)
loss = MSE + lambda * L1
```

Decoder columns are projected to unit L2 norm after initialization and each
optimizer step. This is an explicit regularization-design choice: otherwise
positive ReLU homogeneity permits `E,e → sE,se` and `B → B/s`, arbitrarily
reducing activation L1 without changing outputs. All penalties, including zero,
use the same constraint. It does not establish feature identifiability.
Because L1 sums over units, adding redundant active units incurs extra cost;
there is no hidden division by width. L0 counts strictly positive activations,
without an epsilon activity cutoff. Dead-unit counts and per-unit activation
means, standard deviations, prevalence, and quantiles are retained.

Every configuration's checkpoint minimizes **validation full-output teacher
MSE**, not Boolean accuracy and not test/contrast performance. The MSE/mean-L0
Pareto frontier and best fidelity configuration are committed in `selection.json`
before diagnostic test/contrast or interaction comparisons. Candidate checkpoint
hashes are frozen there. The original test and earlier sample/contrast sets are
diagnostic populations, already inspected during earlier work.

128/256-unit students are smaller than the 1024-neuron teacher. Lower contribution
counts are partly a width constraint; they must not be called a recovered compact
mechanism without fidelity, appropriate controls, and subsequent original-neuron
tests. Same-width zero-L1 controls are part of this sweep.

## Sealed final population

Preparation uses independent seeds 1800/1801 to generate a new sparse-mixture
sample and matched-rule contrast population. Exclude all teacher data partitions,
old contrast sets, earlier saved holdouts, every pair/triple diagnostic corner,
conditional truth-table corner and nuisance-flip probe. The two new sets are
mutually disjoint. Actual counts and the dataset hash are recorded in
`reserved_final/manifest.json`; there are bounded generation attempts. Exclusion
conditions the population and can remove complete contrast groups, so it is not
an unconditional IID claim.

The initial workflow **does not score** the final population, even after
validation selection. It is reserved for later final evaluation. A separate
explicit `evaluate --scope final` command scores only the previously committed
validation frontier; it refuses repeated use once `final_scored.json` exists.
Do not modify the sweep after seeing those results and still call that set blind.

## Fidelity, errors, and nonlinear interactions

For train/validation/original test/previous sample/previous contrast populations,
report full-output MSE against teacher, semantic-coordinate MSE after Q.T,
exact-vector teacher/student agreement (all eight thresholded semantic outputs),
per-output agreement, and agreement conditional on teacher-correct/error rows.
Teacher and student task metrics against Boolean labels are separate. On each
output's teacher failures, retain fractions preserved versus accidentally
corrected by the student. A correction can be a fidelity failure; it is not the
training objective. Full 12D prediction residuals and all evaluation latents are
cached per population and configuration.

For the transcoder, `K=EP`, `A=Q.T B`, and
`g_TC(a)=A ReLU(Ka+e)+Q.T c`. The implementation reuses exact finite-difference
algebra while saving these actual transcoder matrix names. Each original pair
and triple is evaluated on exactly the original diagnostic contexts, not new
or optimized backgrounds. All 496 pairs, 4,960 triples, eight output coordinates,
and 15 contexts are retained.

Each student's `interaction_comparison/orderN/` includes:

- `contributions.npy`: **(term,context,semantic output,transcoder unit)**,
  complete float64 signed contributions without pruning or top-k truncation;
- `full_output_corner_residual.npy`: **(term,context,corner,12)**, student minus
  the frozen teacher on every corner;
- `statistics.npz`, `signed_comparison.csv`: signed interaction, positive and
  negative sums, absolute mass, cancellation, top-1/5/10 absolute-mass fractions,
  participation-ratio effective units and width-normalized effective fraction;
- `unit_distractor_context_sd.npy` and `residuals.npz`: per-unit/context and
  teacher/student/residual interaction variation across the zero/2/4/6-nuisance
  backgrounds, holding other semantic inputs fixed;
- `terms.npy`: exact input-coordinate indices in the original order.

Participation ratio is `(sum |c_k|)^2 / sum c_k^2`, with zero for an all-zero
vector. It is invariant to uniform contribution scaling and does not divide by
the potentially cancelling signed total. Contributions and direct student-output
differences must agree within float64 tolerances, including all eight triple
corners. Original teacher values and signed statistics are independently checked
against its unchanged checkpoint and prior exact archive. The original full
neuron arrays remain in that archive; no copy overwrites them.

Summary masks distinguish nonzero ground-truth interactions, actual teacher
effects above 1e-6, and teacher-success/failure corner sets. The 1e-6 cutoff is
only a descriptive summary mask; every raw contribution and residual is kept.
Interactions are compared **to the teacher's actual values**, even when the
teacher's values differ from ideal Boolean interactions.

Complete per-unit AND/XOR interaction tables retain realized contribution signs,
context variability, and feature IDs. Error-association tables compare each
feature's activity and signed semantic write on teacher-success/error subsets.
They are observational and potentially confounded by other co-occurring inputs.
Units are local to a checkpoint; no alignment across students or with original
neurons is assumed. The research report proposes features for later investigation
in the original weights, without identifying shared mechanisms from fidelity.

## Execution and exact commands

No new dependencies are required. Use the existing CUDA-enabled project `.venv`.
Do not replace working CUDA dependencies with an unverified build. Default device
is CUDA; unavailable CUDA fails clearly, with no silent CPU fallback.

The full run uses two available A100 GPUs, one width sweep on each, and runs
tests/CUDA smoke first. Started by the agent in tmux as authorized:

```bash
cd /home/rana/mlp-feature-circuits
tmux new-session -d -s transcoder-v1 \
  'set -o pipefail; bash scripts/run_transcoder_experiment.sh results/transcoder-v1-seed109 2>&1 | tee -a logs/transcoder-v1-workflow.log; exec bash'
```

Attach with `tmux attach -t transcoder-v1`; detach with Ctrl-b then d. The two
width logs are `logs/transcoder-v1-width128.log` and `...-width256.log`.
`logs/transcoder-v1.exitcode` is written when the workflow ends. The teacher
diagnostic archive is approximately 30 GB; the new student archives add roughly
10–15 GB including full signed comparisons. Results use a new experiment root.

Individual reproducible commands (from the repository root, inside tmux):

```bash
source .venv/bin/activate
set -o pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q
python scripts/run_transcoder.py prepare --run-dir results/transcoder-v1-seed109
python scripts/run_transcoder.py smoke --run-dir results/transcoder-v1-seed109
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_transcoder.py train --width 128 \
  --run-dir results/transcoder-v1-seed109 2>&1 | tee -a logs/transcoder-v1-width128.log
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_transcoder.py train --width 256 \
  --run-dir results/transcoder-v1-seed109 2>&1 | tee -a logs/transcoder-v1-width256.log
python scripts/run_transcoder.py select --run-dir results/transcoder-v1-seed109
python -u scripts/run_transcoder.py evaluate --scope diagnostic --run-dir results/transcoder-v1-seed109
python -u scripts/run_transcoder.py interactions --run-dir results/transcoder-v1-seed109
python scripts/run_transcoder.py report --run-dir results/transcoder-v1-seed109
```

The smoke optimizer graph uses a disposable student, never saves it, and asserts
the teacher has no gradients. Training resumes from each student's last full
epoch under a matching immutable protocol; completed configurations are skipped.
The only training RNG after initialization is the saved shuffle generator; no
dropout or stochastic augmentation is used. Restart each width with the same
CUDA visibility. Diagnostics refuse an existing evaluation directory; do not
rerun completed diagnostic phases blindly. Interaction phases skip completed
configurations. Tracebacks are saved by command and width. Failed configurations
remain available, not discarded from the report.

For a genuinely new experiment, copy/version the YAML and use a new output root;
the source hashes/config cannot change on resume. Do not run two workers for the
same configuration (per-configuration locks reject duplicate writers).

**Later final evaluation only, after reviewing and committing the selection:**

```bash
python -u scripts/run_transcoder.py evaluate --scope final \
  --run-dir results/transcoder-v1-seed109 2>&1 | tee logs/transcoder-v1-final.log
```

That final command is intentionally absent from the initial automatic workflow.
Final metrics live under each selected configuration's `final_evaluation/`;
the initial research report remains diagnostic, clearly labeled.
