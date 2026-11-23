# Original MLP diagnostics: design and user execution

This implements the user's diagnostic/interaction request; it does not authorize
training, a transcoder, or original-neuron interventions. SPEC.md is unchanged.
Code and tests were written and statically inspected, **not executed by Codex**.
Neuron IDs, new conditional results, and a seed recommendation require the user
to run the evaluation. The generated report supplies measured findings rather
than inventing those answers in advance.

## What the repository actually measures

`src/evaluation.py:prediction_metrics` defines `exact_vector_accuracy` as the
fraction of rows for which **all eight** semantic output classifications match.
This is the source of the earlier sample/contrast accuracies. It is not mean
per-feature accuracy. Each output uses the saved threshold (currently 0.5) on
`prediction @ Q`, computed in NumPy float64 from float32 model predictions.
The values are regression scores, not calibrated probabilities. Diagnostics
call that same metric function, preserving its definitions and adding:

- precision TP/(TP+FP), recall TP/(TP+FN), FPR FP/(FP+TN), FNR FN/(FN+TP);
- per-output MSE, MAE, signed prediction bias, prevalence and class support;
- the existing mixed-output MSE, semantic MSE, and orthogonal output residual.

An undefined denominator is JSON null/empty CSV, never an invented zero score.
Training population evaluation enumerates unique saved rows once, unweighted.
It is **not** the online training loss under balanced-label sampling.

Current source runs are the width-1024 teacher for seed 106 in
`results/stage1-coverage-v2/width1024`, and seeds 107–110 in
`results/stage1-replication-v2/seedNNN`. The replication directory's seed106
contains reference metadata, not its own trained checkpoint. The diagnostic
configuration uses the actual five checkpoint locations. Saved configurations
and data manifests show a shared dataset and fixed P/Q. Runtime deserialization,
tensor checks, and forward checks are deliberately deferred to user execution.
Static SHA-256 inspection confirmed that all five current `best.pt` files match
the checkpoint hashes stored with their earlier evaluation metrics. No source
checkpoint was written during this implementation.

Train/validation include the documented coverage extension with densities up to
10. Combined and original/coverage subsets are reported separately; subsets
overlap their combined population and must not be added together. `test` is the
original 7,500-row test partition, already inspected during exploratory work.
`fresh_sample` and `fresh_contrast` are the **saved**, previously evaluated v2
sets, not newly blinded samples. Their inputs are verified disjoint from the
model dataset and one another. Every source is read-only; checkpoint, config,
dataset and saved-holdout hashes are compared before/after the full run.

## Three-way AND, conditional tables, and distractors

For every population, retain all eight g4 antecedent patterns, precision/recall,
FN/FP cases and semantic scores, active non-antecedent semantic inputs, nuisance
identities, nuisance count and total active count. Cross-seed archives identify
rows missed by one through all five seeds. Full per-feature metrics cover all
eight outputs, not just g4.

`observational_truth_tables_<population>.csv` groups existing examples by each
rule's complete antecedent pattern. Other inputs vary, so those tables alone
do not isolate an interaction. `controlled_truth_tables.csv` instead enumerates
each rule's full antecedent truth table in each saved fixed context, reporting
all eight target/model outputs and retaining all other inputs. It covers AND,
OR, XOR, inhibition, three-way AND and nested/overlapping compositions.

`groups_<population>.csv` groups errors by total density, nuisance count, exact
active nuisance identity set, each nuisance feature's presence/absence, and that
presence conditional on nuisance count. Supports and class-specific denominators
are retained. The report screens g4 positive FNR differences at matched nuisance
count with at least ten positives on both sides (configured). This is descriptive,
not a causal inference or a multiple-testing-corrected significance result.
Other semantic inputs and total density can still confound those comparisons.
`controlled_distractor_effects.csv` additionally flips each nuisance bit alone
in every fixed background and reports all eight score changes and classification
flips. Ground-truth outputs must remain identical. This detects additive nuisance
dependence too: zero pair/triple effects alone would not exclude a linear nuisance
effect. These constructed input effects are distinct from original-neuron ablations.

## Exact nonlinear decomposition

SPEC calls the user's `D2 = Q.T @ W2` **D2_eff**, distinguishing this exact
effective readout from a hypothetical intrinsic factorization. The code keeps
that name. For hidden width H:

```
C1:            (H,32) = W1 @ P
D2_eff:        (8,H)  = Q.T @ W2
semantic_bias: (8,)   = Q.T @ b2
g_hat(a) = D2_eff @ ReLU(C1 @ a + b1) + semantic_bias
```

For an ordered set S of two/three distinct input coordinates, enumerate every
bit pattern b in lexicographic order (00,01,10,11; 000,...,111), replacing all
intervened coordinates regardless of their context values. The signed operator
is `sum_b (-1)^(len(S)-sum(b)) * f(a_S=b,u)`. Thus AND has pair interaction +1,
OR -1, XOR -2, inhibition -1; three-way AND has triple interaction +1. Nested
rules depend on context (e.g. `(a4 AND a16) XOR a17` has triple effect -2).

Each original neuron r contributes exactly
`D2_eff[j,r] * sum_b sign(b) * ReLU(C1[r,:] @ a_b + b1[r])`.
All 496 pairs and 4,960 triples of the 32 inputs are measured for **each of the
eight outputs**, including distractor-involving effects whose true value is zero.
Non-intervened inputs stay fixed. Fifteen named backgrounds test empty/nuisance
contexts and enabling, inhibitory, XOR, and competing-DNF semantic conditions.
At a given nuisance count, identities are identical across semantic conditions;
count-2/4/6 sets are nested prefixes of a seeded permutation. These are finite
probes, not exhaustive enumeration of all possible non-intervened contexts.

Constructed counterfactuals are not ordinary test rows. Each corner records a
bitmask: 1 training, 2 validation, 4 original test, 8 saved fresh sample,
16 saved fresh contrast, 32 original contrast, 64 earlier v1 holdouts. Flags can
be combined; zero means absent from these saved populations, not guaranteed
out-of-distribution. No observed overlap is hidden or called a new held-out test.

Population metrics use the original float32 forward. Interaction algebra uses
the same saved parameter values promoted to float64 to reduce cancellation
roundoff. "Exact" means exact finite-difference algebra within stated floating
tolerances, not bitwise real arithmetic or thresholding predictions first.
For each batch, direct original-input forward differences must match coordinate
differences and complete neuron sums; disagreement fails the run. Bias inside
ReLU is preserved, while output bias cancels from every positive-order difference.

## Output schema and complete signed data

The default explicit archive is approximately **30 GB** for these five models
and 15 contexts, plus CSV/cached-output overhead. The runner checks free disk
space before evaluation. GPU interaction batches are bounded; lowering
`interaction_batch_size` reduces working memory without changing the probe set.

Each seed directory contains:

- `metrics.json`, per-population predictions, observed truth tables and grouped
  error CSVs; `three_way_and_errors.csv`; `controlled_truth_tables.csv`.
- `checkpoint_metadata.json`: actual checkpoint hash, epoch, shapes and config.
- `coordinates.npz`: C1, D2_eff, original b1, semantic output bias.
- `order2/terms.npy` and `order3/terms.npy`: lexicographic input-index tuples.
- `orderN/contributions.npy`: float64 array
  **(term, context, output feature, original hidden neuron)**. Every signed
  value is retained, including zeros, negative entries and small terms.
- `orderN/hidden_delta.npy`: **(term, context, original hidden neuron)**.
- `orderN/interactions.npz`: `value` and `ground_truth`, each
  **(term, context, output feature)**.
- `orderN/corner_population_flags.npy`: **(term, context, corner)** provenance.
- `orderN/signed_sums.csv`: complete positive sum, negative sum, signed total,
  absolute mass, cancelled mass `2*min(positive,-negative)`, cancellation fraction
  `1-abs(total)/absolute_mass` (zero when absolute mass is zero), and discrepancy
  from the actual Boolean function. Largest positive/negative neuron IDs are
  display aids, not truncation of the decomposition.
- `neuron_totals.csv`: per-output/neuron positive/negative masses and sign counts
  across all measured cases; these probe-weighted totals are not intrinsic
  importance or causal effects. Small roundoff-level contributions are retained.
- `interaction_metrics.json`: separate nonzero-target, zero-target, and
  distractor-involving interaction errors; a sea of true-zero interactions must
  not conceal poor recovery of nonzero effects.

Setting `store_explicit_contributions: false` in a **copied** diagnostic config
omits the large redundant array only. It does not prune neurons: every contribution
is recoverable losslessly from `hidden_delta[t,c,r] * D2_eff[j,r]`. Both factors
are saved in float64; all signed sums remain. Default behavior saves the explicit
array as well, as requested.

At the root, `cross_seed_*.npz` and `cross_seed_agreement.json` compare identical
rows. `embeddings.npz` preserves the shared P/Q values without changing their
sources. `validation_seed_ranking.csv` retains every validation ranking score.
`report.md` answers the research questions with measured tables, screens
associations with support caveats, identifies signed original-neuron contributors,
and makes a **validation-only** seed recommendation. It reports negative evidence
and leaves the retrain-vs-transcoder decision to human review. The review criteria
in YAML are operational aids, not a proof that a mechanism has been recovered.
Neuron IDs are local to each seed; no one-to-one alignment is assumed.

Complete decompositions are algebraic identities, not intervention evidence.
They explain contributions to this trained function even when it is wrong.
They do not show that the largest neurons form an exhaustive compact circuit or
that more training will resolve remaining errors. No automatic retraining,
transcoder fitting, or intervention follows the report.

## Exact commands: user runs everything

No dependency installation is required by this change; use the existing project
environment. From your terminal, create a tmux session:

```bash
cd /home/rana/mlp-feature-circuits
tmux new -s mlp-diagnostics
```

Inside tmux:

```bash
cd /home/rana/mlp-feature-circuits
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
RUN_DIAGNOSTIC_TESTS=1 bash scripts/run_diagnostics.sh
```

This wrapper performs a CUDA preflight, runs the new analytical/metric tests,
then evaluates only if those checks succeed. It logs to a unique
`logs/diagnostics-<UTC timestamp>-<pid>.log`, writes an exit-status companion,
and uses a unique `results/diagnostics-<UTC timestamp>-<pid>/` directory. It
refuses existing output directories and never starts tmux or detached processes
itself. CUDA is mandatory for the CLI defaults; CPU requires an explicit manual
debug override. Missing CUDA fails clearly. CUDA-marked unit tests can skip on
systems without CUDA, which is why the wrapper preflight is mandatory.

Detach using **Ctrl-b**, then **d**; reattach from a new terminal:

```bash
tmux attach -t mlp-diagnostics
```

For a larger output filesystem, pass a new path explicitly:

```bash
RUN_DIAGNOSTIC_TESTS=1 bash scripts/run_diagnostics.sh /scratch/$USER/mlp-diagnostics-$(date -u +%Y%m%dT%H%M%SZ)
```

Or run the exact evaluation directly, without the test wrapper:

```bash
set -o pipefail
mkdir -p logs
run_tag="$(date -u +%Y%m%dT%H%M%SZ)-$$"
python -u scripts/diagnose_mlp.py --config configs/diagnostics_v1.yaml \
  --device cuda --output-dir "results/diagnostics-$run_tag" \
  2>&1 | tee "logs/diagnostics-$run_tag.log"
```

Do not treat a partial report as success: require `complete.json` and exit status
0. A failure writes `failure.log` when the output directory has been created;
run again using a new output directory to preserve partial artifacts. No
numerical tests, evaluations or GPU checks have been performed for this change.
