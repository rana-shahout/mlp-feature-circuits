# Shared features, two independent decoders, and native computation

**SPEC v0.9 — shared-code pilot on the existing frozen two-layer teacher**

## 0. Purpose and boundaries

Run this procedure:

    Learn one sparse code with two independent heads
      -> check hidden reconstruction and both output paths
      -> evaluate individual contributions across Boolean contexts
      -> derive and interpret C2_native = W2 @ S.

The primary new model is the researcher's proposal, NOT a native-tied SAE:

    actual native h -> shared encoder -> ReLU -> z
                                            |-> learned S,q -> h_hat
                                            |-> learned B,c -> y_hat_free

After training, evaluate `y_hat_native = W2 @ h_hat + b2` through the frozen
native output layer. This native path is NOT a training loss, tied head,
initialization target, or model-selection criterion for the joint model.

This is a new proposal-learning experiment, not a continuation of the v0.8
feature-addition search. Do not launch or extend its width sweep. Preserve
all historical results; do not modify a running unrelated experiment.

Research question: Does shared reconstruction-and-prediction learning give
an informative sparse representation of the native hidden state, compared
with reconstruction-only and prediction-only learning, and what native
output computation do its reconstructed components support?

Separate these claims:
1. A code reconstructs the native state.
2. Its independent head predicts the teacher's output.
3. Its reconstruction preserves the native output through frozen W2.
4. Individual components have understandable activation conditions,
   representation roles, and downstream contributions.

Low errors in 1–3 do not prove 4. Nor is B approximately W2 S a discovery:
for a fixed code and unconstrained minimum-norm least-squares heads, that
relationship follows from the affine teacher suffix. Our head constraints
and finite optimization need not produce exact equality.

No hard accept/reject feature pruning is used. Keep all learned coordinates
and report graded evidence, support, harmful terms, redundancy, and residuals.
A component with reconstruction value but no output effect is not invalid.

## 1. Frozen setting and notation

### 1.1 Native teacher

Use the already approved width-256 teacher, including its mistakes:

    a in {0,1}^20
    h = ReLU(W1 @ a + b1)       # h: 256
    y = W2 @ h + b2             # y: 9 real-valued semantic scores

Two affine layers, one hidden ReLU, no output ReLU, sigmoid, or softmax.
The training target is the actual continuous teacher output, NOT the ideal
Boolean labels. Those labels are allowed only for explanatory annotations.

The displayed 20->256->9 model is the existing canonical export. If importing
legacy weights, perform once:

    W1 = W1_old @ P; b1 = b1_old
    W2 = Q.T @ W2_old; b2 = Q.T @ b2_old

This preserves native hidden states and nine semantic scores. It does not
remove the original input bottleneck from the teacher's learning history or
preserve discarded nonsemantic output coordinates. Keep P,Q in import
provenance only, not the main diagrams.

Historical checkpoint hint, to verify rather than guess:

    /home/rana/mlp-feature-circuits/experiments/v2_pairwise_superposition/
    checkpoints/capacity256-20261027/best.pt

Reported SHA256:

    47a61c86e48e58f09bd1d2da167a5a3a904e8fdb5a39c171bf542b6994d49c70

Resolve source checkpoint, export, and dataset using provenance. A hash or
shape conflict is a blocker to resolve, not permission to substitute a model.
Teacher parameters stay frozen and in evaluation mode throughout.

Input labels: a0,...,a19. Output labels, verify ordering:

    g0 AND(a0,a1)          g1 AND(a0,a2)
    g2 AND(a3,a4)          g3 AND(a5,a6)
    g4 AND(a7,a8)          g5 AND(a9,a10)
    g6 XOR(a11,a12)        g7 a13 AND NOT a14
    g8 OR(g0,g2,g3,g4,g5), excluding g1.

### 1.2 New code, no reuse of old feature identities

In raw native units:

    z(h) = ReLU(E @ h + e)      # p = 512 features
    h_hat = S @ z + q          # learned reconstruction head
    y_hat_free = B @ z + c     # independently learned prediction head

Shapes:

    E: 512x256       e: 512
    S: 256x512       q: 256
    B: 9x512         c: 9

Use new IDs such as `joint_s9101_l0.001:f173`. Never equate them with the
old whole-network TC 173/910. The old TC, M,d, acceptance lists, S2 fits, and
C2 matrices are not training targets, initialization sources, or filters.

`E` is not the old linear D2: the new encoder includes ReLU. `S` is a learned
feature embedding, NOT an assumed inverse or pseudoinverse of E.
For a fixed encoder gate pattern, a local affine readout can be written
`G(h) E`, but it is input-dependent and is not a global D2 factorization.

The native first-layer feature coordinates remain known: D1=I20 and C1=W1.
A new feature's input dependence is explicitly:

    z_m(a) = ReLU(sum_r E[m,r] ReLU(W1[r,:]a+b1[r]) + e[m]).

The outer gate belongs to the analysis encoder, not to the teacher.

## 2. Models and bounded run matrix

Train three families with the same encoder architecture, width, data, and
optimization budget:

| Family | Hidden reconstruction | Independent output head | Loss weights |
|---|---|---|---|
| SAE | S,q learned | absent | alpha=1, beta=0 |
| TC_H | absent | B,c learned | alpha=0, beta=1 |
| JOINT | S,q learned | B,c independently learned | alpha=0.5, beta=0.5 |

For TC_H, hidden reconstruction and native-path metrics are N/A. Do not
fit a post-hoc reverse head to fabricate these metrics. For SAE, free-head
prediction metrics are N/A. Native-path evaluation is available via W2 S.

Primary sparse grid:

    p = 512 only
    seeds = [9101, 9102, 9103]
    lambda = [1e-4, 1e-3, 1e-2]

This is 27 training configurations. Add lambda=0 controls for seed 9101
only, one per family: maximum 30 configurations. No extra loss-weight,
width, architecture, target, or sparsity-algorithm sweep is authorized.
Zero-penalty controls remain separate from sparse claims.

Also evaluate the untrained algebraic native-coordinate reference:

    z_id=h, S_id=I256, q_id=0, B_id=W2, c_id=b2.

It has zero reconstruction/output errors by construction. Measure its mean
active native-neuron count. It is a boundary reference, not a learned feature
discovery result, and its width is 256 rather than 512.

All regimes learn from h. The old x->y TC is only historical context; its
reported errors on other populations are not a head-to-head baseline here.

## 3. Data separation and locks

Reuse exact v0.5 input-key partitions, not the old readout itself:

    FIT: 11,200 original teacher-training inputs
    SELECT: 2,800 disjoint inputs for checkpoint/lambda selection

Load their recorded manifests. Do not reconstruct them from undocumented
seeds or re-split opportunistically. If counts differ, reconcile provenance
before training. Do not combine FIT and SELECT for a final refit.

Use the existing v0.7 development banks:

    V: 21,437 rows, development interpretation/feature selection
    A: 20,812 rows, subsequent input-disjoint reused audit

These are reported reference counts, not constants to force. Verify exact
keys, checksums, exclusions, same-bank one-bit edges, complete four-corner
contexts, and historical exposure tags. Both exclude FIT/SELECT and every
protected-final key. Reuse available pairs (7,8), (11,12), (0,1) and nuisance
flips a15,...,a19. No response-conditioned input sampling.

FIT trains parameters and all normalization statistics. SELECT chooses
checkpoints and designated sparsity operating points. V selects case-study
features, display panels, and locks descriptive interpretations. New A
calculations occur only after all such choices are locked. Neither V nor A
may update learned parameters, choose lambda/checkpoints, or trigger retries
with scientifically different settings.

A has been examined in previous studies: label it `reused development audit`,
not untouched confirmation or final generalization. Record teacher-training,
old-TC-training, and prior-human-analysis exposure separately.

Never load protected final activations, labels, predictions, or scores. Use
only exclusion-key manifests as permitted. If required provenance or exact
bank exclusion cannot be verified, run synthetic checks and report the
specific learned-stage blocker. Do not invent a replacement population.

Cache native h and y once for permitted banks, retaining keys and model hash.
Teacher targets are detached. Chunk memory use; do not create a
N_examples x 512 x 256 tensor for every feature removal.

## 4. Training in normalized units; export exact raw maps

### 4.1 Fit-only global normalization

Compute on FIT:

    mu_h = mean(h)
    mu_y = mean(y)
    sigma_h^2 = mean(||h-mu_h||_2^2)
    sigma_y^2 = mean(||y-mu_y||_2^2)
    u = (h-mu_h)/sigma_h
    v = (y-mu_y)/sigma_y

Use one scalar scale per space, not per-neuron whitening. Both variances
must exceed 1e-12. Verify `mu_y=W2 mu_h+b2` within numerical tolerance.
This makes a mean predictor's normalized reconstruction loss approximately
one on FIT. Do not recompute scales on SELECT/V/A.

Train normalized parameters:

    z = ReLU(E0 @ u + e0)
    u_hat = S0 @ z + q0
    v_hat_free = B0 @ z + c0

Export:

    E = E0/sigma_h
    e = e0 - E0 @ mu_h/sigma_h
    S = sigma_h*S0; q = mu_h + sigma_h*q0
    B = sigma_y*B0; c = mu_y + sigma_y*c0

Missing heads are absent, not populated with a fitted replacement. Test
normalized/raw forward equality and serialization round-trips.

### 4.2 Objective and independent scale constraints

For batch averages, sum errors over the coordinates of each normalized
space and sum latent activations; do not divide L1 by width:

    L_h = mean_examples sum_256 (u_hat-u)^2
    L_y = mean_examples sum_9   (v_hat_free-v)^2
    L_1 = mean_examples sum_512 z_m
    L = alpha*L_h + beta*L_y + lambda*L_1

Unavailable loss terms are omitted. No label loss, latent-semantic label,
B-W2S penalty, native-path output loss, hidden intervention loss, or reuse of
old D2 targets is permitted in training.

Prevent the shrinking-z/enlarging-decoder escape independently for each
present normalized decoder:

    ||S0[:,m]||_2 <= 1
    ||B0[:,m]||_2 <= 1

Initialize each present column at unit norm. After every optimizer update,
project columns with norm>1 onto their unit balls. Leave shorter columns
unchanged. Do not rescale encoder rows to undo this projection. Zero columns
remain zero; no feature resampling. These are norm caps on independently
learned directions, NOT B=W2S, NOT tied encoder/decoder weights, and NOT a
combined decoder-norm constraint.

Use the same per-head cap for that head in its single-objective baseline.
Report norms and saturation frequency. The joint latent gauge is constrained
by both heads; identical lambda does NOT guarantee identical sparsity across
families. Compare measured sparsity and frontiers, not lambda alone.

### 4.3 Initialization, optimizer, checkpoints

For seed s:
- Generate E0 rows from standard Gaussian and normalize each to unit norm.
- e0=0; q0=0; c0=0 in normalized units.
- Independently generate Gaussian S0/B0 columns and normalize each.
- Use independent RNG substreams for E0,S0,B0 so missing heads do not shift
  the draws. For a given seed, a head shared between model families starts
  from identical parameters. Every lambda starts from the same initialized
  weights; do not warm-start from another lambda.
- Shuffle seed is s+1000; pair minibatch ordering across families/lambdas.
- Do not initialize B from W2S or S from old readouts/dictionaries.

Adam: lr 1e-3 to 1e-5 cosine over maximum 2,000 epochs; betas (0.9,0.999),
eps 1e-8, weight_decay=0; batch size 256; no dropout or mixed precision.
Evaluate SELECT once per epoch. Select the checkpoint with minimum complete
family training objective evaluated on SELECT, including L1 and its fixed
lambda. Early-stop after 200 epochs without at least 1e-7 improvement;
keep the earlier epoch on a tie. Save initial/best/last checkpoints and all
component losses. No refit after selection.

Use float32 CUDA training; retain actual checkpoint weights. Use float64
for accounting, analytic removal identities, and numerical tests. Disable
TF32 for reproduction checks. Determinism limitations must be documented;
do not silently turn off failed tests. If a numerical failure occurs, fix
implementation and rerun the same configuration with logs preserved, not a
new scientific hyperparameter configuration.

The budget is a ceiling, not a promise of convergence. Report late-improving
runs, dead latents, NaNs, capped epochs, and collapsed codes. No automatic
rescue epochs or expanded lambda grid.

### 4.4 Check that this really is joint learning

Unit-test separate gradient paths BEFORE decoder projection:

    grad(L_h, E0) can be nonzero; grad(L_h, S0) can be nonzero
    grad(L_h, B0)=0/None
    grad(L_y, E0) can be nonzero; grad(L_y, B0) can be nonzero
    grad(L_y, S0)=0/None

Teacher gradients remain absent. Both losses update the SAME E0/e0 object.
No second hidden encoder, stop-gradient between one head and z, or bypass
from h directly to an output head is permitted.

As a diagnostic only, record norms and cosine of the two fidelity gradients
on E0 for 8 fixed FIT minibatches at initialization, epoch 100 if reached,
and the selected checkpoint. Do not rebalance losses from these measurements.

## 5. Whole-code evaluation and fair comparisons

### 5.1 Every run, every available path

Measure on FIT, SELECT, V, and (after locks) A:

    hidden:       h_hat versus h
    free output:  B z+c versus y
    native output: W2 h_hat+b2 versus y

For each, report raw MSE, relative RMSE `||error||_F/||target||_F`, and
FIT-variance-normalized MSE. For outputs, also report each of nine score
coordinates. Undefined normalizers are N/A, not zero. Optional Boolean
agreement uses the old threshold 0.5 but never selects models or features.

Sparsity: strict L0 using z>0, meaningful L0 using z_m>tau_m, and mean L1.
Define tau_m=max(1e-8,1e-3*RMS_FIT(z_m)) after checkpoint selection. Report
dead-on-FIT latents, inactive-on-A latents, active supports, and distributions
of decoder-column norms. Retain negative h_hat and y predictions unchanged.

Report mean-baseline errors and the native-coordinate reference. Do not
claim speedup from low L0: executing the full teacher to get h plus a coder
is not an inference replacement.

### 5.2 Operating points are selected without looking at A

Show all sparse-grid results and all three seeds. Never retain only the best
seed. Zero-lambda controls are marked separately.

For each family and seed, designate a primary representative by the positive
lambda with SELECT mean strict L0 closest to 128 in absolute log ratio:

    abs(log(max(L0,1e-12)/128))

Tie-break by lower SELECT objective, then smaller lambda. Report the actual
L0; do not label the comparison matched unless each model being compared
falls within 25% of the target. Repeat a numeric, secondary table for target
64 using the same rule. No additional training if the grid lacks overlap.
No interpolation of learned matrices or feature identities between runs.

For JOINT vs SAE, examine hidden and native-output errors at comparable
measured sparsity; inspect the full fidelity/sparsity frontier. A valid
tradeoff claim must disclose differences in both errors, not choose a point
only by output error. For JOINT vs TC_H, compare free-output fidelity and
sparsity; the TC_H hidden/native-path cells remain N/A. Do not assert TC_H
lost native-state information without a separate measurement.

With three seeds, report all points plus mean and range, not unwarranted
significance or an IID confidence interval over correlated input contexts.

## 6. Derive the native map after freezing the coder

For every selected checkpoint that has S:

    C2_native = W2 @ S                 # 9 x 512
    gamma_native = W2 @ q + b2         # 9
    r_h = h - (S @ z + q)
    y_hat_native = C2_native @ z + gamma_native

Exact accounting:

    y = C2_native @ z + gamma_native + W2 @ r_h

For JOINT, also retain:

    y_hat_free - y_hat_native
      = (B-C2_native) @ z + (c-gamma_native)

Both identities are tests, not discoveries. Do not fit C2_native, use an
inverse, use old M,d, refit S after training, or optimize B to resemble C2.
No assumption that S E = I, E S = I, or W2 = C2 E is valid here.

Matrix similarity B vs C2 and output-path agreement are consistency
measurements, not proof of native feature recovery. Redundant coordinates
can allow different columns to make the same aggregate prediction.

For a displayed feature m, show separately:
- Encoder row E[m,:] and e[m]: how the new encoder computes its activation.
- Reconstruction column S[:,m]: where its decoded contribution goes in h.
- Prediction column B[:,m]: its independently learned output write.
- Native column C2_native[:,m]: actual W2 applied to its reconstruction
  direction.

E and S directions are not interchangeable. Gate changes in the new
encoder are not new native gates.

## 7. Individual contribution tests: frozen code, fixed heads

Run full numerical per-feature summaries for all latent IDs of the primary
representatives (target 128, all seeds). TC_H has only free-output tests;
SAE has hidden/native-path tests; JOINT has all three. Do not delete features
from the final model based on these results.

### 7.1 Leave one CODE component out, without re-encoding

For each example, compute z ONCE. Define:

    z_minus_m = z - z_m * unit_m
    v_h,m = S[:,m] * z_m
    v_free,m = B[:,m] * z_m
    v_native,m = C2_native[:,m] * z_m

    h_hat_minus_m = h_hat - v_h,m
    y_free_minus_m = y_hat_free - v_free,m
    y_native_minus_m = y_hat_native - v_native,m

Never re-encode an edited state, refit a head, clip h_hat_minus_m to ReLU,
or modify the teacher. This is removal from a reconstruction/explanation.

Define signed marginal loss changes, each relative to its OWN full-code
baseline, not another path:

    Delta_h(m) = [E||h-h_hat_minus_m||^2 - E||h-h_hat||^2]/sigma_h^2
    Delta_free(m) = [E||y-y_free_minus_m||^2 - E||y-y_hat_free||^2]/sigma_y^2
    Delta_native(m) = [E||y-y_native_minus_m||^2 - E||y-y_hat_native||^2]/sigma_y^2

Positive: removing the component worsens that loss in this fixed model.
Negative: removal improves it. Zero need not mean a meaningless feature;
redundancy, cancellation, task irrelevance, or low support can matter.

Compute efficiently and verify with explicit ablations on small cases:

    ||r+v_m||^2-||r||^2 = 2 r.T v_m + ||v_m||^2

Report global and active-only changes, plus per-output changes. Keep support
counts. Mark a per-feature active-case summary `limited_support` below 32
active examples in that bank and `unassessed_active` at zero; keep its numeric
aggregate contributions visible. These flags never remove its coordinate.
Use the fixed FIT-derived tau_m for meaningful activity. Cosines/relative
ratios with a zero denominator are undefined, not successful agreement.
Loss-change contributions are not additive across simultaneous removals.

### 7.2 Representation magnitude and head-role agreement

Report scale-aware descriptive magnitudes:

    A_m = RMS(z_m)*||S[:,m]||/sigma_h
    N_m = RMS(z_m)*||C2_native[:,m]||/sigma_y
    T_m = RMS(z_m)*||B[:,m]||/sigma_y
    G_m = RMS(z_m)*||B[:,m]-C2_native[:,m]||/sigma_y

Use A-bank RMS only for audit measurements, never to choose features.
Also report cosine of B and C2 columns when both are nonzero, and aggregate
head-gap error including offsets. G_m can be nonzero even when the aggregate
head gap is zero because contributions cancel. Do not add G_m values and
call the sum aggregate error.

Do not assign binary `good feature` labels from these four numbers. Keep
representation usefulness, free-head usefulness, native-path usefulness,
role agreement, and support in separate columns. Features only relevant to
reconstruction are legitimate observations, not automatically rejected.

### 7.3 Native anchored response is an algebraic diagnostic here

For a hypothetical raw-hidden edit `h - S[:,m] z_m`, the suffix obeys:

    F(h)-F(h-S[:,m]z_m) = W2 S[:,m] z_m.

Verify on small tensors. Do not claim an independent causal validation from
this equality. The suffix is affine; edited states may be off the native
post-ReLU manifold, and feature coordinates need not be independently
manipulable. No large native intervention campaign is authorized.

### 7.4 Natural input contexts and complete accounting

For the existing complete four-corner contexts and nuisance flips in V/A,
run the full native first layer and coder separately at EACH actual input.
This differs from latent removal, which freezes the original z.

For each historical focal pair and selected case feature, report all four
values of z_m, encoder preactivation, output scores, and signed decoder
contributions. With corner order 00,01,10,11 compute:

    baseline = z00
    first = z10-z00
    second = z01-z00
    joint = z11-z10-z01+z00

Keep baseline and singles, not only joint coefficients. Verify across every
context and output:

    Delta_ik y_j
      = sum_m C2_native[j,m] Delta_ik z_m
        + W2[j,:] Delta_ik r_h.

Likewise account for ordinary output values using gamma_native and for
finite single-bit changes. Sum other features exactly. A small displayed
subset is never silently substituted for the full model.

Native nuisance invariance is not assumed: the imperfect teacher can change
when task-distractor bits change. Compare to its actual changes.

## 8. Case-study selection without historical feature labels

Lock the primary representatives using SELECT. Use V only to choose two
main JOINT case studies from seed 9101, target-128 representative:

    output g4 with pair (a7,a8)
    output g6 with pair (a11,a12)

Among features with >=32 active V examples and >=16 responsive complete
contexts for that pair (|Delta_ik z|>tau_m), rank by:

    mean_V_contexts |C2_native[j,m] * Delta_ik z_m|.

Select the highest-scoring feature per case with feature-ID tie-breaks.
This is deliberately contribution-based case selection, NOT proof of a
semantic name. Export the top-five shortlist and its signed cancellation,
and keep the selected case even if its head gap or interpretation is poor.
If support is inadequate, report no supported case; do not lower criteria.
The same feature may occur in both cases; that fact must remain visible.

For each selected feature:
1. Fit a descriptive activation template on V corners, not on audit data:
   z_m = a_template + b_template * t(pair), with b_template>=0 and a free
   intercept. Test the full list below; select minimum V MSE, then list
   order. This does not change the coder or its maps.

       a_i, a_k, AND, OR, XOR, a_i AND NOT a_k,
       a_k AND NOT a_i, NOR, NOT a_i, NOT a_k

   Also report a constant-template baseline. Retain signed C2/B weights
   separately from activation-template polarity.
2. Describe this as a template fit, not a discovered truth label. On A report
   normalized error (MSE divided by FIT variance of z_m, if >1e-12),
   variation across backgrounds, gate-pattern frequencies,
   near-threshold counts, and failures. If the best template is poor, say so.
3. Choose the first lexicographic complete A background by input keys BEFORE
   looking at activations. No search for a more dramatic or active example.
   Report inactive cases as such. Use all available contexts for statistics.
4. Show actual native h, learned encoder preactivation and gate, z_m,
   decoded native direction, B/C2 output writes, other-feature contributions,
   reconstruction residual contribution, offsets, and native output.

For compact native-neuron tables, choose at most six neurons with largest
`|E[m,r]| std_FIT(h_r)` for encoder-preactivation accounting and a separate
six with largest `|S[r,m]| RMS_FIT(z_m)` for reconstruction accounting.
No alignment or one-to-one neuron match is assumed. Show explicit remainder
for all undisplayed neurons. Retain full tables numerically.

Repeat numerical context and role summaries for the other two seeds and
all families where their respective paths exist. The two main worked cases
are illustrations, not the basis for seed-robust claims.

## 9. Tests that must pass before learned conclusions

Implement and run tests for:

1. Canonical teacher export equality, frozen parameters, detached targets.
2. Exact normalization/raw export identities and checkpoint reload.
3. Shared-encoder gradient routes, independent decoder parameters,
   no training loss or gradient through the native-output evaluation path.
4. Independent decoder unit-ball projections, including zero columns, and
   no latent shrink/decoder growth escape under the caps.
5. Native identity reference, C2=W2S, bias, and exact residual accounting.
6. Analytic one-feature removal versus explicit forward calculation for
   hidden, free, and native output losses, including negative loss changes.
7. Anchored linear-native response identity and full four-corner accounting.
8. A representation-only control: h=(u,v), y=u, identity dictionary.
   Removing v damages reconstruction but does not damage the output.
   The report must not label v nonexistent.
9. A cancellation control: h=a, z=(a,a), S=(1/2,1/2), B=(2,-1), W2=1.
   Reconstruction and both aggregate outputs are exact, while B != W2S.
   Individual role-gap metrics must reveal the mismatch rather than declare
   every feature validated by aggregate fidelity.
10. A relocation-of-computation control: h=(a,b), z=(a,b,ab), and S selects
    the first two coordinates. A prediction head for AND can use ab, although
    the native boundary coordinates are a,b. This is a separate synthetic
    nonlinear-suffix illustration; do not change the learned teacher or call
    the extra encoder gate a native gate.
11. As a mathematical control only, fixed-code unconstrained least squares
    for linear native outputs can yield B=W2S automatically. This is NOT a
    post-training head-refitting experiment on our learned models.
12. Data exclusions/locks, feature-ID preservation, support/zero-denominator
    handling, unbiased sign retention, readable categorical plotting labels.

Numerical accounting target: FP64 atol 1e-9, rtol 1e-8; legacy FP32 comparison
start from recorded atol 2e-6, rtol 2e-5. If scaling requires a revised tolerance,
record the discrepancy and justification; never change scientific error
criteria to pass a numerical check. Synthetic nonzero-gradient controls must
be constructed so active gates and nonzero residuals make them meaningful.

## 10. Main figures: only six, one chart per image

No giant full-matrix atlas and no SVG-only delivery. Save PNG at >=200 dpi;
PDF versions optional. Use Matplotlib defaults, readable 11–12+ point text,
clear captions, no clipped labels, and explicit entity tick names.

1. **Representation/sparsity frontier.** Mean strict active features on x;
   hidden FIT-variance-normalized reconstruction MSE on y. SAE and JOINT,
   all sparse-grid points and seeds; native identity marked separately.
2. **Output/sparsity frontier.** Same x; normalized output MSE on y, explicitly
   distinguish JOINT-free, JOINT-native, SAE-native, and TC_H-free. Do not
   pool or average paths. Show seed points and the declared representatives.
3. **What individual components do.** Signed Delta_h versus signed
   Delta_native for all supported features of the representative JOINT model;
   zero axes visible, negative values retained. Annotate only the two locked
   cases. Support and free-head effects remain in the associated table.
4. **Native C2, the main matrix.** Nine named output rows and <=10 new feature
   columns, annotated signed coefficients. Include the two case features and
   fill by `||C2[:,m]|| RMS_V(z_m)` with ID ties. Freeze display IDs on V.
   This is a subset of the full model, not a pruned model. B-column comparisons
   and the aggregate residual belong in a compact adjacent table, not an
   unreadable extra full heatmap.
5. **AND-context case.** Four categorical corners, measured feature activation
   across all eligible audit backgrounds: show mean and interquartile range,
   explicitly not an independent-sample confidence interval. Plot the frozen
   V-fitted template predictions alongside. Explain context-dependence.
6. **XOR-context case.** Same construction for the second feature. It need not
   activate like XOR; retain the actual pattern and signed output role.

If a case lacks support, use a clear support-status table and omit its plot.
Do not invent an attractive replacement. Full contribution examples and
native input coefficients are readable tables, not extra oversized figures.

A short Markdown workflow at the start of the report must identify actual
native operations, new encoder gates, learned heads, and the derived path.
Use categorical labels such as a7, neuron 112, and joint_s9101:f37. Never use
2.5 as a neuron/feature/corner index. Fractional numeric errors/colorbars are
valid. Feature IDs are run-local, not old TC identities. Render and visually
inspect every main figure at intended report size.

## 11. Execution and deliverables

Authorized: implement, validate, train the 30 specified feature-model runs,
evaluate, debug, generate figures, inspect them, and report end-to-end.
The teacher is never trained. Use CUDA for learned workloads. Check GPU
availability without interrupting other jobs. Use tmux for durable execution,
monitor actual completion, and save commands, logs, status, and provenance.
Do not ask the researcher to copy/paste shell commands.

Create a new non-overwriting run directory. Archive SPEC/prompt/config and
source snapshot. Hash old checkpoints/readouts before and after. Record
software/GPU environment and actual per-run epochs, failures, selected
checkpoints, loss components, and normalizations. Stream per-feature
statistics and use analytic deletions to avoid excessive memory.

If CUDA or required source artifacts are absent, finish CPU synthetic checks,
report concrete blockers, and do not substitute another teacher or invent
training results. A failed scientific hypothesis is not a blocker requiring
rescue training. Runtime/numerical fixes preserve all failed logs and keep
scientific settings unchanged.

Deliver:

    SUMMARY.md                 # <=1,000 words; measured answer and limits
    REPORT.md                  # compact narrative, six figures, useful tables
    REPORT.pdf                 # optional, 6–8 legible pages if renderer exists
    RUN_INDEX.csv              # every authorized configuration and status
    protocol_lock.json
    data_lock.json, exposure_counts.json
    selection_lock.json, case_lock.json
    normalization.npz
    controls/                  # synthetic inputs/results and tests
    runs/<family_seed_lambda>/ # initial/best/last, loss history, metrics
    matrices/<representative>/
        W1.npy, b1.npy, W2.npy, b2.npy, D1.npy, C1.npy
        E.npy, e.npy
        S.npy, q.npy            # only if present
        B.npy, c.npy            # only if present
        C2_native.npy, gamma_native.npy # only if S exists
        dimensions_and_ids.json
    whole_code_metrics.csv
    sparsity_frontiers.csv
    feature_contributions_V.csv, feature_contributions_A.csv
    feature_roles_and_support.csv
    context_metrics.csv
    worked_cases/              # full native/encoder/code/output accounting
    figures/                   # six maximum, PNG and optional PDF
    tests.log, execution.log, commands.sh
    provenance.json, preservation_verified.json

SUMMARY answers:
1. Did joint learning improve any measured fidelity/sparsity tradeoff over
   the corresponding single-head baseline, across seeds?
2. Do independently good TC predictions also preserve the actual native
   output when going through reconstructed h? Where do paths disagree?
3. Which components help reconstruction, free prediction, and native-path
   prediction; which show cancellation, harmful additions, or little support?
4. What do the two worked examples explain through original gates and W2S?
   What remains in other components and the reconstruction residual?
5. What is algebraic, what is empirical, and what has NOT been established?

The contribution is NOT guaranteed by more parameters or low training loss.
No causal necessity, unique feature basis, exact D2 inverse, inference speedup,
or nonlinear downstream mechanism is claimed by this pilot.

## 12. Not authorized in v0.9

No teacher retraining, added teacher layer, softmax/sigmoid output change,
new task, width sweep, tied B=W2S architecture, B-W2S consistency penalty,
native-path training loss, old M refit, pseudoinverse feature selection,
post-hoc reverse decoder, output-regression replacement, adaptive feature
accept/reject loop, causal-intervention campaign, pruning, symbolic
replacement, inference optimization, or protected-final scoring.

Stop after the report. This specification explicitly supersedes older bans
on training a NEW feature model and learning S within a joint coder; it does
not authorize refitting a reverse S on the old frozen whole-network TC.

## 13. Evidence boundary and background

Source-derived setting: the historical Stage 2 report
`Pasted markdown(20261027-175620).md` supplies the approved teacher hash and
architecture; v0.5 supplies FIT/SELECT input manifests; v0.7 `REPORT.pdf` and
v0.8 Phase I `REPORT(1).pdf` supply canonical notation, V/A banks, and their
prior-exposure limitations. Reopen the actual repository sources/locks before
execution. This specification's model objectives, scale caps, p=512, seeds,
lambda grid, loss changes, and case selection are NEW experimental choices,
not conclusions already established in those reports.

Relevant primary precedents (background, not copied experimental protocols):
- Le, Patterson & White (NeurIPS 2018), Supervised autoencoders: Improving
  generalization performance with unsupervised regularizers.
  https://proceedings.neurips.cc/paper/2018/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html
  Shared reconstruction/prediction architecture; not a native-circuit proof.
- Braun et al. (NeurIPS 2024), Identifying Functionally Important Features
  with End-to-End Sparse Dictionary Learning.
  https://arxiv.org/abs/2405.12241
  Downstream-aware SAE precedent, NOT the independent-head architecture
  authorized here.

Do not present the generic two-head architecture as a novel invention. Our
question is whether this controlled implementation improves discovery and
interpretation of the frozen teacher's representation and native output map.
