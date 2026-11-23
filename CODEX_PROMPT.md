# Execute the shared-encoder, two-independent-decoder experiment

Read the accompanying **SPEC.md, version 0.9**, completely before coding.
Use this specification, not an older repository-root SPEC or the v0.8 width
sweep. Archive the supplied documents and create a new non-overwriting
experiment/run directory. Do not rewrite historical specifications.

## Scientific objective

We want to test the researcher's architecture on the CURRENT frozen teacher:

    actual native h -> one shared encoder -> ReLU -> z
                                                   |-> learned S,q -> h_hat
                                                   |-> learned B,c -> y_hat_free

Both decoders are independent learned parameter sets. The output branch
must backpropagate into the SAME encoder as reconstruction.

The procedure is:

    learn the shared code
      -> check reconstruction and both output paths
      -> inspect per-feature contributions across contexts
      -> derive and interpret the native map C2_native = W2 @ S.

Do NOT replace this with an output-aware SAE whose B is tied to W2 S.
Do NOT train with a native-path loss or B-W2S consistency penalty.
Only AFTER the model is frozen, evaluate:

    y_hat_native = W2 @ (S @ z + q) + b2.

## 1. Verify the native model and data

Use the approved frozen width-256 teacher and the canonical
20-input -> 256 native ReLU neurons -> 9 continuous scores view. Verify
checkpoint hashes and export equivalence. Teacher targets include its errors;
Boolean task labels are annotations, not supervision for the new code.

Use the exact v0.5 FIT/SELECT key partitions and v0.7 V/A manifests described
in SPEC v0.9. Keep protected-final keys excluded and record historical
exposure. A is a reused development audit, not pristine confirmation.

P/Q belong only to the one-time legacy import. They should not clutter the
main analysis. Do not change the teacher, append softmax, or create a deeper
native model. Do not reuse the old TC feature IDs or M,d as training targets.

## 2. Train the bounded comparison

Implement the three specified families:

- SAE-only: reconstruct actual h.
- TC_H-only: predict actual teacher y from h.
- JOINT: reconstruct h and independently predict y through one shared code.

Use p=512, three seeds, the three fixed positive sparsity coefficients, and
the three single-seed zero-penalty controls: maximum 30 configurations.
Honor the exact normalization, objective weights, independent decoder norm
caps, initialization/shuffle pairing, optimizer, checkpointing, and stopping
rules in SPEC.md. No additional sweeps or scientific rescue runs.

For JOINT, S and B are not tied to each other, the encoder, or native weights.
Check gradient routes explicitly. Freeze teacher parameters and detach targets.

TC_H has no reconstruction head; mark its hidden/native-path metrics N/A.
Do not add a post-hoc reverse fit to fill in those cells. SAE has no independent
prediction head. The native-coordinate identity is an algebraic reference,
not a learned discovery.

## 3. Freeze, evaluate, and derive the native map

Evaluate all three available quantities separately:

    h_hat vs h
    y_hat_free vs y
    y_hat_native vs y

Report both raw and normalized errors, per-output results, actual sparsity,
dead/low-support features, convergence, and all seeds. Checkpoint/lambda
choices use SELECT only. Use V to lock case features and explanatory templates;
do not inspect new A results until those locks exist.

Convert normalized parameters back to raw native units and compute:

    C2_native = W2 @ S
    gamma_native = W2 @ q + b2
    r_h = h - (S @ z + q)
    y = C2_native @ z + gamma_native + W2 @ r_h

C2 is computed, not fitted. S is a learned dictionary, not an assumed D2
inverse. E is followed by ReLU and is not the old global linear D2.
No pseudoinverse construction is part of this experiment.

## 4. Evaluate individual contributions without altering the code

For each representative model and feature m, compute z once, zero that
coordinate only, and keep all other latent values and both heads fixed.
Measure signed changes in hidden, free-output, and native-output losses.
Use analytic deletion identities and validate them against explicit small
examples. Do not re-encode, refit, clip reconstructed states, or prune the
final model.

Keep representation usefulness, output usefulness, head-role agreement,
redundancy/cancellation, and support separate. No universal binary feature
accept/reject rule is authorized. A representation-only component is not
invalid merely because the chosen output ignores it.

For fixed natural Boolean contexts, rerun the complete native network and
coder at each corner, preserve all four responses and finite changes, and
account for the full output including other features, bias, and native
reconstruction residual. Never label a feature AND/XOR solely from an output
coefficient or old feature numbering.

The suffix is linear. W2 S[:,m] z_m is an exact downstream write of a decoded
component, not an independent causal discovery. The encoder's new gates
must not be attributed to the native first layer.

## 5. Report only useful figures and finish execution

Produce the six main plots and readable numerical case tables specified in
SPEC.md. Use actual input names, native neuron IDs, new run-local latent IDs,
and output names. No fractional entity ticks, no giant full-matrix atlas,
no unsupported causal labels, and no SVG-only report. Inspect every rendered
figure at reading size; retain full arrays separately.

You are authorized to implement, test, execute, debug, monitor, and finish
this bounded experiment yourself. Use CUDA for learned workloads, tmux where
appropriate, preserve old artifacts, and save logs, actual commands, hashes,
source snapshots, selected checkpoints, and exit status.

Do not ask me to copy/paste execution commands. Do not stop after writing
code. If required artifacts or CUDA are unavailable, finish the small
synthetic checks and report the exact blocker; do not train a substitute
teacher or invent results.

Return paths to SUMMARY.md, REPORT.md/PDF, the main figures, all-run metrics,
per-feature contribution tables, and the raw S/B/C2_native arrays. State what
joint learning improved, what it failed to improve, and which conclusions
remain algebraic or unproven. Stop after the report.
