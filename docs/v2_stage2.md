# V2 Stage 2 approved frozen-teacher pilot

The researcher approved the width256 original MLP for imitation/mechanistic research on 2026-10-27. This accepts studying an imperfect teacher; it does not claim all primitive functions met the original 99% validation balanced-accuracy gate (3/8 passed).

Approved checkpoint: `experiments/v2_pairwise_superposition/checkpoints/capacity256-20261027/best.pt`. SHA-256: `47a61c86e48e58f09bd1d2da167a5a3a904e8fdb5a39c171bf542b6994d49c70`. Architecture12→256→12, teacher seed3105, original unweighted full-output MSE, selected epoch1987. Validation task metrics: exact-vector97.5667%, full-output MSE0.0060496294, semantic MSE0.0080656799. These Boolean-task metrics verify identity, not transcoder targets.

Only the six configurations in `experiments/v2_pairwise_superposition/configs/stage2_transcoders.yaml` are authorized: widths512/1024 × lambda0/1e-4/1e-3. Both widths use student initialization seed4105; lambda variants at each width start identically and use CPU shuffle seed4106. Across widths initialization is not identical. Adam lr.001→.00001 cosine, batch256, budget2000epochs, patience200/min_delta1e-8 on validation teacher reconstruction MSE. Minimum raw validation reconstruction MSE selects checkpoints, independently of sparsity or labels. No dead-unit resampling or auxiliary supervision.

Loss = mean_examples sum_12_outputs (T−F)^2 + lambda mean_examples sum_units |z|. Reported full-output MSE divides the reconstruction term by12. L1 is not divided by width; lambda is comparable across configurations under unit-L2 decoder columns, normalized after every Adam step. Targets are detached full12D teacher outputs, cached after teacher validation verification. No Boolean labels or Q enter the optimization function.

Normalized reconstruction MSE divides coordinate-mean MSE by mean per-output-coordinate validation teacher variance, centered separately per coordinate. Teacher fidelity, teacher task accuracy, student task accuracy, per-input L0, per-interaction participation and semantic interpretability remain separate claims. The full nondominated validation MSE/L0 frontier is reported rather than an arbitrary sparsest winner.

Use existing train/validation partitions; neither historical test nor reserved final population is scored. Reused controlled contexts are historical diagnostics, including density-supported and OOD regimes, never unbiased evidence or checkpoint selectors. Original-neuron interventions/weight edits remain unauthorized. Known endpoint coordinates and exact signed decompositions do not establish mechanism recovery.

Implementation: `experiments/v2_pairwise_superposition/stage2/`. New artifacts: runs/checkpoints/reports `stage2-transcoders-20261027`. Original checkpoints/results and earlier reports remain unchanged. Teacher verification and full provenance are recorded before training.
