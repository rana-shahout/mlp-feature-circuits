"""Report-only interpretation of saved controlled-trial arrays; no model inference."""
import json
import numpy as np
from .balanced import OUT,RUN,CK,BASE,NEW,OLD
from .data import RULES,PAIRS
from .run import write_csv,digest
from src.runtime import write_json

def main():
    m=json.loads((OUT/'metrics.json').read_text());a=json.loads((OUT/'phase_a.json').read_text())
    with np.load(OUT/'signed_interactions.npz') as f:
        rows=[]
        for j in range(8):
            corners=f[f'original_p{j}_corners'];density=corners.sum(-1);ind=(density.min(1)>=3)&(density.max(1)<=6)
            old=f[f'original_p{j}_value'][:,j];new=f[f'balanced_p{j}_value'][:,j];truth=f[f'p{j}_ground_truth'][:,j]
            for label,mask in [('all',np.ones(len(old),bool)),('all_corners_density_3_to_6',ind),('some_corner_OOD_density',~ind)]:
                if not mask.any():continue
                rows.append({'rule':RULES[j],'context_regime':label,'n_contexts':int(mask.sum()),'original_target_mae':float(np.abs(old[mask]-truth[mask]).mean()),'balanced_target_mae':float(np.abs(new[mask]-truth[mask]).mean()),'old_new_signed_difference_mae':float(np.abs(old[mask]-new[mask]).mean()),'original_mean_signed':float(old[mask].mean()),'balanced_mean_signed':float(new[mask].mean())})
    write_csv(OUT/'interaction_density_comparison.csv',rows)
    assessment='''Balancing markedly improved the five difficult AND recalls: g0 88.04%→100%, g1 78.00%→100%, g2 75.62%→97.01%, g3 87.11%→98.45%, and g5 71.21%→98.99% on validation. AND(7,8) remains perfect on both training and validation. Thus the original objective was an important contributor to missed positives under the fixed architecture and threshold; this experiment does not establish that loss imbalance was the only cause.

The cost is substantial. Validation exact-vector accuracy falls 90.70%→83.90%; semantic MSE rises 0.016872→0.029210 and full-output MSE 0.012654→0.021909 (about 73% higher). AND(9,10) FPR rises 0.14%→5.39% and precision falls 97.24%→56.48%. XOR and inhibition balanced accuracy also regress. These are empirical scores, not calibrated probabilities. This is improved rare-positive sensitivity rather than an overall superior Boolean predictor. Under the preserved validation full-output-MSE selection criterion, the original model remains preferable across these two completed trials; no test or probe metric was used for selection.

Nonlinear computation is not uniformly preserved or improved. Mean AND(9,10) interaction error across the eligible fixed contexts improves 0.54997→0.18349, while XOR worsens 0.42300→0.51550 and AND(7,8) 0.01489→0.06721 despite perfect ordinary classification. Signed positive/negative neuron contributions are retained without truncation. Interaction tables pool deliberately broad contexts; interaction_density_comparison.csv separately identifies contexts whose every corner stays within density support. These finite comparisons are descriptive of each trained function, not evidence of the same internal circuit.

The baseline's difficult-AND FN medians were approximately 0.19–0.37, with only 10–20% of FNs within 0.05 below threshold. Together with high but imperfect ranking AUC, this indicates weak semantic responses on a subset of positives, not merely tiny threshold miscalibration. Balancing changes those responses but expands negative scores as well. Training errors and validation errors move similarly. The data do not establish a hard capacity limit: one AND remains perfect, and the same 64 neurons learn much higher recall after the objective change. Nor does this isolate optimization from representational interference. AND(3,4) has a notably negative pair cosine (-0.4003), but AND(9,10) is difficult despite positive cosine (+0.2904); geometry alone does not order difficulty. Prior nuisance sensitivity and current distractor-group tables support residual context dependence, without proving its causal origin.

The unchanged 64-neuron architecture is not yet demonstrated sufficient for the intended reliable-all-primitive baseline: several validation balanced accuracies remain below 99%, and no new unbiased final evaluation has been performed. Do not proceed automatically to transcoder training. A transcoder could study an imperfect teacher, but accepting that changed scientific aim requires an explicit decision.

Recommend ONE next controlled experiment, if approved: repeat the **original unweighted objective** with a single alternative initialization/shuffle seed (e.g. 3107), holding embeddings, data, architecture, optimizer, schedule, threshold, budget and selection criterion fixed. The large rule-specific asymmetry and persistent training errors leave optimization basin dependence unresolved; a seed-only replication tests that without changing capacity or introducing another loss tradeoff. This recommendation has NOT been executed. Do not treat seed variation as proof of an architecture limit if it also fails.

The independently seeded final population is still unscored. Keep it reserved until the scientific model-selection decision is finalized.'''
    p=OUT/'report.md';s=p.read_text().replace('ASSESSMENT_PENDING',assessment)
    extra=['','## Train per-rule comparison','','| Rule | Original recall | Balanced recall | Original FPR | Balanced FPR | Original precision | Balanced precision | Original BA | Balanced BA |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for j,rule in enumerate(RULES):
        old=m['original_train']['per_rule'][j];new=m['balanced_train']['per_rule'][j]
        values=[old['recall'],new['recall'],old['false_positive_rate'],new['false_positive_rate'],old['precision'],new['precision'],old['balanced_accuracy'],new['balanced_accuracy']]
        extra.append('| '+rule+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    complete=json.loads((RUN/'training_complete.json').read_text());checks=json.loads((OUT/'interaction_checks.json').read_text())
    extra+=['','## Execution record','',f"CUDA: NVIDIA A100-SXM4-40GB; PyTorch 2.9.1+cu128. One training run took {complete['seconds']:.2f} seconds; patience stopped it at epoch {complete['epochs']}, selecting epoch {complete['best_epoch']}. All 14 tests passed, including CUDA smoke/backpropagation. Maximum full signed-sum versus direct-network residual: {checks['max_signed_sum_vs_direct_network_residual']:.3g}.",'','The initial combined process exited 1 after training because a diagnostic passed a 3D context/corner tensor to the model’s required 2D batch input. Postprocessing was corrected to flatten/restore corner dimensions and completed with exit 0 from the existing best checkpoint; no training was repeated. Original failure and recovery logs are retained under ../../execution_logs/'+NEW+'/. The original training source snapshot remains frozen; corrected source and hashes are retained separately.','', 'Actual tmux launch scripts: execution_logs/'+NEW+'/execute.sh (training plus initial diagnostics), and finish.sh (postprocessing recovery). Test command: `.venv/bin/python -m pytest experiments/v2_pairwise_superposition/tests -q`. Report-only assessment command: `.venv/bin/python -m experiments.v2_pairwise_superposition.assess_balanced`.','', 'Initialization equality was checked against deterministic reconstruction using the baseline seed and constructor; the first run did not save its initial tensor state, so direct comparison with an archived original initialization is unavailable. The dataset was reused by path and checksum rather than regenerated. Protected V1 and original V2 checkpoints were verified unchanged by SHA-256.','', 'Specification note: SPEC.md’s status still says no V2 results, and its first-run objective remains unweighted MSE. That historical status is stale; this approved objective-only trial is documented separately and does not silently rewrite those scientific assumptions.']
    p.write_text(s+'\n'.join(extra)+'\n')
    write_json(RUN/'assessment_provenance.json',{'report_sha256':digest(p),'assessment_source_sha256':digest(BASE/'assess_balanced.py'),'inference_performed':False})
    print(p)
if __name__=='__main__':main()
