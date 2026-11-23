"""Three-width scientific reporting from saved training/validation diagnostics."""
import json
import numpy as np
from .capacity256 import BASE,OUT,RUN,CK,OLD,PREVIOUS,NEW,WIDTH_RUNS
from .data import RULES,PAIRS
from .run import digest,write_csv
from src.runtime import write_json


def produce(d,mets,dist,probes,interactions):
    plots(d,mets,dist,interactions)
    fmt=lambda v:'—' if v is None else f'{v:.5f}'
    def select(width,j,regime='all'):
        return next(r for r in interactions if r['width']==width and r['indices']==str(PAIRS[j]) and r['rule']==RULES[j] and r['regime']==regime)
    training=json.loads((RUN/'training_complete.json').read_text());checks=json.loads((OUT/'interaction_checks.json').read_text())
    lines=['# Final V2 teacher capacity trial: width 256','','## Frozen protocol','','One 12→256→12 ReLU teacher, original unweighted mean full-output squared error against Qg. The only YAML change from width128 is dimensions.hidden=256. Exact original P(12×20), Q(12×9), 20,000-vector NPZ and split manifest are reused. Seed3105, Adam, cosine LR .001→.00001, batch256, budget2000, patience200/min_delta1e-7, fixed threshold0.5, and minimum validation full-output-MSE checkpoint selection are unchanged. No additional widths/seeds/losses or transcoder were trained. SPEC v0.3 retains its original first-run width64; this isolated override is authorized by the researcher.','','Seed3105 and standard nn.Linear initialization are recorded in initial.pt. Different tensor dimensions change RNG consumption and W2 fan-in scaling, so initial weights are not identical or nested across widths. The initialization, minibatch RNG sequence and LR sequence are verified by reconstruction after training. Width256 has 6,412 parameters (width128:3,212; width64:1,612).','','Historical test metrics do not enter this report or selection. Old-test vector identities only exclude overlapping controlled probes. The reserved final population is not opened or scored. Prior artifacts are preserved and verified by SHA-256. The width64/128 ordinary predictions and signed decompositions are read from frozen caches; only width256 is newly scored on train/validation and the already inspected probe bank.','','## Execution','','CUDA: NVIDIA A100-SXM4-40GB, PyTorch2.9.1+cu128. No CPU fallback or dependency installation. The named tmux job and full stdout/stderr are under ../../execution_logs/'+NEW+'/.',f"Training completed {training['epochs']} epochs, selected epoch {training['best_epoch']}, elapsed {training['elapsed_this_invocation_seconds']:.2f}s, stop reason {training['stop_reason']}. New width256 signed-sum/direct-network maximum residual: {checks['new_width_max_sum_vs_direct_forward_error']:.3g}.",'','## Train/validation aggregate performance','','| Width | Population | Exact-vector accuracy | Full-output MSE | Semantic MSE |','|---:|---|---:|---:|---:|']
    for split in ('train','validation'):
        for width in (64,128,256):
            m=mets[f'w{width}_{split}'];lines.append(f"| {width} | {split} | {m['exact_vector_accuracy']:.5f} | {m['output_mse']:.7f} | {m['semantic_mse']:.7f} |")
    for split in ('train','validation'):
        lines+=['',f'## Primitive rules: {split}','','| Rule | Width | Precision | Recall | FPR | Balanced accuracy | Semantic MSE | FP / FN |','|---|---:|---:|---:|---:|---:|---:|---|']
        for j in range(8):
            for width in (64,128,256):
                r=mets[f'w{width}_{split}']['per_rule'][j];lines.append('| '+RULES[j]+' | '+str(width)+' | '+' | '.join(fmt(r[k]) for k in ('precision','recall','false_positive_rate','balanced_accuracy','mse'))+f" | {r['fp']} / {r['fn']} |")
    lines+=['','## DNF output (secondary; excluded from primitive acceptance)','','| Width / population | Precision | Recall | FPR | Balanced accuracy | Semantic MSE |','|---|---:|---:|---:|---:|---:|']
    for split in ('train','validation'):
        for width in (64,128,256):
            r=mets[f'w{width}_{split}']['per_rule'][8];lines.append(f'| {width} / {split} | '+' | '.join(fmt(r[k]) for k in ('precision','recall','false_positive_rate','balanced_accuracy','mse'))+' |')
    lines+=['','## Pairwise nonlinear computation on identical contexts','','All four input corners are identical for all widths, including the same distractors. The bank is precisely the width128 bank with entire historical-test-overlap contexts excluded. Full contributions are retained for every semantic output and neuron; no absolute top-k truncation. Shared-input triple (0,1,2) is also retained as a secondary diagnostic. These are exact algebraic decompositions, not recovered causal circuits.','','| Primitive | Width | Contexts | Target interaction MAE | Positive sum | Negative sum | Net | Cancellation |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for j in range(8):
        for width in (64,128,256):
            r=select(width,j);lines.append(f"| {RULES[j]} | {width} | {r['n_contexts']} | "+' | '.join(fmt(r[k]) for k in ('target_interaction_mae','positive','negative','net','cancellation'))+' |')
    lines+=['','## Absolute and normalized contribution concentration','','For contributions c_r, N_eff=(sum|c|)²/sum(c²); normalized participation=N_eff/H. Entropy count=exp(−sum p log p), p=|c|/sum|c|; its normalized count divides by H. Cancellation=1−|sum c|/sum|c|. Zero interactions use zero counts/shares/cancellation. Nonzero counts use exact floating nonzero, not a fitted significance cutoff. Means are over identical contexts. Absolute and normalized counts answer different questions: more contributing neurons need not imply a larger fraction of the network. Splitting one contribution among duplicate units can increase effective counts without introducing a new computation.','','| Primitive | Width | N_eff | N_eff/H | Entropy count/H | Top1 abs share | Top5 abs share | Top1 signed residual MAE |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for j in range(8):
        for width in (64,128,256):
            r=select(width,j);lines.append(f'| {RULES[j]} | {width} | '+' | '.join(fmt(r[k]) for k in ('effective_units','effective_fraction_of_width','entropy_effective_fraction_of_width','top1_absolute_share','top5_absolute_share','top1_signed_residual'))+' |')
    lines+=['','Top1 is chosen separately per context and is an optimistic descriptive approximation, not a fixed proposed neuron circuit. Negative sums and top1 signed residuals reveal cancellation hidden by absolute shares. Neuron IDs are local to each model; identities are not aligned across widths. All neurons and all signed context contributions are available in the CSV/NPZ archives.','','## Density-supported probes versus OOD stress','','Density3–6 is within the observed density support, not necessarily an IID draw from the generator. Constructed memberships repeat input vectors and were previously inspected. None are new unbiased held-out evidence or checkpoint selectors. The final reserved population remains unscored.','','| Width / constructed regime | Memberships | Exact-vector accuracy | Full-output MSE |','|---|---:|---:|---:|']
    for label,m in probes.items():lines.append(f"| {label} | {m['n']} | {m['exact_vector_accuracy']:.5f} | {m['output_mse']:.7f} |")
    lines+=['','For interaction comparisons, every corner must be density3–6 to enter the supported subset; if any corner falls outside, the context belongs to the OOD-stress subset. There are only 3–5 complete density-supported contexts per primitive after historical-test exclusions; report these as limited diagnostics. All-context values above pool these regimes.','','| Primitive | Context regime | N | MAE width64 | MAE width128 | MAE width256 |','|---|---|---:|---:|---:|---:|']
    for j in range(8):
        for regime in ('every_corner_density_3_to_6','some_corner_OOD_density'):
            rr=[select(w,j,regime) for w in (64,128,256)];lines.append(f"| {RULES[j]} | {regime} | {rr[0]['n_contexts']} | "+' | '.join(fmt(r['target_interaction_mae']) for r in rr)+' |')
    lines+=['','## Training fit versus generalization','','| Primitive | Width | Train FN rate | Validation FN rate | Train BA | Validation BA |','|---|---:|---:|---:|---:|---:|']
    for j in range(8):
        for width in (64,128,256):
            t=mets[f'w{width}_train']['per_rule'][j];v=mets[f'w{width}_validation']['per_rule'][j]
            lines.append(f'| {RULES[j]} | {width} | '+' | '.join(fmt(z) for z in [t['false_negative_rate'],v['false_negative_rate'],t['balanced_accuracy'],v['balanced_accuracy']])+' |')
    old=mets['w128_validation'];new=mets['w256_validation'];newtrain=mets['w256_train'];primitives=new['per_rule'][:8];passing=[r['rule'] for r in primitives if r['balanced_accuracy']>=.99];failing=[r['rule'] for r in primitives if r['balanced_accuracy']<.99]
    fitfail=[r['rule'] for r in newtrain['per_rule'][:8] if r['balanced_accuracy']<.99]
    improved=[r['rule'] for r,b in zip(primitives,old['per_rule']) if r['balanced_accuracy']>b['balanced_accuracy']]
    lines+=['','## Scientific assessment and recommendation','',f"Width256 versus width128 validation exact-vector accuracy: {old['exact_vector_accuracy']:.4%}→{new['exact_vector_accuracy']:.4%}; full-output MSE {old['output_mse']:.7f}→{new['output_mse']:.7f}; semantic MSE {old['semantic_mse']:.7f}→{new['semantic_mse']:.7f}. The relative full-output MSE change is {(new['output_mse']/old['output_mse']-1)*100:+.2f}%. Validation balanced accuracy improves for: {', '.join(improved) or 'none'}. Regression MSE is the checkpoint-selection criterion; it is distinct from exact-vector accuracy and rule recall.",f"\nThe unchanged ≥99% validation balanced-accuracy gate is met by {len(passing)}/8 primitives: {', '.join(passing) or 'none'}. Rules below the gate: {', '.join(failing) or 'none'}. DNF is evaluated separately. No rounding, averaging, or threshold adjustment is used to pass the gate.",f"\nTraining balanced accuracy is also below99% for {', '.join(fitfail) or 'none'}. Train exact-vector error is {1-newtrain['exact_vector_accuracy']:.3%}, versus validation {1-new['exact_vector_accuracy']:.3%}. Compare per-rule counts and rates above rather than raw counts across unequal population sizes. Persistent training errors indicate incomplete training fit; a validation gap can add generalization error. These observational comparisons cannot uniquely identify optimization versus representation or objective allocation."]
    effective_up=sum(select(256,j)['effective_units']>select(128,j)['effective_units'] for j in range(8));normalized_up=sum(select(256,j)['effective_fraction_of_width']>select(128,j)['effective_fraction_of_width'] for j in range(8));top1_down=sum(select(256,j)['top1_absolute_share']<select(128,j)['top1_absolute_share'] for j in range(8))
    lines+=[f"\nCompared with width128, absolute effective contribution count increases for {effective_up}/8 primitive interactions, normalized N_eff/H increases for {normalized_up}/8, and top1 absolute share decreases for {top1_down}/8. Interpret the full rule-specific table: width alone is not a demonstration of distributed circuitry. Signed cancellation and nonzero single-neuron residuals distinguish a dominant contribution from a sufficient one-neuron explanation. Exact sums are correctness checks, not causal discoveries.",'','ASSESSMENT_DETAIL_PENDING','',('**Recommendation: width256 passes the validation primitive gate and is a candidate for the first frozen-teacher transcoder experiment, subject to scientific review of the controlled-context limitations.**' if not failing else '**Recommendation: width256 does not meet the unchanged reliable-all-primitive criterion, so it is not yet accepted for the intended first transcoder experiment under that criterion.**'),'','No independent final-generalization claim is made. Selection has used an adaptively inspected validation population across trials; historical controlled probes are also reused. Do not treat either as a fresh final test. No transcoder is trained here and no further width increase is proposed or launched.','','## Artifacts and commands','','[Performance](performance.png) · [Learning curves](learning_curves.png) · [Interaction concentration](concentration.png) · [Interaction fidelity](interaction_fidelity.png) · [Validation score CDFs](score_cdfs.png)','','- metrics.json / per_rule.csv / score_distributions.json: all three train/validation metrics, class support, FN/positive/negative score quantiles.','- signed_interactions_all_widths.npz: complete signed neuron contributions, shared corners, ground truth, direct-output values and residuals.','- interaction_summary.csv / interaction_contexts.csv / all_neuron_contributions.csv: positive/negative sums, cancellation, absolute and normalized participation, all neuron identities.','- controlled_predictions_all_widths.npz / controlled_metrics.json / w*_groups_*.json: diagnostic corner and density/distractor groups.','- hidden_train.npz / hidden_activation_summary.json: width256 original hidden activations and prevalence.','- ../../checkpoints/'+NEW+'/: initial.pt, best.pt, last.pt.','- ../../runs/'+NEW+'/: copied YAML, source/test/spec snapshots, provenance, histories, checkpoint hashes and completion record.','- ../../execution_logs/'+NEW+'/: tests.log, execute.sh, execution.log, exit_status.','','Executed CUDA training/evaluation command, inside tmux `v2-capacity256`:','','```bash','CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 .venv/bin/python -m experiments.v2_pairwise_superposition.capacity256','```','','Test command:','','```bash','CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest experiments/v2_pairwise_superposition/tests -q','```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(RUN/'report_generation.json',{'report_sha256':digest(OUT/'report.md'),'source_sha256':digest(BASE/'capacity256_report.py')})


def plots(d,mets,dist,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for width in (64,128,256):
        r=mets[f'w{width}_validation']['per_rule']
        for ax,key in zip(axes,('balanced_accuracy','recall','false_positive_rate')):ax.plot([v[key] for v in r[:8]],'o-',label=str(width))
    for ax,title in zip(axes,('Primitive validation BA','Primitive validation recall','Primitive validation FPR')):ax.set_title(title);ax.set_xticks(range(8),[f'g{j}' for j in range(8)]);ax.legend(title='Width')
    axes[0].axhline(.99,ls='--',color='black',lw=.7);fig.tight_layout();fig.savefig(OUT/'performance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for width,run in WIDTH_RUNS.items():
        h=json.loads((BASE/'runs'/run/'history.json').read_text());x=[r['epoch'] for r in h]
        axes[0].plot(x,[r['train_online_mse'] for r in h],label=str(width));axes[1].plot(x,[r['validation_mse'] for r in h],label=str(width))
    for ax,title in zip(axes,('Online training full-output MSE','Validation full-output MSE')):ax.set_title(title);ax.set_yscale('log');ax.set_xlabel('Epoch');ax.legend(title='Width')
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4));fig2,ax2=plt.subplots(figsize=(8,4))
    for width in (64,128,256):
        rr=[next(r for r in rows if r['width']==width and r['indices']==str(PAIRS[j]) and r['rule']==RULES[j] and r['regime']=='all') for j in range(8)]
        for ax,key in zip(axes,('effective_units','effective_fraction_of_width','cancellation')):ax.plot([r[key] for r in rr],'o-',label=str(width))
        ax2.plot([r['target_interaction_mae'] for r in rr],'o-',label=str(width))
    for ax,title in zip(axes,('Absolute effective unit count','Effective count / hidden width','Signed cancellation')):ax.set_title(title);ax.set_xticks(range(8),[f'g{j}' for j in range(8)]);ax.legend(title='Width')
    fig.tight_layout();fig.savefig(OUT/'concentration.png',dpi=160);plt.close(fig)
    ax2.set_title('Pairwise target-interaction MAE: all matched contexts');ax2.set_xticks(range(8),[f'g{j}' for j in range(8)]);ax2.legend(title='Width');fig2.tight_layout();fig2.savefig(OUT/'interaction_fidelity.png',dpi=160);plt.close(fig2)
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for j,ax in enumerate(axes.flat):
        g=d['g'][d['validation'],j].astype(bool)
        for width,run in WIDTH_RUNS.items():
            with np.load(BASE/'reports'/run/'predictions_validation.npz') as f:s=f['semantic'][:,j]
            for positive,style in ((True,'-'),(False,'--')):
                v=np.sort(s[g if positive else ~g]);ax.plot(v,np.arange(1,len(v)+1)/len(v),ls=style,label=f'{width} '+('positive' if positive else 'negative'))
        ax.axvline(.5,color='black',lw=.6);ax.set_title(RULES[j]);ax.set_ylabel('Empirical CDF')
    axes.flat[0].legend(fontsize=6);fig.tight_layout();fig.savefig(OUT/'score_cdfs.png',dpi=160);plt.close(fig)
