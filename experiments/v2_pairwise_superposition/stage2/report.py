"""Research report and plots from the six completed validation experiments."""
import json,shutil
from pathlib import Path
import numpy as np
from src.runtime import write_json,load_checkpoint
from ..run import BASE,ROOT,digest,write_csv
from ..data import RULES
from .run import C,ID,RUN,OUT,CK,TEACHER,verify_frozen
from .core import name,pareto
from .evaluate import CASES


def load():
    rows=[];detail={}
    for width in C['widths']:
        for coefficient in C['lambdas']:
            key=name(width,coefficient);folder=OUT/key
            if not (folder/'complete.json').exists():raise RuntimeError('Configuration incomplete: '+key)
            m=json.loads((folder/'metrics.json').read_text());t=json.loads((CK/key/'training_complete.json').read_text());interactions=read_csv(folder/'interaction_summary.csv')
            v=m['validation'];f=v['teacher_fidelity'];a=v['activation']
            rows.append({'name':key,'width':width,'lambda':coefficient,'selected_epoch':t['selected_epoch'],'epochs':t['epochs'],'mse':f['full_output_mse'],'variance_normalized_mse':f['variance_normalized_mse'],'semantic_mse':f['semantic_mse'],'agreement':f['exact_vector_agreement'],'teacher_correct_agreement':f['teacher_correct']['exact_vector_agreement'],'teacher_incorrect_agreement':f['teacher_incorrect']['exact_vector_agreement'],'teacher_incorrect_mse':f['teacher_incorrect']['full_output_mse'],'l0':a['mean_l0'],'l0_fraction':a['mean_l0_fraction'],'l1_sum':a['mean_l1_sum'],'dead_train':m['train']['activation']['dead_on_population'],'dead_validation':a['dead_on_population'],'student_task_accuracy':v['student_task']['exact_vector_accuracy']})
            detail[key]={'metrics':m,'training':t,'interactions':interactions,'candidates':json.loads((folder/'candidate_features.json').read_text()),'topk':json.loads((folder/'fixed_topk_approximation.json').read_text()),'controlled':json.loads((folder/'controlled_fidelity.json').read_text())}
    return rows,detail


def read_csv(path):
    import csv
    return list(csv.DictReader(path.open()))


def row_for(detail,key,pi,j,regime='all'):
    return next(r for r in detail[key]['interactions'] if int(r['pair_index'])==pi and int(r['output_index'])==j and r['regime']==regime)


def matched_context_stability(rows):
    output=[]
    for row in rows:
        with np.load(OUT/row['name']/'signed_interactions.npz') as f:
            for pi,j,case in CASES:
                a=f[f'p{pi}_corners'];teacher=f[f'p{pi}_teacher_value'][:,j];student=f[f'p{pi}_student_value'][:,j]
                for c0 in range(len(a)):
                    for c1 in range(c0+1,len(a)):
                        if np.array_equal(a[c0,:,:15],a[c1,:,:15]) and not np.array_equal(a[c0,:,15:],a[c1,:,15:]):
                            delta_t=teacher[c1]-teacher[c0];delta_s=student[c1]-student[c0]
                            output.append({'config':row['name'],'case':case,'context0':c0,'context1':c1,'teacher_interaction_change':float(delta_t),'student_interaction_change':float(delta_s),'change_residual':float(delta_s-delta_t)})
    write_csv(OUT/'matched_distractor_interaction_stability.csv',output)
    return output


def plots(rows,detail,frontier):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for width,color in [(512,'C0'),(1024,'C1')]:
        r=[v for v in rows if v['width']==width];ax.scatter([v['l0'] for v in r],[v['mse'] for v in r],color=color,label=f'width {width}',s=65)
        for v in r:ax.annotate(f"λ={v['lambda']:g}",(v['l0'],v['mse']),xytext=(5,5),textcoords='offset points',fontsize=8)
    r=sorted([v for v in rows if v['name'] in frontier],key=lambda v:v['l0']);ax.plot([v['l0'] for v in r],[v['mse'] for v in r],'k--',alpha=.5,label='Pareto frontier')
    ax.set_xlabel('Validation mean active units (strict ReLU L0)');ax.set_ylabel('Validation teacher reconstruction MSE / coordinate');ax.set_yscale('log');ax.legend();fig.tight_layout();fig.savefig(OUT/'pareto.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for key,color in [(r['name'],f'C{i}') for i,r in enumerate(rows)]:
        h=json.loads((CK/key/'history.json').read_text());axes[0].plot([v['epoch'] for v in h],[v['validation']['mse'] for v in h],label=key,color=color)
        axes[1].plot([v['epoch'] for v in h],[v['validation']['mean_l0'] for v in h],color=color)
    axes[0].set_yscale('log');axes[0].set_title('Validation teacher reconstruction');axes[0].legend(fontsize=6);axes[1].set_title('Validation mean L0')
    x=np.arange(len(rows));axes[2].bar(x,[r['agreement'] for r in rows]);axes[2].set_ylim(0,1);axes[2].set_xticks(x,[r['name'] for r in rows],rotation=65,fontsize=7);axes[2].set_title('Exact-vector agreement with teacher')
    fig.tight_layout();fig.savefig(OUT/'training_and_agreement.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,(pi,j,case) in zip(axes,CASES):
        ax.bar(np.arange(len(rows)),[float(row_for(detail,r['name'],pi,j)['interaction_mae']) for r in rows]);ax.set_title(case+' interaction MAE vs teacher');ax.set_xticks(np.arange(len(rows)),[r['name'] for r in rows],rotation=65,fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'case_interaction_residuals.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for col,(pi,j,case) in enumerate(CASES):
        ref=row_for(detail,rows[0]['name'],pi,j)
        for rr,key,title in [(0,'effective_units','Absolute effective units'),(1,'effective_fraction','Effective count / width')]:
            values=[float(row_for(detail,r['name'],pi,j)['student_'+key]) for r in rows]
            axes[rr,col].bar(np.arange(len(rows)),values);axes[rr,col].axhline(float(ref['teacher_'+key]),ls='--',color='red',label='Original teacher');axes[rr,col].set_title(case+': '+title);axes[rr,col].set_xticks(np.arange(len(rows)),[r['name'] for r in rows],rotation=65,fontsize=7);axes[rr,col].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'interaction_concentration.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for row in rows:
        a=detail[row['name']]['metrics']['validation']['activation'];axes[0].plot(np.sort(a['per_feature_prevalence']),label=row['name']);axes[1].plot(np.sort(a['per_feature_mean']),label=row['name'])
    axes[0].set_title('Sorted per-feature activation prevalence');axes[1].set_title('Sorted per-feature mean activation');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'feature_utilization.png',dpi=170);plt.close(fig)


def main():
    verification=verify_frozen();rows,detail=load();frontier=pareto(rows);best=min(rows,key=lambda r:r['mse']);sparse=min([r for r in rows if r['name'] in frontier],key=lambda r:r['l0']);examples=list(dict.fromkeys([best['name'],sparse['name']]))
    write_json(OUT/'selection.json',{'checkpoint_rule':'minimum validation teacher full-output MSE per configuration','pareto_rule':'nondominated validation MSE and mean L0','pareto_frontier':frontier,'best_validation_fidelity':best['name'],'report_case_detail_endpoints':examples,'note':'Endpoints illustrate tradeoffs; sparsest endpoint is not declared the winner; no test/contrast selection'})
    write_csv(OUT/'validation_comparison.csv',rows);stability=matched_context_stability(rows);plots(rows,detail,frontier)
    affine=json.loads((OUT/'affine_baseline.json').read_text());teacher=json.loads((OUT/'teacher_validation_task.json').read_text());fmt=lambda x:'—' if x is None else f'{float(x):.6g}'
    lines=['# V2 Stage 2: frozen-teacher sparse transcoder pilot','','## Scientific scope and verified teacher','','The researcher explicitly approved the imperfect width256 teacher for this mechanistic research pilot. The goal is to imitate its actual full continuous outputs, including errors, before investigating the original-neuron mechanism. This does not retroactively satisfy the original99% all-primitive task criterion (3/8 primitives passed). No original-neuron interventions, teacher fine-tuning, weight edits, extra models or final-population scoring occurred.','',f"Approved checkpoint: `{verification['checkpoint']}`. SHA-256: `{verification['checkpoint_sha256']}`. Architecture12→256→12 ReLU; teacher seed3105, selected epoch{verification['selected_epoch']}. All teacher parameters are frozen and eval mode verified. Dataset SHA-256 `{verification['dataset_sha256']}`, manifest SHA-256 `{verification['split_manifest_sha256']}`.",f"\nCUDA re-inference reproduced saved train/validation outputs and task metrics within recorded tolerances. Validation Boolean-task exact-vector accuracy {teacher['exact_vector_accuracy']:.6%}, full-output MSE {teacher['output_mse']:.9f}, semantic MSE {teacher['semantic_mse']:.9f}. These verify the teacher; they are not transcoder supervision targets.",'','## Predeclared protocol and reductions','','Exactly widths512/1024 × lambda0/1e-4/1e-3. Initialization seed4105 and CPU shuffle seed4106; each lambda at a fixed width starts from identical weights and ordering. Different widths have differently shaped initial weights. Adam, lr.001→.00001 cosine, batch256, budget2000epochs, patience200/min_delta1e-8 on validation teacher reconstruction MSE. Decoder columns are normalized to unit L2 after initialization and every optimizer step. No dead-feature resampling or auxiliary labels. Both GPUs were verified idle before running one width worker per GPU, with its three penalties sequential.','','Optimization loss = mean_examples sum_12_outputs (T−F)^2 + lambda mean_examples sum_units |z|. Reported full-output MSE averages over examples AND12 output coordinates, so the reported MSE is reconstruction loss divided by12. L1 is summed, not averaged, over width. Validation unpenalized teacher reconstruction MSE alone selects checkpoints. Ground truth g and Q never enter the fit function. Normalized MSE divides coordinate-mean reconstruction error by pooled per-coordinate validation teacher variance about its own coordinate mean; 1−this ratio is variance-weighted explained variance, not classification agreement.','','## All six completed configurations','','| Configuration | Epochs / selected | Val teacher MSE | Normalized MSE | Semantic fidelity MSE | Exact teacher agreement | Mean L0 | L0 / width | Dead train / val | Pareto |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['name']} | {r['epochs']} / {r['selected_epoch']} | "+' | '.join(fmt(r[k]) for k in ('mse','variance_normalized_mse','semantic_mse','agreement','l0','l0_fraction'))+f" | {r['dead_train']} / {r['dead_validation']} | {'yes' if r['name'] in frontier else 'no'} |")
    lines+=['',f"The complete nondominated validation MSE/L0 frontier is: {', '.join(frontier)}. Minimum validation reconstruction MSE is achieved by {best['name']}. No arbitrary sparsest configuration is declared best; case-detail endpoints below illustrate both fidelity and sparsity.",'','## Affine and zero-L1 controls','','Affine regression uses a CUDA float64 least-squares fit of frozen training teacher outputs on the12D input plus intercept (rank13); no Boolean targets or validation fitting. An affine input-output map has analytically zero pairwise finite differences. Thus low-ish output error on common examples would not reproduce the teacher’s nonzero finite interactions.','','| Model | Train teacher MSE | Validation teacher MSE | Validation normalized MSE | Validation exact teacher agreement |','|---|---:|---:|---:|---:|']
    for label,train,v in [('affine',affine['train']['teacher_fidelity'],affine['validation']['teacher_fidelity'])]+[(r['name'],detail[r['name']]['metrics']['train']['teacher_fidelity'],detail[r['name']]['metrics']['validation']['teacher_fidelity']) for r in rows if r['lambda']==0]:
        lines.append(f"| {label} | "+' | '.join(fmt(x) for x in [train['full_output_mse'],v['full_output_mse'],v['variance_normalized_mse'],v['exact_vector_agreement']])+' |')
    lines+=['','## Teacher agreement versus Boolean-task accuracy','','Teacher validation task accuracy is fixed at97.5667%. A student may correct a teacher error or introduce another error; either is a disagreement with the imitation target. Boolean-task metrics are diagnostic and do not select students. Full train/validation per-output task precision/recall and fidelity residuals are in each metrics.json.','','| Config | Teacher agreement: teacher-correct rows | Teacher agreement: teacher-incorrect rows | MSE on teacher-correct rows | MSE on teacher-incorrect rows | Student Boolean exact accuracy |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        correct_mse=detail[r['name']]['metrics']['validation']['teacher_fidelity']['teacher_correct']['full_output_mse']
        lines.append('| '+r['name']+' | '+' | '.join(fmt(v) for v in (r['teacher_correct_agreement'],r['teacher_incorrect_agreement'],correct_mse,r['teacher_incorrect_mse'],r['student_task_accuracy']))+' |')
    lines+=['','## Per-output validation fidelity','','| Config | Output | Teacher/student agreement | Semantic residual MSE | Signed residual mean | Residual MAE | Teacher errors preserved |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        for v in detail[r['name']]['metrics']['validation']['teacher_fidelity']['per_output']:lines.append('| '+r['name']+' | '+v['rule']+' | '+' | '.join(fmt(v[k]) for k in ('agreement','semantic_mse','signed_residual_mean','residual_mae','teacher_errors_preserved_fraction'))+' |')
    lines+=['','## Exact nonlinear interaction fidelity','','Every pair uses all four fixed Boolean corners with the same context for teacher and student. Full signed contributions for every original neuron and every transcoder feature are retained, including negative contributions and cancellation. Teacher direct forward differences are rechecked against the frozen native decomposition. Transcoder feature sums are checked against direct output finite differences in float64 with atol1e-9/rtol1e-8. Biases and ReLU gates remain in both computations; A@K or D2_eff@C1 alone is not the computation.','','The reused context bank is historical constructed diagnostic evidence, not an unbiased evaluation population and not a checkpoint selector. Density-supported means every corner has3–6 active inputs; it does not mean IID. There are only3–5 density-supported complete contexts per primitive, with more OOD contexts. Every historical-test-overlap context is excluded. The reserved final population is untouched.','','| Case / config | Regime | N contexts | Interaction MAE vs teacher | Teacher effective units | Student effective units | Teacher fraction | Student fraction |','|---|---|---:|---:|---:|---:|---:|---:|']
    for pi,j,case in CASES:
        for r in rows:
            for regime in ('all','density_supported','OOD_density'):
                v=row_for(detail,r['name'],pi,j,regime);lines.append(f"| {case} / {r['name']} | {regime} | {v['n_contexts']} | "+' | '.join(fmt(v[k]) for k in ('interaction_mae','teacher_effective_units','student_effective_units','teacher_effective_fraction','student_effective_fraction'))+' |')
    lines+=['','## Signed cancellation and concentration','','N_eff=(sum|c|)^2/sum(c²), normalized by total available width; top-k share uses absolute contribution mass and must be read alongside signed sums and residuals. Mean L0 per example is a separate property. None proves semantic interpretability. All-context averages below mix density regimes; full per-context and regime-separated rows are retained.','','| Case / config | Teacher positive / negative | Student positive / negative | Teacher cancellation | Student cancellation | Teacher top1 / top5 share | Student top1 / top5 share |','|---|---|---|---:|---:|---|---|']
    for pi,j,case in CASES:
        for r in rows:
            v=row_for(detail,r['name'],pi,j);lines.append(f"| {case} / {r['name']} | {fmt(v['teacher_positive_sum'])} / {fmt(v['teacher_negative_sum'])} | {fmt(v['student_positive_sum'])} / {fmt(v['student_negative_sum'])} | {fmt(v['teacher_cancellation'])} | {fmt(v['student_cancellation'])} | {fmt(v['teacher_top1_share'])} / {fmt(v['teacher_top5_share'])} | {fmt(v['student_top1_share'])} / {fmt(v['student_top5_share'])} |")
    lines+=['','## Context dependence and imperfect teacher cases','','Per-context interaction_contexts.csv identifies complete corner correctness both for all outputs and the relevant output, without changing contexts. Each interaction_summary.csv separates teacher-correct and teacher-incorrect relevant-output corners. SD across all contexts includes semantic-background changes; matched nuisance-only changes are separately exported in matched_distractor_interaction_stability.csv. Ordinary teacher-error rows and constructed corner failures are different populations.','','| Case / config | Teacher-correct output corners: N / MAE | Teacher-imperfect output corners: N / MAE | Context residual SD |','|---|---|---|---:|']
    for pi,j,case in CASES:
        for r in rows:
            base=row_for(detail,r['name'],pi,j);cells=[]
            for regime in ('teacher_correct_output_corners','teacher_incorrect_output_corners'):
                match=[v for v in detail[r['name']]['interactions'] if int(v['pair_index'])==pi and int(v['output_index'])==j and v['regime']==regime];cells.append(f"{match[0]['n_contexts']} / {fmt(match[0]['interaction_mae'])}" if match else '0 / undefined')
            lines.append(f"| {case} / {r['name']} | "+' | '.join(cells)+f" | {fmt(base['residual_context_sd'])} |")
    lines+=['','## Full-output agreement on constructed diagnostics','','These memberships are reused and may repeat vectors. Density support is distinct from IID generalization. These scores never select checkpoints.','','| Config | Density regime | Teacher reconstruction MSE | Exact teacher agreement |','|---|---|---:|---:|']
    for r in rows:
        for regime,values in detail[r['name']]['controlled'].items():
            f=values['teacher_fidelity'];lines.append(f"| {r['name']} | {regime} | {fmt(f['full_output_mse'])} | {fmt(f['exact_vector_agreement'])} |")
    lines+=['','## Fixed small-feature explanations (descriptive)','','Fixed unit sets of sizes1/3/5/10 are selected by mean absolute interaction contribution on this reused bank. They are not evaluated as unbiased held-out hypotheses or causal interventions. Their approximation residual includes omitted positive and negative contributions; full signed decompositions are never replaced by this truncation.','','| Case / config | k | Fixed units | Subset versus full student MAE | Subset versus teacher MAE |','|---|---:|---|---:|---:|']
    for key in examples:
        for r in detail[key]['topk']:lines.append(f"| {r['case']} / {key} | {r['k']} | {r['fixed_units']} | {fmt(r['subset_vs_student_interaction_mae'])} | {fmt(r['subset_vs_teacher_interaction_mae'])} |")
    lines+=['','Original-neuron reference under the identical descriptive selection rule:','','| Original case | k | Fixed original neurons | Subset versus full teacher MAE |','|---|---:|---|---:|']
    reference=[]
    with np.load(OUT/rows[0]['name']/'signed_interactions.npz') as f:
        for pi,j,case in CASES:
            contributions=f[f'p{pi}_teacher_contributions'][:,j,:];value=f[f'p{pi}_teacher_value'][:,j];order=np.argsort(-np.abs(contributions).mean(0))
            for count in (1,3,5,10):
                units=order[:count];mae=float(np.abs(contributions[:,units].sum(1)-value).mean());reference.append({'case':case,'k':count,'original_neurons':units.astype(int).tolist(),'mae':mae});lines.append(f"| {case} | {count} | {units.tolist()} | {fmt(mae)} |")
    write_json(OUT/'original_fixed_topk_reference.json',reference)
    lines+=['','## Candidate feature case studies','','Detailed candidates for all six models are in each candidate_features.json: complete20 input coefficients K, encoder bias, all9 signed output writes A, every four-corner activation in every context, signed interaction contributions, gate-pattern counts and matched nuisance changes. Candidate selection is the union of top3 mean positive, negative and absolute contributions, not a detector label. Here the fidelity-optimal and sparsest Pareto endpoint are shown to expose tradeoffs, not to crown the sparsest model. Coefficient vectors below list every coordinate in a0..a19/g0..g8 order, rounded for reading; JSON preserves full precision.']
    for pi,j,case in CASES:
        lines+=['',f'### {case}']
        for key in examples:
            candidates=[v for v in detail[key]['candidates']['cases'] if v['case']==case]
            if not candidates:
                lines.append(f'\n{key}: no feature exceeds the declared 1e-12 contribution reporting floor for this case. This is a failure to obtain a nonzero candidate on the bank, not evidence of a recovered rule.')
                continue
            pos=max(candidates,key=lambda v:v['positive_mean']);neg=min(candidates,key=lambda v:v['negative_mean']);chosen={v['unit']:v for v in (pos,neg) if v['mean_absolute_contribution']>1e-12}
            for unit,v in chosen.items():
                patterns=v['activation_gate_pattern_counts'];majority=max(patterns,key=patterns.get);count=patterns[majority]
                observation='Conditional 11-only gate on this bank' if majority=='0001' else ('Conditional 00-only gate on this bank' if majority=='1000' else 'Context-conditioned ReLU hinge component')
                lines += ['',f"**{key}, feature {unit}.** Mean signed interaction contribution {v['mean_signed_contribution']:.6g}; mean absolute contribution {v['mean_absolute_contribution']:.6g}. {observation}; most common gate pattern `{majority}` occurs in {count}/{sum(patterns.values())} contexts (corner order00,01,10,11). This is not sufficient to call it an AND/XOR detector.",f"\nK[20] = `{[float(f'{x:.6g}') for x in v['K_complete_20_inputs']]}`",f"\nEncoder bias = `{v['encoder_bias']:.9g}`",f"\nA[9] = `{[float(f'{x:.6g}') for x in v['A_complete_9_outputs']]}`",f"\nMean corner activations = `{[float(f'{x:.6g}') for x in v['mean_corner_activations']]}`; SD across contexts = `{[float(f'{x:.6g}') for x in v['corner_activation_sd_across_contexts']]}`. All individual contexts/corners are retained in JSON."]
                changes=v['matched_distractor_context_changes']
                if changes:lines.append(f"\nAcross {len(changes)} matched nuisance-only context pairs, maximum absolute change in this feature’s signed interaction is {max(abs(x['signed_interaction_change']) for x in changes):.6g}. Such sensitivity must be distinguished from a context-independent semantic gate.")
    lines+=['','### Shared input a0 across AND(0,1) and AND(0,2)','','Reuse candidates are ranked by geometric mean of the same transcoder unit’s mean absolute contribution to the two distinct pair/output relationships. Full coefficients and both writes are retained. This measures shared nonlinear contribution usage rather than just a large a0 input weight; it is still descriptive and does not imply one original-neuron mechanism.','','| Config | Feature | K_a0 | A_g0 | A_g1 | Mean abs contribution to AND01→g0 | Mean abs contribution to AND02→g1 |','|---|---:|---:|---:|---:|---:|---:|']
    for key in examples:
        for v in detail[key]['candidates']['reuse_candidates'][:5]:lines.append('| '+key+' | '+str(v['unit'])+' | '+' | '.join(fmt(v[k]) for k in ('K_a0','A_g0','A_g1','mean_abs_and01_to_g0_contribution','mean_abs_and02_to_g1_contribution'))+' |')
    lines+=['','## Scientific assessment','','ASSESSMENT_PENDING','','## Reproducibility, limitations and artifacts','','- Teacher verification/config/source provenance: ../../runs/'+ID+'/teacher_verification.json and provenance.json. Original checkpoint unchanged.','- All six configurations, initial/best/last checkpoints, optimizer state, CPU shuffle state, full history and stopping records: ../../checkpoints/'+ID+'/.','- Validation comparison, complete Pareto selection rule, affine fit, plots and this report are in this directory. Per-configuration metrics, latent arrays, predictions, coordinates, complete signed contributions, interaction summaries, candidate features and truncation residuals are under the six configuration folders.','- Tests, teacher verification log, resource check, managed tmux script, both workers’ stdout/stderr and exit statuses: ../../execution_logs/'+ID+'/.','- No final population, historical test scoring, original-neuron interventions, extra training configurations or teacher weight edits. Reused validation/probes are not blinded evidence. One student seed per width limits stability claims. Known endpoints and exact decompositions do not prove a unique original circuit.','','Executed preparation/tests/managed worker commands are recorded in execute.sh, command manifests and logs. Reporting command: `.venv/bin/python -m experiments.v2_pairwise_superposition.stage2.report`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    before=json.loads((RUN/'protected_before.json').read_text());assert all(digest(Path(p))==h for p,h in before.items())
    for width in C['widths']:
        status=int((BASE/'execution_logs'/ID/f'worker-{width}.exit_status').read_text());assert status==0
    shutil.copyfile(Path(__file__),RUN/'source_snapshot/stage2_report.final.py')
    write_json(RUN/'reported.json',{'six_complete':True,'protected_files_verified':len(before),'report_sha256':digest(OUT/'report.md'),'report_source_sha256':digest(__file__),'teacher_sha256':verification['checkpoint_sha256'],'reserved_final_accessed':False})
    print(OUT/'report.md')

if __name__=='__main__':main()
