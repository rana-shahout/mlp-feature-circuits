"""Saved-results-only report for the single width128 controlled experiment."""
import json
import numpy as np
from .capacity import BASE,ROOT,OUT,RUN,CK,OLD,NEW,SOURCE,quantiles
from .data import RULES,PAIRS,keys
from .analysis import group_metrics
from .run import write_csv,digest
from src.runtime import write_json


def concentration(v):
    absolute=np.abs(v);total=absolute.sum(-1);norm=np.divide(absolute,total[...,None],out=np.zeros_like(absolute),where=total[...,None]>0)
    positive=np.maximum(v,0).sum(-1);negative=np.minimum(v,0).sum(-1);net=v.sum(-1)
    neff=np.divide(total**2,(v*v).sum(-1),out=np.zeros_like(total),where=(v*v).sum(-1)>0)
    top=np.argmax(absolute,axis=-1);topvalue=np.take_along_axis(v,top[...,None],axis=-1)[...,0]
    return {'positive':positive,'negative':negative,'net':net,'cancellation':np.divide(total-np.abs(net),total,out=np.zeros_like(total),where=total>0),'effective_units':neff,'top1_abs_share':norm.max(-1),'top1_signed_residual':np.abs(net-topvalue),'top1_neuron':top}


def interactions():
    rows=[];units=[];contexts=[]
    with np.load(OUT/'signed_interactions.npz') as f:
        for j,indices in enumerate(list(PAIRS)+[(1,2),(0,1,2)]):
            corners=f[f'original_p{j}_corners'];density=corners.sum(-1);inside=(density.min(1)>=3)&(density.max(1)<=6)
            for output,rule in enumerate(RULES):
                for label in ('original','capacity'):
                    signed=f[f'{label}_p{j}_contributions'][:,output,:];stats=concentration(signed);truth=f[f'p{j}_ground_truth'][:,output]
                    for regime,mask in [('all',np.ones(len(signed),bool)),('every_corner_density_3_to_6',inside),('some_corner_OOD_density',~inside)]:
                        if not mask.any():continue
                        row={'model':label,'indices':str(indices),'rule':rule,'regime':regime,'n_contexts':int(mask.sum()),'target_mae':float(np.abs(stats['net'][mask]-truth[mask]).mean())}
                        row.update({k:float(v[mask].mean()) for k,v in stats.items() if k!='top1_neuron'});rows.append(row)
                    for context in range(len(signed)):
                        contexts.append({'model':label,'indices':str(indices),'rule':rule,'context_index':context,'all_corners_within_density_support':bool(inside[context]),'target_interaction':float(truth[context]),**{k:float(v[context]) for k,v in stats.items()}})
                    # Every neuron: no absolute-top-k truncation in this export.
                    for r in range(signed.shape[1]):
                        units.append({'model':label,'indices':str(indices),'rule':rule,'neuron':r,'mean_signed_contribution':float(signed[:,r].mean()),'mean_absolute_contribution':float(np.abs(signed[:,r]).mean()),'positive_context_fraction':float((signed[:,r]>0).mean()),'negative_context_fraction':float((signed[:,r]<0).mean())})
    write_csv(OUT/'interaction_concentration.csv',rows);write_csv(OUT/'interaction_contexts.csv',contexts);write_csv(OUT/'all_neuron_contributions.csv',units)
    return rows


def distractors(d):
    with np.load(OUT/'controlled_predictions.npz') as f:bank={k:f[k] for k in f.files}
    mapping={int(k):i for i,k in enumerate(keys(bank['a']))}
    with np.load(SOURCE/'constructed_probes.npz') as f:old_a=f['a'].copy()
    metadata=json.loads((SOURCE/'probe_metadata.json').read_text());rows=[]
    contexts=sorted(set(m['context'] for m in metadata))
    for context in contexts:
        for bit in range(15,20):
            idx=[i for i,m in enumerate(metadata) if m['kind']=='distractor_flip' and m['context']==context and m['designed_rule']==str(bit)]
            if len(idx)!=2:raise ValueError('Nuisance pairing changed')
            off,on=map(lambda i:mapping.get(int(keys(old_a[i:i+1])[0])),idx)
            if off is None or on is None:continue
            np.testing.assert_array_equal(bank['g'][off],bank['g'][on])
            for j,rule in enumerate(RULES):
                for label in ('original','capacity'):
                    score=bank[label].astype(float)@d['Q'].astype(float);s0,s1=score[[off,on],j];g=int(bank['g'][off,j]);dens=bank['a'][[off,on]].sum(1)
                    rows.append({'model':label,'context':context,'distractor':bit,'rule':rule,'target':g,'off_score':float(s0),'on_score':float(s1),'signed_change':float(s1-s0),'absolute_change':float(abs(s1-s0)),'classification_flip':bool((s0>=.5)!=(s1>=.5)),'off_error':bool((s0>=.5)!=g),'on_error':bool((s1>=.5)!=g),'off_density':int(dens[0]),'on_density':int(dens[1]),'within_density_support':bool(((dens>=3)&(dens<=6)).all())})
    write_csv(OUT/'matched_distractor_flips.csv',rows)
    summary={}
    for label in ('original','capacity'):
        for regime in ('all','within_density_support'):
            r=[v for v in rows if v['model']==label and (regime=='all' or v['within_density_support'])]
            summary[label+'_'+regime]={'rule_context_memberships':len(r),'mean_abs_score_change':float(np.mean([v['absolute_change'] for v in r])),'class_flip_fraction':float(np.mean([v['classification_flip'] for v in r])),'max_abs_score_change':float(max(v['absolute_change'] for v in r))}
    write_json(OUT/'distractor_summary.json',summary)
    return summary


def extra_plots(d,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label,color in [('original','C0'),('capacity','C1')]:
        rr=[next(v for v in rows if v['model']==label and v['indices']==str(PAIRS[j]) and v['rule']==RULES[j] and v['regime']=='all') for j in range(8)]
        axes[0].plot([v['effective_units'] for v in rr],'o-',color=color,label=label);axes[1].plot([v['cancellation'] for v in rr],'o-',color=color,label=label)
    for ax,title in zip(axes,['Effective units: (sum |c|)^2 / sum c²','Cancellation: 1 − |sum c| / sum |c|']):ax.set_title(title);ax.set_xticks(range(8),[f'g{j}' for j in range(8)]);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'interaction_distribution.png',dpi=160);plt.close(fig)
    geometry=json.loads((OUT/'pretraining_geometry.json').read_text())['groups']['validation']
    fig,ax=plt.subplots(figsize=(9,4))
    for group,color in [('positive_false_negative','C3'),('positive_correct','C2')]:
        vals=[]
        for j in range(8):
            q=geometry[RULES[j]][group]['opposite_distance']['quantiles_0_10_25_50_75_90_100'];vals.append(q[3] if q else np.nan)
        ax.plot(vals,'o-',color=color,label=group)
    ax.set_xticks(range(8),[f'g{j}' for j in range(8)]);ax.set_ylabel('Median distance to opposite-label training example');ax.set_title('Frozen embedding: baseline FN vs correctly recalled positives');ax.legend();fig.tight_layout();fig.savefig(OUT/'conflicting_label_geometry.png',dpi=160);plt.close(fig)


def produce(d):
    mets=json.loads((OUT/'metrics.json').read_text());dist=json.loads((OUT/'score_distributions.json').read_text());rows=interactions();nuisance=distractors(d)
    geometry=json.loads((OUT/'pretraining_geometry.json').read_text());training=json.loads((RUN/'training_complete.json').read_text());checks=json.loads((OUT/'interaction_checks.json').read_text())
    for split in ('train','validation'):
        with np.load(SOURCE/f'predictions_{split}.npz') as f:old=f['output'].copy()
        write_json(OUT/f'original_groups_{split}.json',group_metrics(d['a'][d[split]],old,d['g'][d[split]],d['Q'],.5))
    extra_plots(d,rows)
    fmt=lambda v:'—' if v is None else f'{v:.5f}'
    lines=['# V2 controlled capacity experiment: 64 → 128 hidden neurons','','## Protocol and execution','','Exactly one new model was trained: 12→128→12 ReLU, original unweighted full-output MSE, frozen original P/Q and 20,000-example NPZ and split manifests, seed 3105, original CPU minibatch shuffle generator, Adam, cosine LR, batch size 256, maximum 2,000 epochs, patience 200/min_delta 1e-7, checkpoint selection by unweighted validation full-output MSE, threshold 0.5. The copied YAML differs from the original run only in dimensions.hidden. No resampling, threshold fitting, balanced loss, further widths/seeds, or transcoder training occurred.','','The same initialization seed does not imply identical or nested weights across architectures: tensor sizes, RNG consumption, and W2 fan-in initialization change with width. This is an unavoidable implication of the approved architecture change, not an additional intervention. Initialization was saved before training; training reseeds with the original seed.','','All geometry queries use train/validation only, with training examples as neighbors and self-matches excluded. Original test scores were not read or recomputed. Test vector identities were used only to exclude old-test-overlap constructed probes and entire interaction contexts. The independently reserved final population was neither opened nor scored. Prior checkpoints, logs and reports are hashed before and after execution.', '',f"CUDA hardware: NVIDIA A100-SXM4-40GB. PyTorch 2.9.1+cu128. Training completed {training['epochs']} epochs in {training['elapsed_this_invocation_seconds']:.2f} seconds; selected epoch {training['best_epoch']}, stop reason {training['stop_reason']}. Maximum full signed neuron-sum versus direct-network residual {checks['max_signed_sum_vs_direct_network_residual']:.3g}.",'','## Overall train and validation performance','','| Model / population | Exact vector | Full-output MSE | Semantic MSE |','|---|---:|---:|---:|']
    for label,m in mets.items():lines.append(f"| {label} | {m['exact_vector_accuracy']:.5f} | {m['output_mse']:.7f} | {m['semantic_mse']:.7f} |")
    for split in ('train','validation'):
        lines+=['',f'## Primitive recovery: {split}','','| Primitive | Model | Precision | Recall | FPR | Balanced accuracy | Semantic MSE |','|---|---|---:|---:|---:|---:|---:|']
        for j in range(8):
            for label in ('original','capacity'):
                r=mets[label+'_'+split]['per_rule'][j];lines.append('| '+RULES[j]+' | '+('width64' if label=='original' else 'width128')+' | '+' | '.join(fmt(r[k]) for k in ('precision','recall','false_positive_rate','balanced_accuracy','mse'))+' |')
    lines+=['','## DNF composite target (separate from primitives)','','| Population / model | Precision | Recall | FPR | Balanced accuracy | Semantic MSE |','|---|---:|---:|---:|---:|---:|']
    for label,m in mets.items():r=m['per_rule'][8];lines.append('| '+label+' | '+' | '.join(fmt(r[k]) for k in ('precision','recall','false_positive_rate','balanced_accuracy','mse'))+' |')
    lines+=['','## False-negative score distributions','','Fixed threshold remains 0.5; proximity bands are [0.45,0.5) and [0.4,0.5). All positive/negative/FN quantiles (including train) and ranking diagnostics are saved in score_distributions.json.','','| Primitive | Model | Validation FN count | FN q10 | FN median | FN q90 | FN within .05 |','|---|---|---:|---:|---:|---:|---:|']
    for j in range(8):
        for label in ('original','capacity'):
            r=dist[label+'_validation'][j];q=r['false_negative']['quantiles'];lines.append('| '+RULES[j]+' | '+label+' | '+str(r['false_negative']['n'])+' | '+' | '.join(fmt(v) for v in ([q[2],q[4],q[6]] if q else [None]*3)+[r['fn_fraction_within_0.05_below_threshold']])+' |')
    lines+=['','## Pre-run conflicting-label geometry','','Exact float64 Euclidean nearest-neighbor search in embedded 12D space; no latent-distance proxy or approximate search. Same/opposite labels are defined separately for each rule and for the full target vector. Float64 torch.cdist was validated against direct pairwise subtraction on toy data. All training queries exclude themselves. Conflicting neighbors can be training rows; validation rows are never training references. All nearest distances/indices are archived.','','| Primitive | Baseline validation FNs: N | Median distance to opposite-label train row | Correct positive N | Correct positives: median distance | FN opposite/same distance ratio median |','|---|---:|---:|---:|---:|---:|']
    for j in range(8):
        gg=geometry['groups']['validation'][RULES[j]];fn=gg['positive_false_negative'];correct=gg['positive_correct']
        def med(obj):q=obj['quantiles_0_10_25_50_75_90_100'];return q[3] if q else None
        lines.append('| '+RULES[j]+' | '+str(fn['opposite_distance']['n'])+' | '+fmt(med(fn['opposite_distance']))+' | '+str(correct['opposite_distance']['n'])+' | '+fmt(med(correct['opposite_distance']))+' | '+fmt(med(fn['opposite_to_same_ratio']))+' |')
    lines+=['',f"P mutual coherence: {geometry['P_mutual_coherence']:.5f}. Distances and example identities are in geometry_train.npz, geometry_validation.npz and closest_conflicting_label_pairs.csv. A close conflicting-label neighbor requires a sharp learned decision boundary; nonzero distance alone is not irreducible ambiguity. Associations are confounded by feature identities, label prevalence and input density, and do not establish geometry as the causal bottleneck.",'','## Controlled nonlinear interactions and signed contributions','','Identical fixed contexts for both models. Full four-corner primitive pairs and shared-input three-way contrasts retain every signed contribution from all 64 or 128 original neurons, including negative contributions. No neuron identity alignment between widths is assumed. Direct network finite differences are checked numerically against the signed sums.','','Historical constructed probes are diagnostic, not independent final evidence. Density 3–6 means within density support, not necessarily IID. Entire interaction contexts touching old test are excluded; remaining contexts are few and may repeat effective backgrounds. All-context summaries mix density-supported and OOD corners; interaction_concentration.csv separates regimes.','','| Primitive / model | Contexts | Target interaction MAE | Positive sum | Negative sum | Effective units | Top-1 abs share | Cancellation | Top-1 signed residual MAE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for j in range(8):
        for label in ('original','capacity'):
            r=next(v for v in rows if v['model']==label and v['indices']==str(PAIRS[j]) and v['rule']==RULES[j] and v['regime']=='all')
            lines.append('| '+RULES[j]+' / '+label+' | '+str(r['n_contexts'])+' | '+' | '.join(fmt(r[k]) for k in ('target_mae','positive','negative','effective_units','top1_abs_share','cancellation','top1_signed_residual'))+' |')
    probes=json.loads((OUT/'controlled_metrics.json').read_text())
    lines+=['','Effective units = (sum |c|)² / sum c²; cancellation = 1−|sum c|/sum |c|; top-1 is the largest absolute contribution separately in each context, an optimistic descriptive bound rather than a fixed compact circuit. Top-1 signed residual exposes cancellation that absolute concentration can hide. Zero-contribution contexts use zero for these statistics. all_neuron_contributions.csv contains every neuron, while signed_interactions.npz preserves every context and output. These are exact algebraic decompositions, not causal intervention validation.','','| Historical controlled population / model | Memberships | Exact vector | Full-output MSE |','|---|---:|---:|---:|']
    for name,m in probes['metrics'].items():lines.append(f"| {name} | {m['n']} | {m['exact_vector_accuracy']:.5f} | {m['output_mse']:.7f} |")
    lines+=['','## Distractor dependence','','Ordinary train/validation groups by distractor identity mask/count, total density, antecedent corners and DNF clauses are saved for both widths. These groups are observational. Matched distractor flips hold every other input fixed; only complete pairs with no historical-test overlap are retained. Counts below are rule-context memberships (nine outputs per flip), not independent unique examples.','','| Model / context regime | Rule-context memberships | Mean absolute score change | Classification flip fraction | Max absolute change |','|---|---:|---:|---:|---:|']
    for label,r in nuisance.items():lines.append('| '+label+' | '+str(r['rule_context_memberships'])+' | '+' | '.join(fmt(r[k]) for k in ('mean_abs_score_change','class_flip_fraction','max_abs_score_change'))+' |')
    lines+=['','## Scientific decision','','ASSESSMENT_PENDING','','## Files, provenance and reproducibility','','[Recall/FPR](recall_fpr.png) · [Score CDFs](score_cdfs.png) · [Learning curves / PR](learning_and_pr.png) · [Geometry](conflicting_label_geometry.png) · [Interaction distribution](interaction_distribution.png)','','- metrics.json, per_rule.csv, predictions_train.npz, predictions_validation.npz: ordinary performance and raw continuous outputs.','- pretraining_geometry.json, geometry_*.npz, closest_conflicting_label_pairs.csv: diagnostics recorded before training.','- signed_interactions.npz, interaction_checks.json, interaction_contexts.csv, interaction_concentration.csv, all_neuron_contributions.csv: signed nonlinear analysis.','- controlled_predictions.npz, controlled_metrics.json, matched_distractor_flips.csv, distractor_summary.json, *_groups_*.json: controlled and observational context diagnostics.','- Checkpoints: ../../checkpoints/'+NEW+'/initial.pt, best.pt, last.pt.','- Provenance / history / copied config / source snapshot: ../../runs/'+NEW+'/.','- Exact tmux launch script, stdout/stderr and exit status: ../../execution_logs/'+NEW+'/.','- Reserved final population was not accessed. No transcoder was trained.','', 'The original SPEC v0.3 first-run width remains 64; the user’s explicit capacity decision authorizes this isolated width128 override. SPEC.md is not rewritten.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(RUN/'report_generation.json',{'report_sha256':digest(OUT/'report.md'),'source_sha256':digest(BASE/'capacity_report.py'),'model_inference':False})
