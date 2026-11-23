"""Postprocess saved V2 predictions only; never rescore models or train."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .data import RULES,PAIRS,targets,clause_counts,keys
from .analysis import metrics
from .run import locations,write_csv,digest
from src.runtime import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();paths=locations(args.run_id);out=paths['reports'];run=paths['runs']
    if not (run/'complete.json').exists():raise RuntimeError('Review requires a successfully completed Stage 1 run')
    if (out/'hard_case_analysis.md').exists():print('Saved review:',out/'hard_case_analysis.md');return
    m=json.loads((out/'metrics.json').read_text());groups=json.loads((out/'grouped_metrics.json').read_text())
    with np.load(paths['data']/'dataset.npz',allow_pickle=False) as data, np.load(out/'predictions_test.npz') as prediction:
        idx=data['test'];a=data['a'][idx];g=data['g'][idx];y=prediction['output'];scores=prediction['semantic'];q=data['Q']
        clauses=clause_counts(a)
        lines=['# V2 hard-case review from saved results','',
            'This review reads cached predictions and exact contributions. No training, test re-inference, threshold change, or architecture revision occurred. All point metrics still refer to the one planned test evaluation.','',
            '## Primitive errors: fit versus generalization','',
            '| Output | Train recall | Validation recall | Test recall | Test FPR |','|---|---:|---:|---:|---:|']
        for j,rule in enumerate(RULES[:8]):
            lines.append(f"| {rule} | {m['train']['per_rule'][j]['recall']:.4f} | {m['validation']['per_rule'][j]['recall']:.4f} | {m['test']['per_rule'][j]['recall']:.4f} | {m['test']['per_rule'][j]['false_positive_rate']:.4f} |")
        lines += ['', 'The conjunction failures are predominantly missed positives, not excess false positives. Nonzero training errors mean the failures are not purely a held-out generalization issue; this does not distinguish optimization, loss/label balance, and representational limitations. One run cannot identify the cause.','',
            '## Observed AND/XOR/inhibition corners','',
            '| Output | 00 error / N | 01 error / N | 10 error / N | 11 error / N |','|---|---|---|---|---|']
        for j,rule in enumerate(RULES[:8]):
            cells=[]
            for pattern in ('00','01','10','11'):
                record=groups['test']['antecedent_'+rule][pattern]
                row=record['per_rule'][j]
                cells.append(f"{row['fp']+row['fn']} / {record['n']}")
            lines.append(f"| {rule} | "+' | '.join(cells)+' |')
        lines += ['', 'These observational cells have varying other inputs; they are not interchangeable with fixed-background causal/input contrasts. Controlled four-corner tables are preserved separately.','',
            '## Does g1 appear to leak into the DNF?','',
            '| Condition among true g8 negatives | N | g8 FPR | Mean g8 score |','|---|---:|---:|---:|']
        conditional=[]
        for g1 in (0,1):
            mask=(g[:,1]==g1)&(clauses==0)
            value=metrics(y[mask],g[mask],q,.5)
            row=value['per_rule'][8]
            lines.append(f"| g1={g1}, no DNF clauses | {int(mask.sum())} | {row['false_positive_rate']:.4f} | {float(scores[mask,8].mean()):.4f} |")
            conditional.append({'g1':g1,'n':int(mask.sum()),'dnf_false_positive_rate':row['false_positive_rate'],'mean_dnf_score':float(scores[mask,8].mean())})
        lines += ['', 'A difference is an association conditional on these inputs, not proof that the hidden network reuses a g1 circuit. The full a0/a1/a2 controls include g1-only positives with g8=false, and other-clause-on backgrounds.','',
            '## Controlled nuisance failures, including density support','']
        metadata=json.loads((out/'probe_metadata.json').read_text())
        with np.load(out/'constructed_probes.npz') as probes:
            pp=probes['semantic'];aa=probes['a'];gg=probes['g']
            cases=[]
            names=sorted(set(row['context'] for row in metadata))
            for context in names:
                for bit in range(15,20):
                    ids=[i for i,row in enumerate(metadata) if row['kind']=='distractor_flip' and row['context']==context and row['designed_rule']==str(bit)]
                    off,on=ids
                    for j,rule in enumerate(RULES):
                        cases.append({'context':context,'distractor':bit,'rule':rule,'target':int(gg[off,j]),'score_off':float(pp[off,j]),'score_on':float(pp[on,j]),
                            'delta':float(pp[on,j]-pp[off,j]),'density_off':int(aa[off].sum()),'density_on':int(aa[on].sum()),
                            'off_overlap':metadata[off]['population_overlap'],'on_overlap':metadata[on]['population_overlap'],
                            'both_within_training_density':bool(3<=aa[off].sum()<=6 and 3<=aa[on].sum()<=6)})
            write_csv(out/'distractor_effects_with_provenance.csv',cases)
            for row in sorted((r for r in cases if r['both_within_training_density']),key=lambda r:abs(r['delta']),reverse=True)[:5]:
                lines.append(f"- {row['rule']}, nuisance {row['distractor']}, {row['context']}: score {row['score_off']:.4f}→{row['score_on']:.4f}, target {row['target']}; densities {row['density_off']}→{row['density_on']}; overlaps {row['off_overlap']} / {row['on_overlap']}.")
        lines += ['', 'These demonstrate nuisance dependence of the actual teacher, including within the training density support. Constructed rows can be seen or unseen and are labeled above; they are not an independent generalization population.','',
            '## Native interaction magnitudes and descriptive concentration','',
            '| Primitive | Target isolated effect | Mean absolute interaction error over contexts | Mean effective original neurons | Top-1 held-context approximation MAE |','|---|---:|---:|---:|---:|']
        with np.load(out/'native_coordinates_interactions.npz') as native:
            contributions=native['contributions'];value=native['native_values'];truth=native['ground_truth']
            fixed=json.loads((out/'dominant_neuron_approximation.json').read_text())
            interaction_rows=[]
            for j,pair in enumerate(PAIRS):
                c=contributions[j,:,j,:];mass=np.abs(c).sum(-1);square=np.square(c).sum(-1)
                effective=np.divide(mass**2,square,out=np.zeros_like(mass),where=square>0)
                residual=float(np.abs(value[j,:,j]-truth[j,:,j]).mean())
                top_error=fixed[str(pair)]['1']['held_context_mae_to_full_teacher_interaction'][j]
                expected=1 if j<6 else (-2 if j==6 else -1)
                lines.append(f"| {RULES[j]} | {expected} | {residual:.5g} | {effective.mean():.3f} | {top_error:.5g} |")
                interaction_rows.append({'rule':RULES[j],'mean_interaction_error':residual,'mean_effective_neurons':float(effective.mean()),'top1_held_context_mae':top_error})
            with np.load(out/'hidden_diagnostics.npz') as hidden:
                active=hidden['activation_prevalence'];usage=hidden['interaction_usage_overlap']
                lines += ['',f"Mean active hidden units on training examples: {active.sum():.2f}/64. Dead on the training population: {int((active==0).sum())}. Interaction-usage cosine overlap g0/g1: {usage[0,1]:.4f}.",
                    'Correlations, shared usage, and top-neuron truncation are descriptive. They do not demonstrate a feature-channel code, original-neuron necessity, or a compact causally validated mechanism. Exact signed sums are implementation identities, not discoveries.']
        lines += ['', '## Assessment and proposed next decision','',
            'The 64-neuron teacher does not meet the frozen 99% per-primitive balanced-accuracy target. Inhibition and one AND are strong, while several other ANDs miss many positives and XOR remains below target. The DNF is harder and is reported separately; its multi-clause/dense stress behavior is not an IID failure rate.',
            'Do not proceed automatically to V2 transcoder training. First review whether to diagnose optimization/positive-support weighting under the unchanged architecture, compare a separately approved seed, or approve a population/capacity revision. These are alternatives for researcher decision, not actions taken here. Additional epochs alone are not proven sufficient by this run.',
            'No architecture, task rule, embedding geometry, threshold, dataset partition, or loss was changed after seeing the results. V1 and all existing V2 checkpoints remain preserved.']
        (out/'hard_case_analysis.md').write_text('\n'.join(lines)+'\n')
        write_json(out/'hard_case_analysis.json',{'conditional_g1_dnf':conditional,'native_interactions':interaction_rows})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4))
        for name in ('train','validation','test'):ax.plot(range(9),[r['false_negative_rate'] for r in m[name]['per_rule']],'o-',label=name)
        ax.set_xticks(range(9),[f'g{j}' for j in range(9)]);ax.set_ylabel('False-negative rate');ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(out/'false_negative_rates.png',dpi=160);plt.close(fig)
        original=out/'report.md'
        shutil.copyfile(original,out/'report_initial.md')
        original.write_text(original.read_text()+'\n\n## Additional saved-result review\n\n[Detailed hard-case analysis](hard_case_analysis.md)\n\n![Per-output false-negative rates](false_negative_rates.png)\n')
        write_json(run/'postprocessing_provenance.json',{'command':' '.join(__import__('sys').argv),'source_sha256':digest(Path(__file__)),'uses_cached_test_predictions_only':True,'test_inference_count_unchanged':1})
        print('Hard-case review:',out/'hard_case_analysis.md',flush=True)


if __name__=='__main__':main()
