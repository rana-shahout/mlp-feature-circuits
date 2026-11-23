"""Final saved-prediction interpretation/audit; no training or model inference."""
import json,shutil
from pathlib import Path
import numpy as np
from .capacity256 import BASE,ROOT,OUT,RUN,CK,NEW,OLD,PREVIOUS,WIDTH_RUNS,load_frozen_data
from .run import digest,write_csv
from src.runtime import write_json

ASSESSMENT='''**Continuous fidelity favors width256 over width128, but the unchanged all-primitive reliability gate is still not met.** Validation full-output MSE falls by 37.76% (0.0097205→0.0060496), semantic MSE falls 0.0129601→0.0080657, and exact-vector accuracy increases 94.50%→97.57% (+3.07 percentage points). Against width64, exact-vector accuracy improves by 6.87 points. This is the strongest original-loss teacher among the three on the permitted validation metric; it is not a guarantee of independently confirmed final generalization.

Validation balanced accuracy improves versus width128 for every primitive except AND(7,8), which remains perfect. AND(0,1) recall is unchanged at 97.61%, but its FPR drops 0.143%→0.036%. AND(0,2) recall improves 90.50%→98.00%; AND(3,4) 81.59%→95.02%; AND(5,6) 91.24%→94.33%; AND(9,10) 82.32%→93.43%. XOR balanced accuracy rises 98.73%→99.19%, while inhibition becomes perfect on validation. All AND FPRs are ≤0.036% and XOR FPR falls 1.271%→0.635%. The DNF improves separately to 98.87% balanced accuracy (97.94% recall, 0.193% FPR); it does not count toward the primitive gate.

**Three primitives pass:** AND(7,8), XOR and inhibition. The five remaining AND balanced accuracies are 98.786%, 98.982%, 97.512%, 97.165% and 96.717% (g0,g1,g2,g3,g5). AND(0,2) is very close but is still below99%; no rounding or threshold relaxation is used. Residual errors are predominantly missed positives rather than false positives. AND(3,4), AND(5,6), AND(9,10) retain validation FN rates of 4.98%, 5.67%, 6.57%, versus training FN rates 3.51%, 3.15%, 3.56%. The same ANDs fail the training BA gate too. AND(0,1) has comparable training/validation FN rates (2.21%/2.39%), and AND(0,2) has lower validation than training FNR (2.00%/2.88%). Thus incomplete training fit remains a substantial component, with an additional generalization gap for several rules, not a failure arising only on validation. Exact-vector errors are 1.66% on training versus 2.43% on validation. This is a descriptive comparison, not a causal partition of validation errors into fit and generalization components.

**The actual nonlinear computation remains nontrivial.** Width256 pairwise interaction MAE improves over width128 for all eight primitives in the pooled fixed contexts and in each small density-supported subset. Examples: pooled AND(9,10) interaction MAE 0.4594→0.2648, XOR 0.3405→0.2884; density-supported XOR 0.3902→0.0243 and AND(0,1) 0.1664→0.0699. Improvements over width64 are not uniformly monotonic: AND(7,8) pooled MAE at width256 is 0.01625 versus width64 0.01489, and density-supported AND(5,6) is 0.14193 versus width64 0.04139. Perfect classification does not imply exact continuous Boolean interactions.

Width256 contributions retain substantial positive/negative cancellation: XOR has average positive sum +2.34068 and negative sum −4.08006 (net −1.73938 versus Boolean target −2), with mean cancellation72.70%. AND(3,4) has +1.91316 and −1.03480 (net+0.87837), cancellation69.28%. AND(7,8), despite perfect ordinary classification, has +1.68420 and −0.69829, cancellation55.60%; even its context-specific largest neuron leaves signed interaction residual MAE0.24294. The largest-neuron residual for XOR is1.37771. These computations are not explained by simply reading off one strongest weight or discarding negative contributors. Exact full-neuron sums are implementation identities, not compact causal discoveries.

**Distribution must be interpreted in absolute and normalized terms.** For g0..g7, width256 effective counts are approximately 8.15,10.00,15.62,8.55,2.62,11.75,10.77,10.38, or 3.18%,3.91%,6.10%,3.34%,1.02%,4.59%,4.21%,4.05% of its256 units. Absolute counts exceed width64 for all eight rules and exceed width128 for7/8; XOR is the exception (11.44→10.77). Relative to width128, N_eff/H rises only for AND(3,4) and AND(9,10), falling for six primitives. Top1 absolute share falls for six rules but slightly rises for AND(5,6) and XOR. Thus more original neurons generally contribute in absolute terms, while most interactions occupy a smaller fraction of the wider network. Do not summarize that as uniformly more distributed or uniformly more concentrated. AND(7,8) remains the most concentrated case (top1 absolute share52.04%), but its signed cancellation and residual rule out an exact one-neuron account on this bank. Average active hidden units on training rows are124.75/256 (48.73%), with no training-dead units; interaction-effective counts refer to a different quantity than activation counts.

The reused controlled bank improves strongly within density support: all-nine-output exact accuracy rises95.625%→99.875% over800 memberships, whereas OOD-density accuracy remains47.19%. These constructed memberships repeat vectors and include previously seen cases; the contrast bank is not an IID validation population. Its near-perfect supported accuracy must not override the weaker ordinary validation AND recalls. Only3–5 complete density-supported finite-difference contexts remain per primitive after excluding historical-test overlaps, so nonlinear fidelity estimates have limited contextual coverage.

The width series supports a useful capacity/width-dependent optimization contribution: increasing width under unchanged embeddings, labels, loss and schedule improves both training fit and validation fidelity. One initialization seed and architecture-dependent initialization cannot isolate representational capacity from optimization dynamics, and neither the residual errors nor embedding geometry prove irreducible interference. No further width or training variant is proposed or run.

**Final recommendation:** retain width256 as the preferred candidate if the next scientific decision explicitly accepts studying an imperfect teacher’s actual computation. Under the currently stated ≥99% per-primitive validation criterion, it is not yet sufficiently reliable for automatic progression to the intended all-primitive transcoder experiment. A transcoder could imitate this nontrivial frozen computation including errors, but that would be an explicit acceptance decision, not evidence that the original Boolean task has been solved. Stop here; no transcoder or further capacity experiment was launched.'''


def main():
    d=load_frozen_data();transitions=[]
    for split in ('train','validation'):
        truth=d['g'][d[split]].astype(bool);decisions={}
        for width,run in WIDTH_RUNS.items():
            with np.load(BASE/'reports'/run/f'predictions_{split}.npz') as f:decisions[width]=f['semantic']>=.5
        for width in (64,128):
            for j,label in enumerate(list(__import__('experiments.v2_pairwise_superposition.data',fromlist=['RULES']).RULES)+['exact_vector']):
                old=(decisions[width][:,j]!=truth[:,j]) if j<9 else (decisions[width]!=truth).any(1)
                new=(decisions[256][:,j]!=truth[:,j]) if j<9 else (decisions[256]!=truth).any(1)
                transitions.append({'population':split,'baseline_width':width,'new_width':256,'rule':label,'n':len(truth),'both_correct':int((~old&~new).sum()),'corrected_by_256':int((old&~new).sum()),'introduced_by_256':int((~old&new).sum()),'both_wrong':int((old&new).sum())})
    write_csv(OUT/'paired_error_transitions.csv',transitions)
    protected=json.loads((RUN/'protected_before.json').read_text());assert all(digest(Path(p))==h for p,h in protected.items())
    p=OUT/'report.md';s=p.read_text();assert 'ASSESSMENT_DETAIL_PENDING' in s;s=s.replace('ASSESSMENT_DETAIL_PENDING',ASSESSMENT)
    tests=(BASE/'execution_logs'/NEW/'tests.log').read_text().strip().splitlines()[-1]
    residual=json.loads((OUT/'interaction_checks.json').read_text())['new_width_max_sum_vs_direct_forward_error']
    s+='\n## Final checks and preservation\n\n'+tests+'. Tests include a source-level check that the training loop differs from width128 only in model width, plus CUDA checkpoint roundtrips, exact pair/triple signed sums, and normalized-count behavior under zero-padding and duplicate units.\n\n'
    s+=f'One training/evaluation process exited0 without retry. All {len(protected)} protected files retain their original SHA-256 values, including old V1/V2 checkpoints and prior V2 logs/reports, original dataset and split manifest, SPEC and prompt. Initialization reconstruction, CPU minibatch shuffle reconstruction and original-baseline LR equality passed. Maximum new signed-sum/direct-forward error was {residual:.3g}. The reserved final population was not accessed.\n\n'
    s+='paired_error_transitions.csv records corrected, introduced, shared and absent ordinary errors for width256 versus each baseline, without rescoring either older model. All report tables refer only to train/validation or the reused labeled controlled bank. A report-only finalization command was executed after the managed run: `.venv/bin/python -m experiments.v2_pairwise_superposition.capacity256_assessment`. No model inference occurs in this finalization step.\n'
    p.write_text(s)
    for name in ('capacity256_assessment.py','capacity256_README.md'):shutil.copyfile(BASE/name,RUN/(name+'.snapshot'))
    write_json(RUN/'final_verification.json',{'report_sha256':digest(p),'assessment_source_sha256':digest(BASE/'capacity256_assessment.py'),'tests':tests,'protected_files_verified':len(protected),'experiment_exit':int((BASE/'execution_logs'/NEW/'exit_status').read_text()),'historical_test_scored':False,'reserved_final_accessed':False,'additional_variants_launched':False})
    print(p)

if __name__=='__main__':main()
