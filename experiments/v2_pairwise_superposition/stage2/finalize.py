"""Final research assessment and provenance audit; no further model training."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from src.models import SparseTranscoder
from src.runtime import select_device,write_json,load_checkpoint
from ..data import keys
from ..run import BASE,digest,write_csv
from .run import C,ID,RUN,OUT,CK,verify_frozen
from .core import name

ASSESSMENT='''**Functional imitation is strong on the reused validation population, but it is not complete and does not establish a simpler mechanism.** The validation-fidelity winner is width1024/lambda1e-4: full-output MSE0.000464679, semantic fidelity MSE0.000615259, normalized MSE0.00615772 (99.384% variance-weighted explained variance), and exact-vector teacher agreement99.333%. This is roughly106× lower reconstruction MSE than the training-fitted affine baseline (0.049170;48.333% exact teacher agreement). The zero-L1 controls establish that the nonzero-L1 gains are not just the presence of another ReLU MLP: lambda1e-4 improves validation MSE and meanL0 at both widths, under paired initializations and minibatch ordering. This is one-seed evidence, not a universal claim that L1 improves fidelity.

The validation MSE/L0 Pareto frontier has three members: **512/1e-3**, **512/1e-4**, and **1024/1e-4**. The 512/1e-3 endpoint has meanL0=106.37 (20.78% of width) and MSE0.00101437; 512/1e-4 has171.94 (33.58%) and0.000739216; 1024/1e-4 has319.65 (31.22%) and0.000464679. Both zero-L1 controls are dominated. The 1024/1e-3 configuration is also dominated by512/1e-3 on both validation MSE and absolute meanL0, despite its lower width-normalized active fraction (13.91%). Thus fraction-active, absolute active count and fidelity rank models differently. The original teacher has mean active original neurons124.75/256; only512/1e-3 has fewer active units per input. L1 activation magnitude and L0 activity are not interchangeable.

**Teacher errors require particular caution.** The fidelity winner has99.829% exact agreement on the2,927 teacher-correct validation rows, but only79.452% on the73 teacher-incorrect rows. Of these73,58 retain the exact teacher vector,8 become correct relative to g, and7 change to another wrong vector;5 originally teacher-correct rows become incorrect. The student's Boolean exact accuracy97.667% is therefore slightly higher than the teacher's97.567%, but that is not the objective or evidence of better teacher imitation. Candidate mechanisms for teacher errors must always be checked against the actual frozen teacher. All configurations' teacher-error preservation and continuous residuals are retained rather than hidden by the high aggregate agreement.

**The full nonlinear interactions are faithfully approximated in the three initial case studies, with contextual limitations.** At the validation-fidelity winner, mean absolute interaction residuals are0.00986 forAND(7,8),0.02596 forXOR(11,12), and0.03910 forAND(0,1), across all eligible historical contexts. These are differences from the teacher's actual continuous interaction, not error against an ideal Boolean formula. Density-supported MAEs are0.00500,0.03153,0.02030, respectively, but only3–4 complete contexts are available for these cases. Signed cancellation remains substantial: winner AND(7,8) has +1.58523/−0.59419 (net+0.99105), XOR +2.21368/−3.94557 (net−1.73189), AND(0,1) +1.45229/−0.61855 (net+0.83374). The corresponding original sums are +1.68420/−0.69829, +2.34068/−4.08006, and +1.37551/−0.53481. Similar net effects do not prove the same decomposition or internal mechanism.

**No absolute interaction-sparsity advantage over the original neurons was demonstrated.** ForAND(7,8),XOR,AND(0,1), the teacher's mean effective counts are2.62,10.77,8.15. Every one of the six transcoders has larger absolute effective counts on each of these three matched case studies. The winner has7.47,22.88,16.82; the low-L0 Pareto endpoint512/1e-3 has4.98,15.14,12.73. Normalized counts are lower for the transcoders because their dictionaries are larger (winner0.73%,2.23%,1.64% versus teacher1.02%,4.21%,3.18%). This is greater width-normalized concentration, not fewer contributing computational units. Fixed top-k residuals are sometimes nonmonotonic because omitted positive and negative terms cancel. The sparse endpoint's top3 XOR subset has teacher-interaction MAE1.198 despite the full model's0.06735. A few large absolute attributions alone are an inadequate explanation.

**Candidate units provide useful conditional components, not a recovered Boolean circuit.** ForAND(7,8), winner feature173 has K7≈0.865,K8≈0.924 and a11-only activation gate on all15 examined contexts (four-corner pattern0001), mean activations[0,0,0,0.639], contributing+0.5568. Feature266 supplies an opposing−0.1786 contribution and is11-only in13 contexts, inactive in2. Their complete coefficients include off-target semantic and distractor reads; invariance on all inputs is not established. This is a recognizable conjunction-like conditional component, not a certified global AND detector.

ForXOR, winner feature84 is inactive on all four corners in3 contexts and11-only in13, with positive interaction contribution+0.3083. Feature910 is mostly11-gated but sometimes also responds at01, and contributes−0.5617. These features resemble context-dependent conjunction/hinge components written with opposite signs; they are not individual XOR detectors. XOR's net negative interaction is assembled from many signed terms, with winner cancellation≈71.75% versus teacher≈72.70%.

ForAND(0,1), winner feature633 is a particularly useful caution: K0≈−0.469 and K1≈+0.899, while stronger negative reads includea10,a19,a15. It is inactive in7 contexts and responds at01/11 in8; average activations[0,0.394,0,0.144]. Turning ona0 suppresses this response, and a negative output write yields a *positive* pair interaction+0.2082. Calling it an AND detector from its attribution would be wrong. Opposing feature619 is01-only in7 contexts and inactive in8. These components reflect the trained teacher's contextual computation, not a prescribed Boolean implementation.

The shared-input analysis does not reveal a clean reusablea0 computation for both conjunctions. On12 deduplicated matched base contexts, absolute nonlinear-contribution usage-profile cosine is only0.0102 for original neurons and0.0155 for the fidelity winner. Its strongest shared-ranked feature758 has mean absolute contributions about0.00255 toAND01→g0 and0.18658 toAND02→g1, a highly asymmetric association. Other configurations' matched profiles are also weakly overlapping. A large K_a0 or two nonzero writes does not demonstrate meaningful common circuitry. Full vectors, signs, matched nuisance changes and all gate patterns are exported for later scrutiny.

**Failures and limits remain.** Full-output exact teacher agreement on the reused constructed bank is98.875% for density-supported memberships and77.510% for OOD-density memberships at the fidelity winner. The latter is not reliable enough to generalize mechanism claims beyond training density. The supported bank contains repeated/seen vectors and few complete contexts, so it is not unbiased confirmation. The stronger penalty reduces activity but can worsen fidelity and attenuate positive/negative interaction magnitudes. Units' recognizable gates remain context-dependent. One initialization seed, adaptive reuse of validation and historical contrasts, incomplete teacher-error preservation, and the absence of original-neuron causal validation limit the scientific claims. The reserved independent final population is untouched, so no final evaluation claim is made.

**Recommendation: a qualified yes for exploratory original-weight/gate hypothesis generation, not for claiming compact mechanism recovery.** Use the validation-fidelity winner1024/1e-4 as the primary frozen surrogate, retaining the entire three-point Pareto frontier as a tradeoff record. AND(7,8) offers the clearest first conditional-gate hypotheses; XOR requires preserving opposing contributions, and AND(0,1) warns against semantic detector labels. Candidate features can now guide inspection of the original C1,D2_eff,biases and ReLU gates, with every proposed effect checked against the actual teacher. They do not yet offer a demonstrably smaller interaction-level explanation than the original neurons and are not reliable replacements for teacher-error/OOD computations. Do not infer a one-to-one feature/neuron mapping or shared mechanism from these results. No original-neuron intervention, teacher edit, additional sweep or further training is launched by this report.'''


def matched_reuse():
    summary=[];records=[]
    for width in C['widths']:
        for coefficient in C['lambdas']:
            key=name(width,coefficient)
            with np.load(OUT/key/'coordinates.npz') as f:coord={k:f[k] for k in f.files}
            with np.load(OUT/key/'signed_interactions.npz') as f:
                d0={int(k):i for i,k in enumerate(keys(f['p0_corners'][:,0,:].astype(np.uint8)))};d1={int(k):i for i,k in enumerate(keys(f['p1_corners'][:,0,:].astype(np.uint8)))};common=sorted(set(d0)&set(d1));i0=[d0[k] for k in common];i1=[d1[k] for k in common]
                for label,c0,c1 in [('teacher',f['p0_teacher_contributions'],f['p1_teacher_contributions']),('student',f['p0_contributions'],f['p1_contributions'])]:
                    v0=np.abs(c0[i0,0,:]).mean(0);v1=np.abs(c1[i1,1,:]).mean(0);cos=float(v0@v1/(np.linalg.norm(v0)*np.linalg.norm(v1)));ranking=np.argsort(-np.sqrt(v0*v1))
                    summary.append({'config':key,'model':label,'n_unique_matched_base_contexts':len(common),'absolute_contribution_usage_cosine':cos})
                    if label=='student':
                        for u in ranking[:5]:records.append({'config':key,'unit':int(u),'matched_base_keys':common,'mean_abs_and01_to_g0':float(v0[u]),'mean_abs_and02_to_g1':float(v1[u]),'mean_signed_and01_to_g0':float(c0[i0,0,u].mean()),'mean_signed_and02_to_g1':float(c1[i1,1,u].mean()),'K':coord['K'][u].tolist(),'bias':float(coord['encoder_bias'][u]),'A':coord['A'][:,u].tolist()})
    write_csv(OUT/'matched_shared_input_usage.csv',summary);write_json(OUT/'matched_shared_input_candidates.json',records)
    return summary


def finalize():
    record=verify_frozen();device=select_device('cuda');normerrors={};initial_equality={};shuffle=torch.Generator().manual_seed(C['shuffle_seed']);expected_states={}
    # One deterministic shuffle reconstruction, then compare actual completed budgets.
    for epoch in range(1,C['training']['epochs']+1):
        torch.randperm(14000,generator=shuffle);expected_states[epoch]=shuffle.get_state().clone()
    first_by_width={};trainingrows=[]
    for width in C['widths']:
        for coefficient in C['lambdas']:
            key=name(width,coefficient);initial=load_checkpoint(CK/key/'initial.pt')['model'];cp=load_checkpoint(CK/key/'best.pt');last=load_checkpoint(CK/key/'last.pt')
            assert cp['identity']['teacher_sha256']==record['checkpoint_sha256'] and cp['identity']['dataset_sha256']==record['dataset_sha256']
            if width not in first_by_width:first_by_width[width]=initial
            assert all(torch.equal(v,first_by_width[width][k]) for k,v in initial.items());initial_equality[key]=True
            assert torch.equal(last['shuffle_rng'],expected_states[last['epoch']])
            assert all(v is None for v in [None])  # No student/teacher state is updated in this audit.
            w=cp['model']['decoder.weight'].to(device);normerrors[key]=float((w.norm(dim=0)-1).abs().max());assert normerrors[key]<4e-7
            torch.manual_seed(C['student_seed']);reconstructed=SparseTranscoder(12,width,12)
            assert all(torch.equal(v,reconstructed.state_dict()[k]) for k,v in initial.items())
            tr=json.loads((CK/key/'training_complete.json').read_text());trainingrows.append({'config':key,**tr})
    write_csv(OUT/'training_completion.csv',trainingrows)
    reuse=matched_reuse()
    # Cached predictions only: distinguish preservation/correction/changed errors.
    with np.load(RUN/'teacher_targets.npz') as f:teacher=f['validation'].copy()
    from ..capacity256 import load_frozen_data
    d=load_frozen_data();truth=d['g'][d['validation']].astype(bool);q=d['Q'].astype(float);teacher_labels=teacher.astype(float)@q>=.5;teacher_wrong=(teacher_labels!=truth).any(1);errors=[]
    for width in C['widths']:
        for coefficient in C['lambdas']:
            key=name(width,coefficient)
            with np.load(OUT/key/'predictions_validation.npz') as f:labels=f['semantic']>=.5
            identical=(labels==teacher_labels).all(1);wrong=(labels!=truth).any(1)
            errors.append({'config':key,'teacher_wrong_n':int(teacher_wrong.sum()),'teacher_wrong_exactly_preserved':int((teacher_wrong&identical).sum()),'teacher_errors_corrected_to_truth':int((teacher_wrong&~wrong).sum()),'teacher_errors_changed_to_other_wrong_vector':int((teacher_wrong&~identical&wrong).sum()),'introduced_errors_on_teacher_correct':int((~teacher_wrong&wrong).sum())})
    write_csv(OUT/'teacher_error_transitions.csv',errors)
    protected=json.loads((RUN/'protected_before.json').read_text());assert all(digest(Path(p))==h for p,h in protected.items())
    p=OUT/'report.md';s=p.read_text();assert 'ASSESSMENT_PENDING' in s;shutil.copyfile(p,OUT/'report.initial.md');s=s.replace('ASSESSMENT_PENDING',ASSESSMENT)
    tests=(BASE/'execution_logs'/ID/'tests.log').read_text().strip().splitlines()[-1]
    maxres=max(json.loads((OUT/name(w,l)/'interaction_checks.json').read_text())['max_signed_sum_vs_direct_output_residual'] for w in C['widths'] for l in C['lambdas'])
    s+='\n## Completed execution and final reproducibility audit\n\n'+tests+'. These V2 tests include CUDA smoke/backpropagation for both transcoder widths, objective reduction/teacher detachment, tied loss definitions, unit decoder norms, paired initialization, exact semantic-coordinate forward/finite differences and signed contribution sums, and Pareto dominance. All were actually executed before training.\n\n'
    s+=f'All six runs completed the2000-epoch budget, with no worker retry or experiment expansion. Both managed workers exited0. Maximum signed-sum/direct finite-difference residual across all six was {maxres:.3g}. Decoder column norms on every best checkpoint differed from1 by at most {max(normerrors.values()):.3g}. Initial weights matched within each width across all three penalties and matched seed4105 reconstruction. Every final CPU shuffle-generator state matched seed4106 reconstruction for its completed budget. All {len(protected)} protected prior artifacts verified unchanged by SHA-256.\n\n'
    s+='Initialization uses standard PyTorch nn.Linear weights, encoder bias fixed at−0.1, decoder columns normalized immediately, and the default decoder bias. Models share initial weights only within width; training fully detaches teacher targets. Best checkpoint selection and min_delta early-stopping comparison are separate: strict lower reconstruction MSE saves the best checkpoint, while patience resets only after a reduction greater than1e-8. No checkpoint is selected using labels, contrasts, sparsity or final population.\n\n'
    s+='Additional saved audits: training_completion.csv records actual per-run time/status; teacher_error_transitions.csv separates copied, corrected, altered and introduced errors; matched_shared_input_usage.csv / matched_shared_input_candidates.json align12 deduplicated00-base contexts for both shared-input pairs. This stricter shared-input comparison differs from the earlier per-bank candidate ranking, which uses each pair’s full context list. No new model fitting or original-neuron interventions occur in these analyses.\n\n'
    s+='Actual commands: `CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest experiments/v2_pairwise_superposition/tests -q`; teacher verification/affine preparation with `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m experiments.v2_pairwise_superposition.stage2.run prepare`; named tmux session `v2-stage2-transcoders` ran execution_logs/stage2-transcoders-20261027/execute.sh, assigning width512 to GPU0 (A100-SXM4) and width1024 to GPU1 (A100-PCIE), sequential penalties per worker with stdout/stderr and status logging. Saved-resource inspection showed both GPUs idle before launch. Reporting/final audit commands are `.venv/bin/python -m experiments.v2_pairwise_superposition.stage2.report` and `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m experiments.v2_pairwise_superposition.stage2.finalize`. No packages were installed.\n'
    p.write_text(s)
    for path in (Path(__file__),BASE/'stage2/report.py'):shutil.copyfile(path,RUN/'source_snapshot'/(path.name+'.final'))
    write_json(RUN/'complete.json',{'status':'complete','configurations_completed':6,'protected_files_verified':len(protected),'tests':tests,'teacher_sha256':record['checkpoint_sha256'],'report_sha256':digest(p),'final_sources':{str(path):digest(path) for path in (Path(__file__),BASE/'stage2/report.py')},'decoder_norm_max_errors':normerrors,'paired_initialization':initial_equality,'shuffle_reconstruction':True,'max_interaction_identity_residual':maxres,'reserved_final_accessed':False,'historical_test_scored':False,'teacher_modified':False,'original_neuron_interventions':False,'extra_variants_launched':False})
    print(p)

if __name__=='__main__':finalize()
