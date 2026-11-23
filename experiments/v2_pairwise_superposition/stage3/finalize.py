"""Saved-result scientific assessment; no additional model evaluation or training."""
import json,csv,shutil
from pathlib import Path
import numpy as np
from src.runtime import write_json,load_checkpoint
from ..run import BASE,ROOT,digest,write_csv
from ..data import targets
from .run import C,RUN,OUT,ID,TEACHER,TC
from .methods import group_metrics


def main():
    fixed=json.loads((RUN/'fixed_hypotheses.json').read_text());groups=fixed['groups'];selection=fixed['selection'];tcrows=[];discovery_rows=[];collateral=[];compact=[]
    with np.load(OUT/'native_coordinates.npz') as f:native={k:f[k] for k in f.files}
    with np.load(OUT/'transcoder_coordinates.npz') as f:tc={k:f[k] for k in f.files}
    with np.load(OUT/'discovery_exact.npz') as f:discovery={k:f[k] for k in f.files}
    for split in ('discovery','heldout','stress'):
        with np.load(OUT/f'{split}_exact.npz') as f:bank={k:f[k] for k in f.files}
        if split=='discovery':
            for group in groups:
                if group['control']:continue
                m=group_metrics(bank['h'],bank['output'],native['D2_eff'],native['semantic_bias'],group['neurons'],np.array(group['discovery_intercept']))
                discovery_rows.append({'name':group['name'],'method':group['method'],'k':group['k'],**{k:v for k,v in m.items() if not k.startswith('per_context')}})
        for r in selection['tc_sets']:
            units=r['units'];offset=(discovery['output']-discovery['z'][:,:,units]@tc['D2_eff'][:,units].T).mean((0,1))
            m=group_metrics(bank['z'],bank['output'],tc['D2_eff'],tc['semantic_bias'],units,offset)
            m={k:v for k,v in m.items() if not k.startswith('per_context')};tcrows.append({'population':split,'k':r['k'],'units':str(units),**m})
        if split=='discovery':continue
        with np.load(OUT/f'{split}_candidate_effects.npz') as f:
            for group in groups:
                if group['control']:continue
                for kind in ('zero','clamp'):
                    actual=f[group['name']+'_'+kind+'_actual_effect']
                    for predictor in ('invariant_gate','context2_gate'):
                        predicted=f[group['name']+'_'+kind+'_'+predictor+'_predicted_effect']
                        for j in range(9):collateral.append({'population':split,'name':group['name'],'k':group['k'],'intervention':kind,'predictor':predictor,'output':j,'actual_mean_effect':float(actual[:,:,j].mean()),'predicted_mean_effect':float(predicted[:,:,j].mean()),'effect_rmse':float(((actual[:,:,j]-predicted[:,:,j])**2).mean()**.5),'actual_mean_abs_effect':float(np.abs(actual[:,:,j]).mean())})
                # Standalone compact g4 prediction, with discovery-only intercept, versus target-AND sanity check.
                proxy=-f[group['name']+'_zero_invariant_gate_predicted_effect']+np.array(group['discovery_intercept'])
                compact.append({'population':split,'name':group['name'],'k':group['k'],'invariant_g4_absolute_rmse':float(((proxy[:,:,4]-bank['output'][:,:,4])**2).mean()**.5),'boolean_AND_posthoc_sanity_rmse':float(((targets(bank['a'])[:,:,4]-bank['output'][:,:,4])**2).mean()**.5),'note':'Boolean rule is a post hoc sanity reference, not a preregistered intervention prediction.'})
    write_csv(OUT/'discovery_group_metrics.csv',discovery_rows);write_csv(OUT/'tc_small_group_fidelity.csv',tcrows);write_csv(OUT/'collateral_prediction.csv',collateral);write_csv(OUT/'compact_formula_fidelity.csv',compact)
    # Record complete compact formula parameters for the common native trio.
    with np.load(RUN/'fixed_gate_parameters.npz') as f:parameters={k:f[k] for k in f.files}
    trio=[224,192,30];formula={'neurons':trio,'pair_columns':[7,8],'terms':[{'neuron':r,'output_write':float(native['D2_eff'][4,r]),'a7_coefficient':float(native['C1'][r,7]),'a8_coefficient':float(native['C1'][r,8]),'discovery_mean_context_offset':float(parameters['invariant_offset'][r]),'original_bias':float(native['b1'][r]),'context2_indices':parameters['context_indices'][r].astype(int).tolist()} for r in trio]}
    formula['constant_g4_intercept']=next(g['discovery_intercept'][4] for g in groups if g['name']=='tc_guided_signed_k3');write_json(OUT/'three_neuron_formula.json',formula)
    held=list(csv.DictReader((OUT/'heldout_group_metrics.csv').open()));inter=list(csv.DictReader((OUT/'heldout_interventions.csv').open()))
    ref=next(r for r in inter if r['name']=='tc_guided_signed_k3' and r['intervention']=='zero')
    nativeheld=json.loads((OUT/'heldout_thresholds/units.json').read_text());gates={r['unit']:r for r in nativeheld if r['model']=='native'}
    assertion='''**Central result: no incremental discovery advantage over direct native analysis was demonstrated for AND(7,8).** Transcoder-guided signed selection finds neurons30/192/224 at k3. Direct signed native selection and even mean-absolute native attribution recover the identical trio without the transcoder. All three therefore have exactly the same held-out pair-response RMSE0.01166, interaction MAE0.01208 and absolute g4 MSE0.0001165. The direct signed baseline is more accurate at larger budgets (k5 interaction MAE0.00806 versus guided0.01117; k20 0.00434 versus guided0.01213). This equality/advantage must not be hidden by comparing guidance only against the weaker net-only greedy search.

The net-only greedy/one-swap search fails to recover the opposing witness combination, despite optimizing the teacher's net response directly: at k3 its interaction MAE is0.22994, and even k20 remains0.22659. This is a search limitation, not evidence that direct access to the original computation lacks information. Signed channel matching escapes that particular trap, and the transcoder suggests such a decomposition, but the same information is already available from native signed writes. The preregistered primary comparison favors guidance over this net-only heuristic; the strong native-signed and attribution references show that the broader claim of extra transcoder value is not supported. No selection rule or baseline was changed after seeing held-out outcomes.

**Features173 and266 are conditional opposing conjunction-like gates, not a complete circuit or certified global AND detectors.** On1,000 new held-out supported contexts,173 is11-only in97.2%, while266 is11-only in93.2% and inactive otherwise. Their mean signed pair contributions are+0.55922 and−0.15932. The pair alone is missing much of the target interaction; discovery pair-response RMSE versus full TC is0.2983 and interaction MAE0.5917. The discovery-augmented set first meets the declared thresholds at10 TC features, not2. Full held-out results for all fixed TC group sizes are in tc_small_group_fidelity.csv. Features' nonzero off-target reads and failure contexts are exported rather than suppressed.

**A compact native three-neuron explanation is supported on this development population.** The relevant formulas preserve the actual original weights: neuron224 has pair coefficients(0.69706,0.75114), bias−0.81508 and g4 write+2.35606; neuron192 has(0.61357,0.64771), bias−0.95545 and write−3.44189; neuron30 has(0.61931,0.61671), bias−0.68375 and write+1.01680. All retain context reads; full20-dimensional coefficients are in the coordinate archive. Their held-out11-only gate fractions are98.7%,96.4%,98.8%. Average interaction contributions are+1.16964,−0.57617,+0.39011, summing+0.98358. Opposing terms are essential: a largest-positive-unit account overshoots; the negative neuron is not noise. Individual context SDs are0.2381,0.2944,0.1105 while the group's SD is only0.04998, consistent with stabilizing signed cancellation. This describes an approximate context-modulated gate group, not a unique Boolean implementation across all2^20 inputs. Three is the smallest *tested* successful native budget; k2 was not tested, so global minimality is not established.

**Actual causal tests confirm the selected group matters, but exact identities are not mechanistic discovery.** Zeroing the trio reduces the mean g4 pair interaction by0.98358, removing nearly all of the teacher's relevant interaction on these contexts. Effects are not inferred solely from attributions: hidden activations are actually replaced and passed through the original readout. The exact signed-sum prediction is verified numerically, as expected for this one-hidden-layer network; that identity alone is not independent validation. The discovery-only invariant threshold formula predicts held-out g4 intervention effects with RMSE0.0250, a nontrivial cross-context test. It retains the original pair coefficients and output writes while replacing all other reads by discovery means; exact fixed offsets are in three_neuron_formula.json. This is a limited generalizing summary of the selected computation, not proof of one unique mechanism.

Several failures qualify that result. Adding two context reads per neuron worsens trio effect prediction to RMSE0.09697 and interaction-effect MAE0.14936, indicating that independently truncated context reads can destroy coordinated cancellation. A saved error-covariance audit makes this precise: each neuron’s signed g4 effect RMSE improves with two context reads (224:0.1199→0.0995,192:0.1472→0.1244,30:0.0558→0.0434), but their combined error worsens0.0250→0.0970 because the compensating error covariances weaken. An apparently richer neuronwise summary is not automatically a better group explanation. The discovery00 means for the core trio are allzero, so its chosen baseline-clamp intervention is identical to zero ablation; this does not supply a second independent causal contrast. On other groups the replacement levels/effects can differ, but constant replacement still removes the same pair interaction by construction. OOD stress results are separate and can be worse. A post hoc Boolean-AND sanity predictor is also reported: because the teacher already approximates AND closely, low ordinary output error alone is insufficient evidence of novel mechanism recovery.

**Collateral effects are real.** The trio's g8 mean absolute output effect is about0.24422 per corner, with its interaction reduced as well; g8 includes this AND clause, so this is consistent with computational sharing rather than automatically an error. The inhibition output g7 also changes substantially (mean absolute effect0.17990), although its Boolean definition excludes a7/a8. This is not an isolated g4 module: other original neurons must compensate cross-output writes in the intact model. The invariant formula’s held-out effect RMSE is0.0250 on g4,0.04263 on g7 and0.10854 on g8. Full output-specific intervention changes and discovery-only predictor errors are in collateral_prediction.csv and the intervention tables; the average absolute collateral effect over the other eight outputs is0.06359. A group that is compact for g4 is not a perfectly isolated g4 module. Candidate/control comparison must also respect matching quality and overlap: contribution-matched groups often contain core neurons, and some k1 targets have no adequate alternative. Random controls alone would overstate the specificity of a discovery advantage.

**Answer to the research question:** the transcoder supplied recognizable opposing gate components and guided a successful original-neuron group, but direct signed attribution found the same compact group at the same budget and improved beyond it. This experiment does not establish that the transcoder identifies the original mechanism more effectively than direct native analysis. It supports the native trio as a useful, approximate, development-validated explanation with contextual limits and collateral outputs. Further causal/final generalization claims require new scientific review; the reserved final population has not been accessed. No XOR or other case study, model training, additional selection variant, or weight edit was launched.'''
    p=OUT/'report.md';text=p.read_text();assert 'ASSESSMENT_PENDING' in text;shutil.copyfile(p,OUT/'report.initial.md');text=text.replace('ASSESSMENT_PENDING',assertion)
    text+='\n## Additional sufficiency and formula results\n\n| Population | Fixed TC budget | Pair-response RMSE versus teacher | Interaction MAE versus teacher |\n|---|---:|---:|---:|\n'
    for r in tcrows:text+=f"| {r['population']} | {r['k']} | {r['pair_response_g4_rmse']:.6g} | {r['interaction_g4_mae']:.6g} |\n"
    text+='\nThe fixed native trio compact formula is `g4_hat = intercept + Σ_r d_r ReLU(w7_r a7 + w8_r a8 + eta_r)` with discovery-derived mean background offsets; it is not refitted on held-out outcomes.\n\n```json\n'+json.dumps(formula,indent=2)+'\n```\n'
    text+='\n## Complete input/readout coefficients for the five inspected units\n\nAll20 input coefficients are listed in a0..a19 order and all9 output writes in g0..g8 order; the exact float64 coordinate archives preserve full precision.\n'
    for label,coord,ids in [('Original neuron',native,[224,192,30]),('Transcoder feature',tc,[173,266])]:
        for unit in ids:
            text+=f'\n**{label} {unit}**: original/encoder bias `{coord["b1"][unit]:.9g}`.\n\nInput coefficients: `'+str([float(f'{v:.7g}') for v in coord['C1'][unit]])+'`\n\nSigned semantic writes: `'+str([float(f'{v:.7g}') for v in coord['D2_eff'][:,unit]])+'`\n'
    text+='\n## Execution record\n\n'+(BASE/'execution_logs'/ID/'tests_verified.log').read_text().strip().splitlines()[-1]+'. Initial test log records a dtype mismatch in the tiny CUDA replacement fixture; that fixture was corrected before the experiment. No actual experiment retry or post-outcome selection modification occurred.\n\n'
    status=json.loads((RUN/'execution.json').read_text());maximum=max(json.loads((OUT/f'{s}_intervention_identity.json').read_text())['max_actual_vs_exact_signed_effect_error'] for s in ('heldout','stress'))
    text+=f"The tmux-managed experiment exited0, taking {status['seconds']:.2f}s after model verification on NVIDIA A100-SXM4-40GB with PyTorch{status['torch']}. All {status['protected_files_verified']} prior protected artifacts verified unchanged. Maximum actual-intervention/exact-effect discrepancy {maximum:.3g}. Discovery hypotheses SHA-256 `{status['hypotheses_sha256']}` was verified unchanged after held-out/stress evaluation.\n\n"
    text+='Executed commands: `CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest experiments/v2_pairwise_superposition/tests -q`; tmux session `v2-stage3-and78` ran `execution_logs/stage3-and78-20261027/execute.sh`, which sets CUDA_VISIBLE_DEVICES=0 and deterministic CuBLAS, invokes `.venv/bin/python -m experiments.v2_pairwise_superposition.stage3.run`, logs stdout/stderr and Python exit status. Report-only finalization command: `.venv/bin/python -m experiments.v2_pairwise_superposition.stage3.finalize`. No packages were installed. Protocol notes are in docs/v2_stage3.md; the exact user request and all fixed configuration/prediction records are archived in the run directory.\n'
    p.write_text(text)
    # Cross-check data/model identities with the earlier Stage2 verification and each checkpoint.
    stage2=json.loads((BASE/'runs/stage2-transcoders-20261027/teacher_verification.json').read_text());teacher=load_checkpoint(TEACHER);student=load_checkpoint(TC)
    assert json.loads((RUN/'model_verification.json').read_text())['dataset_sha256']==stage2['dataset_sha256'];assert json.loads((RUN/'model_verification.json').read_text())['manifest_sha256']==stage2['split_manifest_sha256'];assert digest(TEACHER)==stage2['checkpoint_sha256'];assert teacher['identity']['dataset_sha256']==stage2['dataset_sha256'];assert student['width']==1024 and student['lambda']==.0001;assert student['identity']['dataset_sha256']==stage2['dataset_sha256']
    protected=json.loads((RUN/'protected_before.json').read_text());assert all(digest(Path(path))==sha for path,sha in protected.items());assert digest(RUN/'fixed_hypotheses.json')==status['hypotheses_sha256']
    shutil.copyfile(Path(__file__),RUN/'sources/stage3_finalize.final.py');shutil.copyfile(ROOT/'docs/v2_stage3.md',RUN/'protocol_notes.md')
    write_json(RUN/'complete.json',{'status':'complete','report_sha256':digest(p),'finalize_source_sha256':digest(__file__),'models_trained':0,'original_weights_changed':False,'hypotheses_sha256':status['hypotheses_sha256'],'protected_files_verified':len(protected),'actual_intervention_max_identity_error':maximum,'reserved_final_file_accessed':False,'cases':['AND(7,8)'],'exit_status':int((BASE/'execution_logs'/ID/'exit_status').read_text())});print(p)

if __name__=='__main__':main()
