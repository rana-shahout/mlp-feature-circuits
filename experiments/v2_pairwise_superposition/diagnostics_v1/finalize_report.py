"""Report-only precision improvements after inspecting completed saved diagnostics."""
import csv,shutil,json
from pathlib import Path
import numpy as np
from ..workflow_v04.core import read,write,sha,now
from ..workflow_v04.report import table,vec
from .run import OUT,OLD,LOG,CASES

def main():
 assert (LOG/'exit_status').read_text().strip()=='0'
 for name in ['prepare','discovery','native','confirmation']:
  assert sha(OLD/f'{name}_lock.json')==read(OUT/'protocol.json')['original_locks'][name]
 path=OUT/'REPORT.md';shutil.copyfile(path,OUT/'REPORT.initial.md');text=path.read_text();summary=read(OUT/'diagnostic_summary.json');us=list(csv.DictReader(open(OUT/'unit_summary.csv')));ns=read(OLD/'native_explanations.json')
 text=text.replace('no bias/refitting bug','no bias/refitting bug')
 text=text.replace('Full native positive coefficient sums=', 'Full native unit-only positive coefficient sums=').replace('Full student positive coefficient sums=','Full student unit-only positive coefficient sums=')
 text=text.replace('The fixed partition assigns **every** unit', 'All per-unit sign sums exclude the output bias. Add native beta (output4: −0.086738269; output6: −0.125576911) or student gamma (output4: −0.034191114; output6: +0.288387096) to the baseline only for the complete output. These are the actual projected saved biases, never refitted.\n\nThe fixed partition assigns **every** unit')
 # Verify printed rounded biases against saved arrays, not the hand-typed report numbers.
 biasrows=[]
 for j in [4,6]:
  for label in ['native','student']:
   a=dict(np.load(OLD/f'confirmation_{label}_{j}.npz'));biasrows.append([j,label,f"{a['beta'][j]:.12f}"])
 # Use dynamically verified exact bias table to supersede finite rounding in prose.
 text=text.replace('All per-unit sign sums exclude the output bias. Add native beta (output4: −0.086738269; output6: −0.125576911) or student gamma (output4: −0.034191114; output6: +0.288387096) to the baseline only for the complete output. These are the actual projected saved biases, never refitted.', 'All per-unit sign sums exclude the output bias. Add the following actual projected saved bias to the baseline only for the complete output; it is never refitted.\n\n'+table(['output','model','true semantic output bias'],biasrows))
 marker='Native 224 is the positive conjunction-like driver'
 insert='The historical native triple has confirmation joint mean **0.999306** and SD **0.043440**, despite individual joint SDs **0.274860, 0.325892, 0.111427**. On discovery, its mean/SD are0.976597/0.056396. This is direct evidence of coordinated signed compensation, not just large contributors. It is not a full-response acceptance test of that triple; no new circuit is declared. The analogous student173/266 pair has confirmation mean joint0.407205 and SD0.087402, and therefore does not account for the full AND interaction. Similar gates do not make the groups equivalent.\n\n'
 assert abs(summary['4']['H']['native_reference_joint']['sum_mean']-.999306200484276)<1e-10
 text=text.replace(marker,insert+marker)
 marker='These are loss identities for a pre-named historical pair'
 text=text.replace(marker,'For the frozen guided k20 group, both singleton deltas are positive (+0.034499 for192 and +0.044363 for30), while the historical pair delta is negative (−0.009571). The first recorded [224] step has the same barrier. The fixed pair is a genuinely complementary move under the existing objective, not just a pair with similar activation patterns. This diagnoses a concrete limitation of one-unit greedy steps, though it does not audit every possible single swap or show that this pair alone fixes the full function.\n\n'+marker)
 # Explicitly separate positive sources of missing XOR singles from opposing large terms.
 rows=[];native_ids=ns['6']['sets']['guided']['20']
 for z,name in [(1,'first input'),(2,'second input'),(3,'joint')]:
  relevant=[r for r in us if r['model']=='native' and r['output']=='6' and r['split']=='H' and int(r['unit']) not in native_ids]
  sides=[]
  for sign in [1,-1]:
   rr=sorted([r for r in relevant if sign*float(r[f'mean_alpha{z}'])>0],key=lambda r:-sign*float(r[f'mean_alpha{z}']))[:4]
   sides.append(', '.join(f"{r['unit']}: {float(r[f'mean_alpha{z}']):+.5f}" for r in rr))
  rows.append([name,*sides])
 explain='The positive and negative sources of the missing XOR coefficients are:\n\n'+table(['omitted XOR coefficient','largest positive mean sources','largest negative mean sources'],rows)+'\n\nNative68 supplies a large first-input rise, while57 and200 supply both single-input terms;12 counters those singles while writing a positive joint, and63/137/57 provide major missing negative joint terms. These terms must be interpreted together. The exact per-context responses, q values and output writes for these named units are in the component tables; no new group is selected.\n\n'
 text=text.replace('XOR failure includes both missing single-input responses',explain+'XOR failure includes both missing single-input responses')
 # Clarify XOR mechanisms actually observed in both fixed partitions.
 marker='## Facts, computational hypotheses, and unresolved claims'
 specific='### A specific shared XOR motif, with its limits\n\nThe fixed discovery-modal **negative-write 00-only families** contribute confirmation mean coefficients native[−1.74132,+1.64047,+1.59387,−1.49453] and student[−1.02772,+0.98665,+0.97547,−0.93441]. Suppressing a negative00 response as either input turns on creates positive single-input terms plus a negative joint. Positive-write00-only families oppose those terms. Separately, negative-write11-only families supply roughly−0.99504 native/−1.00013 student joint, with other hinges and always-on affine units carrying compensation. These are functions of groups, not one-to-one neuron matches. H retention of the discovery-modal pattern is only0.587/0.545 for the negative00 families and0.522/0.571 for the negative11 families, so neither is an invariant Boolean detector family.\n\nThis suggests a sharper hypothesis than “XOR has negative interaction”: **offset-dependent00 suppression plus11 subtraction, supported by additive and sign-opposing compensation**. It predicts different single/baseline changes for the00 and11 families when a background moves a unit through its q boundary. Native-only gate/sign inspection can formulate the same hypothesis.\n\n'
 text=text.replace(marker,specific+marker)
 # Replace the sole follow-up with a more discriminating non-tautological target.
 start=text.index('## One targeted next experiment');end=text.index('## Execution and artifacts',start)
 proposed='''## One targeted next experiment — prospective XOR motif-transition prediction

Use output6 only to compare student-guided and native-only **hypothesis generation**, with frozen models and no circuit-subset optimization. Before new outcome tensors are inspected, fix four role families shared by both arms: negative-write00-only, positive-write00-only, negative-write11-only, and always-on affine. Their exact contextual definitions are gate inequalities and readout sign, not fitted feature/neuron correspondences. A unit may change contextual family at a boundary; this is functional accounting, not a claim of fixed circuit recovery.

Give both arms the same discovery backgrounds, known pair, hypothesis slots, compute limit and total component-row inspection budget (for example24 rows). Charge every student or native row inspection against that total; student access is the declared treatment, not free extra inspection. Native-only spends its budget on native components. No arm may enumerate all256 native rows, fit a readout, optimize a subset or consult held-out native activations. Use isolated routines or independent evaluators to prevent the already known motifs leaking asymmetrically. The experiment measures the value of guidance under this declared resource limit; it cannot establish an information-theoretic advantage.

Each arm must lock a quantitative prediction for the **aggregate native response change of each whole role family**, including uninspected native members, when a matched permitted development background crosses a specified q boundary. Predict all four coefficients and the expected total native four-corner change, with an explicit remaining-family remainder. Do not score only an inspected neuron's exact `write*ReLU` identity: that would be an algebraic tautology once its complete row is known, and would not test structural guidance. The challenge is predicting which family-wide compensation matters and its magnitude despite limited inspection.

An isolated administrative routine selects new matched background pairs from metadata and boundary conditions after the hypotheses and selection rule are locked, without viewing outcome values. Exclude every previously inspected input key, historical tests and the reserved final population; enforce complete corners, density support and a common development split. If insufficient matched contexts exist, report that limitation without relaxing protection. Neither the bank nor its outputs is constructed in this diagnostic.

Primary comparison: held-out family-aggregate four-corner-change MAE, averaged over **all four registered families**, with separate baseline/first/second/joint errors and gate-transition failures. Report total native response-change error, unmodeled remainder, and every failed prediction as secondary checks. Advantage on total output alone is insufficient if the proposed family changes are wrong. Better family-change predictions at the same inspection budget would support useful transcoder structural guidance; a tie or native-only win would leave that added value unsupported. This is one proposed experiment, not an executed study or a new selector campaign.

'''
 text=text[:start]+proposed+text[end:]
 text += '\nFinal status: diagnostic process exited0; all11 test groups passed. The report-only finalization adds explicit sign-separated XOR sources, coordinated-reference variability and a non-tautological follow-up target; it does not recompute inference or alter scientific tables. Initial report and its source snapshot are retained.\n'
 path.write_text(text)
 # Readable one-page entry point with five direct answers.
 brief='''# Diagnostic v1 decisions

Completed on the original frozen models and original v0.4 D/H contexts for outputs4/6 only. Eleven tests passed; CUDA identities passed; diagnostic exit status0. No selector, fitting, training, hidden intervention or new population scoring was run.

1. **Why failure?** AND mainly misses nonlinear joint calibration: guided k20 H omitted coefficients=[+0.00312,+0.01843,+0.03323,−0.17498], joint MAE0.27263. XOR misses both compensation and nonlinearity: [−0.18136,+0.37399,+0.30860,−0.77774], joint MAE0.81798. Both already fail on discovery, so generalization alone is not the explanation.
2. **Specific motif?** AND has opposing thresholded11 writes; native224 is retained but192/30 are omitted. Their separate loss deltas are positive while their joint delta is negative, exposing a coordination barrier in the greedy objective. XOR suggests negative00 suppression creating singles and negative joint, opposed by positive00 writes, plus negative11 subtraction and additive compensation. Exact q boundaries make those conditional motifs falsifiable.
3. **Could native-only find it?** Yes in principle from the same semantic coefficients/gates/signs. AND's historical native triple was already known. No measured information-budget advantage for student guidance exists here.
4. **Function-level grouping?** Supported as a useful explanatory description: native224/192/30 individual H joint SDs0.275/0.326/0.111 cancel to a sum SD0.043. Similar gate/sign families exist in both representations, but magnitudes, memberships and stability differ. Neither resemblance nor this post-hoc grouping establishes a recovered circuit, uniqueness, necessity or compact full-response fidelity.
5. **One next test:** a prospectively locked, equal-inspection-budget XOR family-transition prediction challenge, guided versus native-only. Predict aggregate native role-family four-corner changes including uninspected members and explicit remainders on newly permitted matched contexts. Scoring a known unit's exact ReLU formula alone would be tautological; full output reconstruction alone is insufficient. This experiment is proposed, not executed.

[Readable report](REPORT.md) · [Error curves](fixed_set_errors.png) · [Coefficient errors](omitted_coefficients.png) · [Motif accounting](conditional_motif_joint.png) · [Provenance](model_validation.json) · [Execution](execution.log)

All results remain exploratory diagnostics on the reused conditioned development banks. No compactness impossibility or broad nuisance-invariance claim follows. See preservation_verified.json and artifact_hashes.json for preservation and artifact records.
'''
 (OUT/'SUMMARY.md').write_text(brief)
 shutil.copyfile(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
 shutil.copyfile(OUT/'complete.json',OUT/'complete.initial.json');c=read(OUT/'complete.json');c.update({'report_sha256':sha(path),'summary_sha256':sha(OUT/'SUMMARY.md'),'report_only_finalization':now(),'finalization_source_sha256':sha(__file__),'diagnostic_exit_status':0});write(OUT/'complete.json',c)
 print('Report-only finalization complete; original data and locked sets unchanged')
if __name__=='__main__':main()
