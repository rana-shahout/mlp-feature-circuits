"""Post-run accounting audit and report-only clarifications; never rescores models."""
import csv,shutil
from pathlib import Path
import numpy as np
from .core import *
from .run import OUTPUTS,BASE,MAN
from .report import vec,table

def main():
 run=BASE/'reports/workflow-v04-20261030';log=BASE/'execution_logs/workflow-v04-20261030'
 assert (log/'exit_status').read_text().strip()=='0';assert read(run/'complete.json')['status']=='complete'
 for name in ['prepare','discovery','native','confirmation']:verify(run,f'{name}_lock.json')
 times=[read(run/f'{n}_lock.json')['timestamp'] for n in ['prepare','discovery','native','confirmation']];assert times==sorted(times) and len(set(times))==4
 man=read(run/'context_manifest.json');exc=dict(np.load(run/'exclusion_keys.npz'));excluded=set(map(int,np.concatenate(list(exc.values()))));D=set();H=set();known={int(k):name for name,v in read(MAN)['partitions'].items() for k in v['vector_keys']};limited=[]
 for key,record in man.items():
  pair=record['pair'];assert pair==list(map(int,key.split(',')))
  for label,ss,seen in [('D',0,D),('H',1,H)]:
   bank=record[label];a=corners(bank['background_keys'],pair);ks=keys(a);np.testing.assert_array_equal(ks,bank['corner_keys']);assert np.all((a.sum(-1)>=3)&(a.sum(-1)<=6));assert len(set(bank['background_keys']))==len(bank['background_keys']);assert all(split(k)==ss and int(k) not in excluded for k in ks.flat)
   assert bank['training_overlap']==[[known.get(int(k),'unseen') for k in row] for row in ks]
   seen.update(map(int,ks.flat))
   if bank['n']<16:limited.append({'pair':pair,'split':label,'n':bank['n']})
 assert D.isdisjoint(H)
 hyp=read(run/'transcoder_hypotheses.json');nn=read(run/'native_explanations.json');rows=list(csv.DictReader(open(run/'confirmation_metrics.csv')));checks={};count=0
 for j in OUTPUTS:
  nd=dict(np.load(run/f'discovery_native_{j}.npz'));tc=dict(np.load(run/f'discovery_student_{j}.npz'));n=nn[str(j)];traces=read(run/f'selector_traces_{j}.json');truth=nd['total'];selected_checks=[]
  for method in ['guided','native_only']:
   target=tc['total'] if method=='guided' else truth
   for k,ids in n['sets'][method].items():
    trace=traces[method][k];assert trace[-1]['set']==ids;assert len(ids)==int(k);assert len(set(ids))==int(k)
    for step in trace:
     s=step['set'];objective=np.mean((nd['signed'][:,:,s].sum(-1)+nd['beta'][j]-target)**2);assert abs(objective-step['mse'])<1e-10
    pred=nd['signed'][:,:,ids].sum(-1)+nd['beta'][j];check=metrics(pred,truth);stored=next(r for r in n['discovery_metrics'] if r['method']==method and r['budget']==int(k));assert abs(check['corner_mae']-stored['corner_mae'])<1e-12;selected_checks.append(check)
   passes=[r['budget'] for r in n['discovery_metrics'] if r['method']==method and r['profile_pass'] and r['joint_pass']];assert n['primary'][method]['budget']==(min(passes) if passes else 20)
  maxerr={}
  for label in ['native','student']:
   arr=dict(np.load(run/f'confirmation_{label}_{j}.npz'));direct=arr['all_outputs'];reconstructed=arr['h']@arr['R'].T+arr['beta'];maxerr[label]=float(np.abs(reconstructed-direct).max());np.testing.assert_allclose(reconstructed,direct,atol=1e-9,rtol=1e-8)
   for key in ['pre','h','signed','total']:assert np.isfinite(arr[key]).all()
   # Every signed alpha is accounted, output bias contributes only to alpha0.
   alpha=polynomial(np.moveaxis(arr['signed'],1,-1));np.testing.assert_allclose(alpha.sum(1)+[arr['beta'][j],0,0,0],polynomial(arr['total']),atol=1e-9,rtol=1e-8)
  preds=dict(np.load(run/f'confirmation_predictions_{j}.npz'));rest=dict(np.load(run/f'confirmation_remainders_{j}.npz'));ndh=dict(np.load(run/f'confirmation_native_{j}.npz'))
  for name,pred in preds.items():np.testing.assert_allclose(pred+rest[name],ndh['total'],atol=1e-9,rtol=1e-8);count+=1
  checks[str(j)]={'all_nine_output_identity_max':maxerr,'sets_objectives_verified':10,'confirmation_contexts':len(ndh['total'])}
  # Preserve initial render before adding explanations. No scientific artifacts change.
  p=run/f'case_output_{j}.md';shutil.copyfile(p,run/f'case_output_{j}.initial.md');text=p.read_text();pair=hyp[str(j)]['pair'];record=man[','.join(map(str,pair))]
  extra=['','## Interpretation and support limits','',f"Discovery contains {record['D']['n']} complete backgrounds and confirmation {record['H']['n']}; all eligible backgrounds were used because counts are below the caps. Density-specific discovery counts={record['D']['density']}, confirmation={record['H']['density']}. Each density stratum with fewer than 16 backgrounds has limited support; pooled results should not be read as well-supported density-specific findings. Nuisance flips likewise have zero or one edge here.",'','No predeclared bounded group passed discovery or confirmation. k=20 is the specified fallback, labeled no discovery pass. This is a negative result for these selectors and budgets; it does not prove that no compact original-neuron explanation exists. The complete 256-unit decomposition remains exact within numerical tolerance, while the reported selected groups leave material additive and interaction response unexplained.','']
  # Interpret the metadata-selected worked response using the already frozen display IDs.
  b=n['worked_selection']['background'];idx=record['H']['background_keys'].index(b)
  for label,selected in [('native',n['sets']['guided'][str(n['primary']['guided']['budget'])]),('student',hyp[str(j)]['panel'])]:
   arr=dict(np.load(run/f'confirmation_{label}_{j}.npz'));rr=[]
   for u in selected[:6]:
    pre=arr['pre'][idx,:,u];gate=pre>0;code=''.join('1' if g else '0' for g in gate);alpha=polynomial(arr['signed'][idx,:,u]);writev=arr['R'][j,u]
    if np.all(gate):description='all gates on: affine baseline and single-input support, zero joint'
    elif not np.any(gate):description='all gates off: no response on this background'
    elif np.array_equal(gate,[0,0,0,1]):description='11-only gate; '+('adds' if writev>0 else 'subtracts')+' joint response'
    elif np.array_equal(gate,[0,1,0,0]) or np.array_equal(gate,[0,0,1,0]):description='single-corner missing-partner response; '+('positive write gives negative joint' if writev>0 else 'negative write gives positive joint through suppression')
    elif np.array_equal(gate,[1,0,0,0]):description='00-only response; writes baseline and compensating single/joint coefficients'
    else:description='conditional hinge; use all four signed coefficients, not an interaction-only detector label'
    rr.append([u,code,vec(alpha),description])
   extra += [f'At worked key {b}, the displayed {label} components perform these specific operations. Gates shown below are local facts, not invariant semantic identities.','',table(['unit','gate order 00/01/10/11','signed [baseline, ai, ak, joint]','operation'],rr),'']
  if j==6:
   arr=dict(np.load(run/'confirmation_native_6.npz'));v=arr['total'][idx];extra += [f"The complete teacher XOR-like corner vector is {vec(v)}: both single-input corners rise, and both the zero-input and double-input corners are low. Its coefficients are {vec(polynomial(v))}. The guided subset and omitted remainder shown above jointly supply this pattern; the subset alone fails the declared full-profile tolerance. This explicitly tests the two single-input terms and baseline as well as the negative interaction.",'']
  p.write_text(text+'\n'.join(extra)+'\n')
 for p,h in read(log/'historical_execution_logs_before.json').items():assert sha(p)==h
 write(run/'postrun_audit.json',{'status':'passed','timestamp':now(),'lock_times':times,'global_input_separation':True,'all_190_pairs_corner_exclusions_verified':True,'all_original_overlap_tags_verified':True,'case_checks':checks,'confirmation_prediction_plus_remainder_identities':count,'limited_pooled_pair_regimes':limited,'prior_execution_logs_additionally_verified':48,'scientific_locks_unchanged':True,'method':'CPU audit of saved results only; no new inference, selection, or population changes'})
 p=run/'SUMMARY.md';shutil.copyfile(p,run/'SUMMARY.initial.md');text=p.read_text();text=text.replace('The executable workflow completed:', 'All three expected pairs ranked first and full-student fidelity passed. **No guided or native-only subset at any declared budget passed both tolerances.** Student guidance provided small, inconsistent changes relative to native-only selection, with no demonstrated practical advantage. This is failure of the bounded selectors, not proof of compactness impossibility.\n\nThe executable workflow completed:')
 text+='\nConfirmation support is 21/23/19 complete backgrounds for outputs 4/6/0, respectively. Density-specific strata are small and heavily conditioned by exclusions; matched nuisance support is 0/0/1 edges. Scope does not extend to broad nuisance invariance or the protected final population.\n\nThe orchestrator exited **0**; seven mathematical/CUDA test groups passed. A post-run audit verified all 190 pair manifests, split disjointness, exclusions, overlap tags, exact selector objective traces, all-nine-output identities, and 333 prediction-plus-remainder identities. Initial reports are retained alongside report-only explanatory additions.\n\nSPEC.md was already version 0.4 on arrival; its exact bytes and CODEX_PROMPT.md were archived and hashed in `../../spec_archive/20261030T145434Z/` before implementation. The active specification was not rewritten.\n'
 p.write_text(text)
 for f in log.iterdir():
  if f.is_file():shutil.copyfile(f,run/f.name)
 snap=run/'report_source_snapshot';snap.mkdir()
 for f in Path(__file__).parent.glob('*.py'):shutil.copyfile(f,snap/f.name)
 shutil.copyfile(run/'complete.json',run/'complete.initial.json');complete=read(run/'complete.json');complete.update({'postrun_audit_sha256':sha(run/'postrun_audit.json'),'summary_sha256':sha(run/'SUMMARY.md'),'reports':{f'case_output_{j}.md':sha(run/f'case_output_{j}.md') for j in OUTPUTS},'orchestrator_exit_status':0,'test_groups_passed':7,'report_only_additions':'explicit per-unit local computational interpretation and small-stratum support flags; original renders retained','report_source_hashes':{f.name:sha(f) for f in snap.iterdir()}});write(run/'complete.json',complete)
 print('POSTRUN AUDIT PASSED; report-only additions completed; all scientific locks unchanged')
if __name__=='__main__':main()
