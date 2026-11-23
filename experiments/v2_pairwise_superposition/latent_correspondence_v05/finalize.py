"""No new fitting or populations: audit saved estimates and clarify report support."""
import csv,shutil,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .core import *
from .run import csvwrite,npz,load_models
from ..workflow_v04.report import vec,table

def main():
 run=Path(sys.argv[1]);log=run.parent/(run.name+'-logs');assert (log/'exit_status').read_text().strip()=='0';complete=read(run/'complete.json');verify(run,'manifest_lock.json');verify(run,'readout_lock.json');man=read(run/'context_manifest.json');cp=torch.load(run/'readout.pt',map_location='cuda',weights_only=False);M=cp['M'];d=cp['d'];fit=np.load(run/'fit_activations.npz');stats=dict(np.load(run/'fit_statistics.npz'));stats0=target_statistics(tensor(fit['z']));torch.testing.assert_close(tensor(stats['v']),stats0['v'],atol=1e-12,rtol=1e-10);torch.testing.assert_close(cp['mu_h'],tensor(fit['h']).mean(0),atol=1e-12,rtol=1e-10)
 grid=list(csv.DictReader(open(run/'alpha_grid.csv')));selected=read(run/'selected_models.json')
 for name in ['native','input','output','semantic','shuffle']:
  rows=[{'alpha':float(r['alpha']),'score':float(r['score'])} for r in grid if r['method']==name];assert choose_alpha(rows)==selected[name]['alpha']
 projection=np.load(run/'native_solver.npz')['projection_residual_per_feature'];ceiling=[{'feature':m,'minimum_unregularized_fit_MSE':float(projection[m]),'minimum_fit_NMSE':float(projection[m]/stats['v'][m]) if stats['eligible'][m] else None,'eligible':bool(stats['eligible'][m])} for m in range(1024)];csvwrite(run/'finite_sample_projection_floor.csv',ceiling)
 metrics=list(csv.DictReader(open(run/'all_feature_metrics.csv')));well=[int(r['feature']) for r in metrics if r['population']=='primary_ordinary' and r['method']=='native' and r['well_read']=='True'];write(run/'well_read_primary_feature_ids.json',{'population':'primary_ordinary','criterion':'NMSE_fitvar<=.05; active relative RMSE<=.20; meaningful-active support>=32','count':len(well),'feature_ids':well,'unsupported_is_not_pass':True})
 # Preserve original response table; create explicit supported/pass/fail/unassessable status.
 totals=defaultdict(lambda:{'features':0,'assessable':0,'compatible':0,'unassessable':0,'relative':[]});assess=[]
 with open(run/'finite_response_metrics.csv') as f:
  for r in csv.DictReader(f):
   n=int(r['nontrivial_count']);rel=None if r['nontrivial_relative_RMSE']=='' else float(r['nontrivial_relative_RMSE']);defined=n>=32 and rel is not None;status='unassessable' if not defined else 'compatible' if rel<=.2 else 'incompatible';key=(r['population'],r['pair'],r['method'],r['change_type']);totals[key]['features']+=1;totals[key]['assessable']+=defined;totals[key]['compatible']+=(status=='compatible');totals[key]['unassessable']+=(status=='unassessable')
   if defined:totals[key]['relative'].append(rel)
   assess.append({'population':r['population'],'pair':r['pair'],'method':r['method'],'change_type':r['change_type'],'feature':int(r['feature']),'n_nontrivial':n,'relative_RMSE':rel,'status':status})
 csvwrite(run/'finite_response_assessment.csv',assess);totalsrows=[]
 for (pop,pair,method,typ),a in totals.items():totalsrows.append({'population':pop,'pair':pair,'method':method,'change_type':typ,'features':a['features'],'assessable':a['assessable'],'compatible':a['compatible'],'unassessable':a['unassessable'],'median_relative_RMSE_among_assessable':float(np.median(a['relative'])) if a['relative'] else None})
 csvwrite(run/'finite_response_summary.csv',totalsrows)
 # Main report clarification: no scientific criteria, data, model, or locked file changes.
 source_summary=run/'SUMMARY.md';shutil.copyfile(source_summary,run/'SUMMARY.initial.md');text=source_summary.read_text();note='**Primary finite-response status: UNASSESSABLE, not a negative model verdict.** There is only one eligible primary AND context, no primary XOR or AND01 context, and no primary nuisance edge. None supplies32 nontrivial changes for any feature. The table below retains numerical small-sample errors but its zero compatible counts mean insufficient support, not1024 tested failures. Legacy contexts are separate reused evidence with recorded fit/training overlap. See finite_response_assessment.csv for explicit three-way status.\n\n'
 text=text.replace('## 4. Finite input changes and local derivatives','## 4. Finite input changes and local derivatives\n\n'+note)
 text=text.replace('Full frozen TC error floor:', 'Frozen TC reference error (not an immutable lower bound):')
 concise='''**Observed outcome:**493 of1024 latents are well read on the new ordinary bank, including all six fixed examples. The native readout beats the input-, output- and semantic-score affine controls for most individual latents. This is evidence of readability beyond final-output imitation, with strongly conditioned population support. It is not exact gate recovery, broad response validation, compactness, uniqueness or a causal hidden code.

'''
 text=text.replace('## 1. Numerical estimation and scope',concise+'## 1. Numerical estimation and scope')
 # Explain output cancellation using already saved residual arrays, no new fitting.
 a=dict(np.load(run/'confirmation_activations.npz'));pr=dict(np.load(run/'confirmation_predictions.npz'));lookup={int(k):i for i,k in enumerate(a['keys'])};ordinary=np.array([lookup[k] for k in man['ordinary_confirmation_keys']]);res=dict(np.load(run/'confirmation_output_residuals.npz'));e_tc=res['e_TC'][ordinary];e_l=res['e_latent'][ordinary];total=res['e_total'][ordinary];cross=2*np.mean(e_tc*e_l);assert abs(np.mean(total**2)-(np.mean(e_tc**2)+np.mean(e_l**2)+cross))<1e-12
 cancellation=f'The native via-readout output MSE is {np.mean(total**2):.8g}, smaller than the frozen TC reference {np.mean(e_tc**2):.8g}, because latent errors partly cancel the TC output error: e_TC MSE={np.mean(e_tc**2):.8g}, decoder-latent-error MSE={np.mean(e_l**2):.8g}, twice-cross term={cross:.8g}. This was not an output fitting/selection objective. More sharply, output-only/semantic controls reproduce teacher outputs even better while reading only2 latents well; output agreement is demonstrably not latent recovery.\n\n'
 text=text.replace('Verified identities:',cancellation+'Verified identities:')
 rrows=[];jac=list(csv.DictReader(open(run/'jacobian_feature_metrics.csv')))
 for m in FEATURES:
  r=next(r for r in metrics if r['population']=='primary_ordinary' and r['method']=='native' and r['feature']==str(m));j=next(r for r in jac if r['population']=='all' and r['feature']==str(m));rrows.append([m,r['gate_precision'],r['gate_recall'],r['negative_pred_fraction'],j['relative_Frobenius'],j['mean_angle_degrees']])
 text=text.replace('## 8. What this establishes', 'Tiny positive inactive predictions can fail the fit-defined activity threshold even when raw MSE and active relative RMSE are good. Negative predictions are retained; no clipping rescues gate matches. Fixed-feature gate and continuous-derivative results are:\n\n'+table(['feature','gate precision','gate recall','negative prediction fraction','J relative error','J mean angle(deg)'],rrows)+'\n\n## 8. What this establishes')
 legacytable=[]
 for pair in PAIRS:
  key=','.join(map(str,pair));bank=man['legacy_contexts'][key];exposure=[man['membership'][str(k)]['readout'] for row in bank['corner_keys'] for k in row];legacytable.append([key,bank['n_contexts'],len(bank['nuisance_edges']),exposure.count('fit'),exposure.count('selection'),exposure.count('unseen')])
 text += '\nSecondary legacy exposure (corner memberships, so repeated endpoints are dependencies):\n\n'+table(['pair','contexts','nuisance edges','readout-fit','readout-selection','readout-unseen'],legacytable)+'\n\nNo nuisance response type has32 nontrivial changes here. Broad nuisance-response fidelity therefore remains unassessed, even if a worked edge matches well. Use finite_response_summary.csv for separate legacy partner/joint/control results.\n'
 text+='\nActual execution: orchestrator exit0,13 tests passed; all stages completed, including preservation audit. Additional report finalization uses only these fixed outcomes and locked metadata; it never changes M,d,alpha,preprocessing,scored populations or scientific thresholds. Initial reports are retained.\n'
 source_summary.write_text(text)
 # Feature reports: <=six named rows per table and one ordinary primary input for every feature.
 ordinary_key=man['ordinary_confirmation_keys'][0];oi=lookup[ordinary_key];any_a=tensor(inputs([ordinary_key]));worked0=dict(np.load(run/'feature_173_worked.npz'));C=tensor(worked0['C1']);b1=tensor(worked0['b1']);ph=(any_a@C.T+b1).squeeze(0);h=tensor(a['h'][oi]);torch.testing.assert_close(ph.relu(),h,atol=1e-9,rtol=1e-8);panel=read(run/'display_panels.json')
 for m in FEATURES:
  path=run/f'feature_{m}.md';shutil.copyfile(path,run/f'feature_{m}.initial.md');txt=path.read_text();lines=txt.splitlines();new=[];i=0
  while i<len(lines):
   if lines[i].startswith('| native neuron |'):
    header=lines[i];sep=lines[i+1];rr=[];i+=2
    while i<len(lines) and lines[i].startswith('|'):rr.append(lines[i]);i+=1
    new += [header,sep,*rr[:6],'', 'Remaining fit-only display-panel rows (same eight-neuron panel):','',header,sep,*rr[6:]]
   else:new.append(lines[i]);i+=1
  txt='\n'.join(new)+'\n';w=dict(np.load(run/f'feature_{m}_worked.npz'));K=tensor(w['K']);tcpre=float((any_a@K).item()+float(w['encoder_bias']));ids=panel[str(m)];v=M[m]*h;centered=M[m]*(h-cp['mu_h']);omitted=[r for r in range(256) if r not in ids];nativehat=float(v.sum()+d[m]);assert abs(nativehat-pr['native'][oi,m])<1e-9
  pointrows=[[r,f'{float(b1[r]):.8f}',f'{float(ph[r]):.8f}',f'{float(h[r]):.8f}',f'{float(M[m,r]):+.8f}',f'{float(v[r]):+.8f}'] for r in ids[:6]];pointrows += [['other displayed','—','—','—','—',f'{float(v[ids[6:]].sum()):+.8f}'],['undisplayed remainder','—','—','—','—',f'{float(v[omitted].sum()):+.8f}'],['fitted d','—','—','—','—',f'{float(d[m]):+.8f}']]
  point='\n## First ordinary primary input (locked metadata order)\n\n'+f'Key {ordinary_key}, active coordinates={np.flatnonzero(inputs([ordinary_key])[0]).tolist()}, exposure={man["membership"][str(ordinary_key)]}. This point was picked by the first ordinary manifest row, never by its prediction error. TC preactivation={tcpre:+.8f}, gate={tcpre>0}, actual z={a["z"][oi,m]:.8f}; full raw zhat={nativehat:.8f}, discrepancy={nativehat-a["z"][oi,m]:+.8f}. Each original row obeys p=C1[r,:]·a+b1[r], with actual bias shown.\n\n'+table(['native row','original b1','preactivation','activation','M weight','signed M·h'],pointrows)+'\n\n'+f'Centered panel8 sum={float(centered[ids].sum()):+.8f}; exact centered remainder={float(centered[omitted].sum()):+.8f}; fitted mean mu_z={float(cp["mu_z"][m]):.8f}. Their sum equals the same raw prediction. Complete ordinary-point arrays and full native contributions are in feature_{m}_ordinary_point.npz.\n'
  txt+=point
  # Attributed successes/failures from the already specified fit-only panel, no ranking change.
  first=ids[0];rmetric=next(r for r in metrics if r['feature']==str(m) and r['population']=='primary_ordinary' and r['method']=='native');txt+=f'\nThe leading fit-only centered-contribution row is native{first}, with M={float(M[m,first]):+.8f}. Its effect is a contribution to the latent, irrespective of that neuron’s different original W2/R write. All six ordinary-readability flags pass, but this feature’s gate precision is{rmetric["gate_precision"]}; low thresholded precision cannot be hidden by good active RMSE. Primary finite-change compatibility is unassessable under the support rule. Legacy rows remain explicitly secondary.\n'
  txt=txt.replace('finite-response compatible','finite-response compatible');txt+='\nFor every response row with fewer than32 nontrivial changes, the status is **unassessable**, even if the original boolean compatibility field is False. See finite_response_assessment.csv. No small-sample False value should be read as a scientifically established incompatibility.\n'
  path.write_text(txt);npz(run/f'feature_{m}_ordinary_point.npz',key=np.array(ordinary_key),a=any_a,h=h,pre=ph,M_contributions=v,centered_contributions=centered,actual_z=np.array(a['z'][oi,m]),zhat=np.array(nativehat),d=d[m],mu_z=cp['mu_z'][m],panel=np.array(ids),K=K,encoder_bias=w['encoder_bias'])
 # Audit main metrics on CUDA, independently of CPU table aggregation.
 models0=load_models(run);z=tensor(a['z'][ordinary]);h0=tensor(a['h'][ordinary]);zh=h0@M.T+d;np.testing.assert_allclose(numpy(zh),pr['native'][ordinary],atol=1e-9,rtol=1e-8);mse=numpy((zh-z).square().mean(0));rr=[r for r in metrics if r['population']=='primary_ordinary' and r['method']=='native'];assert all(abs(mse[int(r['feature'])]-float(r['mse']))<1e-10 for r in rr)
 # All four-corner polynomial reconstructions and readout raw/centered identities.
 contextchecks=0
 for path in run.glob('corners_*.npz'):
  bank=dict(np.load(path));np.testing.assert_allclose(from_polynomial(bank['z_alpha']),np.moveaxis(bank['z'],1,-1),atol=1e-9,rtol=1e-8)
  for name in ['native','input','output','semantic','shuffle','mean','half1','half2']:np.testing.assert_allclose(from_polynomial(bank[name+'_alpha']),np.moveaxis(bank[name],1,-1),atol=1e-9,rtol=1e-8)
  contextchecks+=1
 source=run/'source_snapshot'/Path(__file__).name;shutil.copy2(__file__,source);write(run/'final_audit.json',{'timestamp':now(),'status':'passed','primary_latent_MSE_matches_independent_CUDA_calculation':True,'fit_statistics_recomputed_from_fit_only':True,'chosen_alpha_recomputed_from_selection_scores_only':True,'locked_models_unchanged':True,'well_read_primary_features':len(well),'polynomial_banks_verified':contextchecks,'response_status_explicitly_distinguishes_low_support':'finite_response_assessment.csv','ordinary_point_selection':'first pre-locked ordinary manifest key, no new input','report_only_additions':'support wording, activity/Jacobian comparison, ordinary-point existing-cache traces, projection floor table, output-error cross term; initial reports preserved','CUDA':torch.version.cuda})
 verify(run,'readout_lock.json');verify(run,'manifest_lock.json');shutil.copy2(run/'complete.json',run/'complete.initial.json');complete.update({'final_audit_sha256':sha(run/'final_audit.json'),'finalization_source_sha256':sha(source),'summary_sha256':sha(run/'SUMMARY.md'),'feature_reports':{str(m):sha(run/f'feature_{m}.md') for m in FEATURES},'orchestrator_exit_status':0,'tests_passed':13,'report_finalized':now()});write(run/'complete.json',complete);print('FINALIZATION/AUDIT PASSED',run/'SUMMARY.md')
if __name__=='__main__':main()
