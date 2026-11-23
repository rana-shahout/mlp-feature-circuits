"""Audit existing fixed sets and partition all units by gate/sign motifs. No search or fit."""
import csv,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..workflow_v04.core import read,write,sha,raw,now,verify,corners,metrics,polynomial,cpu,evaluate
from ..workflow_v04.run import student,teacher,embeddings,DATA,MAN
from .core import alpha,gate_codes,modal_groups,LABELS,partition,signed_stats,polynomial_error_parts,error_alignment,loss_delta
BASE=Path('experiments/v2_pairwise_superposition');OLD=BASE/'reports/workflow-v04-20261030';OUT=BASE/'diagnostics/workflow-failures-v1-20261030';LOG=BASE/'execution_logs/workflow-failures-v1-20261030';CASES={4:(7,8),6:(11,12)}
SOURCES=[OLD/'SUMMARY.md',*[OLD/f'case_output_{j}.md' for j in [4,6,0]],BASE/'traces/and78-two-backgrounds-20261028/report.md',BASE/'reports/stage2-transcoders-20261027/report.md']
def csvwrite(p,rows):
 rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def array(prefix,label,j):return dict(np.load(OLD/f'{prefix}_{label}_{j}.npz'))
def vectorfields(prefix,v):return {prefix+str(i):float(x) for i,x in enumerate(v)}
def protect():
 roots=[BASE/'checkpoints',BASE/'data',BASE/'reports',BASE/'runs',BASE/'traces',BASE/'execution_logs',Path('results'),Path('logs')]
 return {str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file() and LOG not in p.parents}
@torch.no_grad()
def validate_models():
 assert torch.cuda.is_available(),'CUDA is mandatory';p,q=embeddings();t=teacher(OLD);s=student(OLD);prot=read(OLD/'protocol.json');checks=[]
 assert raw(p.float().cpu().numpy())==prot['P_raw_sha256'];assert raw(q.float().cpu().numpy())==prot['Q_raw_sha256'];assert sha(DATA)==prot['hashes'][str(DATA)];assert sha(MAN)==prot['hashes'][str(MAN)]
 man=read(OLD/'context_manifest.json')
 for j,pair in CASES.items():
  for label,m in [('native',t),('student',s)]:
   for prefix,split in [('discovery','D'),('confirmation','H')]:
    bg=man[','.join(map(str,pair))][split]['background_keys'];saved=array(prefix,label,j);d=cpu(evaluate(m,p,q,bg,pair,j));errs={}
    for key in ['C','R','b','beta','pre','h','signed','total','q','all_outputs']:
     np.testing.assert_allclose(d[key],saved[key],atol=1e-9,rtol=1e-8);errs[key]=np.max(np.abs(d[key]-saved[key]))
    a=torch.tensor(corners(bg,pair),device='cuda',dtype=torch.float32);x=a@p.float().T;m.float()
    direct=(m(x.reshape(-1,12))@q.float()).reshape(len(bg),4,9).cpu().numpy();m.double();np.testing.assert_allclose(direct,saved['all_outputs'],atol=2e-6,rtol=2e-5)
    checks.append({'output':j,'model':label,'split':split,'exact_saved_identity_max':errs,'FP32_vs_saved64_max':np.max(np.abs(direct-saved['all_outputs'])),'n':len(bg)})
 return {'checks':checks,'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'models':{'teacher':prot['teacher'],'student':prot['student']},'checkpoint_hashes':{k:v for k,v in prot['hashes'].items() if k.endswith('.pt')},'P_raw_sha256':prot['P_raw_sha256'],'Q_raw_sha256':prot['Q_raw_sha256'],'identity_tolerance':{'atol':1e-9,'rtol':1e-8},'FP32_tolerance':{'atol':2e-6,'rtol':2e-5},'inference_inputs':'Only original D/H context manifests for outputs4/6; no reserved file indexed or loaded by diagnostics'}

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;OUT.mkdir(parents=True,exist_ok=False)
 protected=protect();write(OUT/'protected_before.json',protected)
 for name in ['prepare','discovery','native','confirmation']:verify(OLD,f'{name}_lock.json')
 for path,h in read(OLD/'artifact_hashes.json').items():assert sha(OLD/path)==h,path
 snap=OUT/'source_snapshot';snap.mkdir()
 files=[*Path(__file__).parent.glob('*.py'),BASE/'workflow_v04/core.py',BASE/'workflow_v04/run.py',BASE/'trace/core.py',Path('src/models.py'),Path('src/runtime.py'),Path('SPEC.md')]
 for p in files:shutil.copyfile(p,snap/str(p).replace('/','__'))
 write(OUT/'protocol.json',{'version':'diagnostic-v1','created':now(),'cases':CASES,'scope':'post-hoc diagnosis on previously scored original D/H contexts, not new confirmation','fixed_sets':'all existing guided/native_only budgets1/3/5/10/20; no calls to selector','sign_definitions':'write sign and polynomial coefficient sign are separate exports','motifs':'all units partitioned by discovery modal exact gate mask and signed output write, smallest mask breaks modal ties; memberships remain fixed for H; instantaneous motifs are separate contextual accounting','reference_native_AND':[224,192,30],'reference_student_AND':[173,266],'worst_errors':'post-hoc three largest original guided k20 absolute corner residuals per split, flattened context/corner tie order; descriptive, no selection','no_refit':True,'no_intervention':True,'new_contexts':0,'figures':'saved data only; no new model population','source_material_hashes':{str(p):sha(p) for p in SOURCES},'source_hashes':{str(p):sha(p) for p in files},'original_locks':{name:sha(OLD/f'{name}_lock.json') for name in ['prepare','discovery','native','confirmation']}})
 (OUT/'git_status.txt').write_text(subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout);(OUT/'gpu.txt').write_text(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
 write(OUT/'model_validation.json',validate_models());print('CUDA provenance and all saved tensors validated',flush=True)
 sets=read(OLD/'native_explanations.json');man=read(OLD/'context_manifest.json');allmetrics=[];coefs=[];omitted_stats=[];unitstats=[];motifs=[];fixedgroups=[];alignmentrows=[];worst=[];tracerows=[];reference=[];barriers=[];caseinfo={};profiles={}
 for j,pair in CASES.items():
  caseinfo[str(j)]={};discovery={label:array('discovery',label,j) for label in ['native','student']};groups={label:modal_groups(a['pre'],a['R'][j]) for label,a in discovery.items()}
  for prefix,split in [('discovery','D'),('confirmation','H')]:
   n=array(prefix,'native',j);tc=array(prefix,'student',j);bank=man[','.join(map(str,pair))][split];bg=bank['background_keys'];count=len(bg);v=n['signed'];truth=n['total'];alph=alpha(v);bias=np.array([n['beta'][j],0,0,0]);details={};primary=sets[str(j)]['sets']['guided']['20']
   for method in ['guided','native_only']:
    for budget in [1,3,5,10,20]:
     ids=sets[str(j)]['sets'][method][str(budget)];omitted,r=partition(v,ids);pred=n['beta'][j]+v[:,:,ids].sum(-1);np.testing.assert_allclose(r,truth-pred,atol=1e-10,rtol=1e-8);ar=alph[:,omitted].sum(1);np.testing.assert_allclose(ar,polynomial(r),atol=1e-10,rtol=1e-8)
     met=metrics(pred,truth);target=tc['total'] if method=='guided' else truth;row={'output':j,'split':split,'method':method,'budget':budget,'ids':ids,'target_MSE':np.mean((pred-target)**2),'teacher_MSE':np.mean((pred-truth)**2),'student_vs_teacher_MSE':np.mean((tc['total']-truth)**2),**met,**polynomial_error_parts(r)};allmetrics.append(row)
     for label,a in [('teacher',polynomial(truth)),('selected_with_actual_bias',alph[:,ids].sum(1)+bias),('omitted',ar)]:
      for z in range(4):coefs.append({'output':j,'split':split,'method':method,'budget':budget,'term':label,'coefficient':z,'mean':a[:,z].mean(),'mean_abs':np.abs(a[:,z]).mean(),'sd':a[:,z].std(),'min':a[:,z].min(),'max':a[:,z].max()})
     stat=signed_stats(alph[:,omitted]);writepos=v[:,:,omitted][:,:,n['R'][j,omitted]>0].sum(-1);writeneg=v[:,:,omitted][:,:,n['R'][j,omitted]<0].sum(-1)
     np.testing.assert_allclose(writepos+writeneg,r,atol=1e-10,rtol=1e-8)
     omitted_stats.append({'output':j,'split':split,'method':method,'budget':budget,**{k:value.tolist() for k,value in stat.items()},'positive_write_corner_mean':writepos.mean(0).tolist(),'negative_write_corner_mean':writeneg.mean(0).tolist(),'positive_write_alpha_mean':polynomial(writepos).mean(0).tolist(),'negative_write_alpha_mean':polynomial(writeneg).mean(0).tolist()})
     align=error_alignment(v[:,:,omitted],r);assert abs(align.sum()-np.mean(r*r))<1e-10
     for u,val in zip(omitted,align):alignmentrows.append({'output':j,'split':split,'method':method,'budget':budget,'omitted_unit':int(u),'signed_MSE_alignment':val,'mean_corner_contribution':v[:,:,u].mean(),'mean_abs_corner_contribution':np.abs(v[:,:,u]).mean(),**vectorfields('mean_alpha',alph[:,u].mean(0))})
     if method=='guided' and budget==20:
      details['primary_remainder_mean_alpha']=ar.mean(0);details['primary_remainder_abs_alpha']=np.abs(ar).mean(0);details['primary_metrics']=row;details['omitted_signed_stats']=omitted_stats[-1];details['top_omitted_units_per_alpha']={str(z):[{'unit':int(u),'mean':float(alph[:,u,z].mean()),'mean_abs':float(np.abs(alph[:,u,z]).mean())} for u in sorted(omitted,key=lambda u:(-np.abs(alph[:,u,z]).mean(),u))[:6]] for z in range(4)}
      # Largest errors are diagnosed post hoc; they are not a new evidence bank.
      for ix in np.argsort(-np.abs(r).reshape(-1),kind='stable')[:3]:
       c,t=divmod(int(ix),4);contributors=sorted(omitted,key=lambda u:(-abs(v[c,t,u]),u));rec={'output':j,'split':split,'context_index':c,'background_key':bg[c],'active_background':np.flatnonzero(np.array([(bg[c]>>i)&1 for i in range(20)])).tolist(),'corner':['00','01','10','11'][t],'teacher':truth[c,t],'selected':pred[c,t],'remainder':r[c,t],'positive_omitted':np.maximum(v[c,t,omitted],0).sum(),'negative_omitted':np.minimum(v[c,t,omitted],0).sum(),'remainder_alpha':ar[c].tolist(),'top_omitted':[]}
       for u in contributors[:6]:rec['top_omitted'].append({'unit':int(u),'write':n['R'][j,u],'q':n['q'][c,u],'u':n['C'][u,pair[0]],'v':n['C'][u,pair[1]],'bias':n['b'][u],'gate':int(gate_codes(n['pre'])[c,u]),'pre_corners':n['pre'][c,:,u].tolist(),'signed_corners':v[c,:,u].tolist(),'signed_alpha':alph[c,u].tolist(),'at_error_corner':v[c,t,u]})
       worst.append(rec)
      profiles[f'{j}_{split}_primary_remainder']=r
     if split=='D':
      traces=read(OLD/f'selector_traces_{j}.json')[method][str(budget)];last=None
      for stepno,step in enumerate(traces):
       ss=step['set'];loss=np.mean((n['beta'][j]+v[:,:,ss].sum(-1)-target)**2);assert abs(loss-step['mse'])<1e-10
       if step['operation']=='swap' and last is not None:assert loss<last+1e-10
       tracerows.append({'output':j,'method':method,'budget':budget,'step':stepno,'operation':step['operation'],'ids':ss,'recorded_MSE':step['mse'],'recomputed_MSE':loss,'change_from_previous':loss-last if last is not None else None});last=loss
      assert traces[-1]['set']==ids
   # Full units: exact coefficients, gate offset dependence, all corner responses.
   for label,a in [('native',n),('student',tc)]:
    aa=alpha(a['signed']);codes=gate_codes(a['pre']);modal,sign=groups[label];nunits=len(a['b']);fullstats=signed_stats(aa);details[label+'_signed_stats']=fullstats
    detailed=[]
    for u in range(nunits):
     row={'output':j,'split':split,'model':label,'unit':u,'bias':a['b'][u],'u':a['C'][u,pair[0]],'v':a['C'][u,pair[1]],'write':a['R'][j,u],'complete_input_row':a['C'][u].tolist(),'all9_output_writes':a['R'][:,u].tolist(),'discovery_modal_mask':int(modal[u]),'discovery_group_sign':int(sign[u]),'q_min':a['q'][:,u].min(),'q_max':a['q'][:,u].max(),'q_sd':a['q'][:,u].std(),'modal_failure_fraction':np.mean(codes[:,u]!=modal[u]),'nearzero_margin_fraction':np.mean(np.abs(a['pre'][:,:,u])<=1e-7),'gate_counts':np.bincount(codes[:,u],minlength=16).tolist(),'mean_corners':a['signed'][:,:,u].mean(0).tolist(),**vectorfields('mean_alpha',aa[:,u].mean(0)),**vectorfields('mean_abs_alpha',np.abs(aa[:,u]).mean(0))};unitstats.append(row)
     for c in range(count):detailed.append({'context_index':c,'background_key':bg[c],'unit':u,'q':a['q'][c,u],'gate_mask':int(codes[c,u]),'pre_corners':a['pre'][c,:,u].tolist(),'activation_corners':a['h'][c,:,u].tolist(),'signed_corners':a['signed'][c,:,u].tolist(),'signed_alpha':aa[c,u].tolist()})
    csvwrite(OUT/f'components_output{j}_{label}_{split}.csv',detailed)
    for mask in sorted(set(codes.flat)):
     for sg in [-1,1]:
      include=(codes==mask)&(np.sign(a['R'][j])==sg)[None,:];response=(a['signed']*include[:,None,:]).sum(-1);aa_group=polynomial(response);positive=np.maximum(aa,0)*include[:,:,None];negative=np.minimum(aa,0)*include[:,:,None]
      motifs.append({'output':j,'model':label,'split':split,'mask':int(mask),'label':LABELS.get(int(mask),'mixed hinge'),'write_sign':sg,'mean_units_per_context':include.sum(1).mean(),'mean_corner':response.mean(0).tolist(),'mean_alpha':aa_group.mean(0).tolist(),'positive_alpha_mean':positive.sum(1).mean(0).tolist(),'negative_alpha_mean':negative.sum(1).mean(0).tolist()})
    group_arrays={}
    for mask in sorted(set(modal)):
     for sg in [-1,1]:
      ids=np.flatnonzero((modal==mask)&(sign==sg))
      if not len(ids):continue
      vv=a['signed'][:,:,ids].sum(-1);ag=polynomial(vv);group_arrays[f'mask{mask}_sign{sg}']=vv;fixedgroups.append({'output':j,'model':label,'split':split,'discovery_modal_mask':int(mask),'label':LABELS.get(int(mask),'mixed hinge'),'write_sign':sg,'unit_count':len(ids),'ids':ids.tolist(),'mean_corner':vv.mean(0).tolist(),'sd_corner':vv.std(0).tolist(),'mean_alpha':ag.mean(0).tolist(),'sd_alpha':ag.std(0).tolist(),'modal_gate_retention':np.mean(codes[:,ids]==mask)})
    summed=sum(group_arrays.values());np.testing.assert_allclose(summed+a['beta'][j],a['total'],atol=1e-9,rtol=1e-8);np.savez_compressed(OUT/f'motif_profiles_output{j}_{label}_{split}.npz',**group_arrays)
    if j==4:
     refids=[224,192,30] if label=='native' else [173,266]
     for u in refids:reference.append({'model':label,'split':split,'unit':u,'in_guided20':u in primary if label=='native' else None,'bias':a['b'][u],'u':a['C'][u,pair[0]],'v':a['C'][u,pair[1]],'write':a['R'][j,u],'q_range':[a['q'][:,u].min(),a['q'][:,u].max()],'11only_fraction':np.mean(codes[:,u]==8),'gate_counts':np.bincount(codes[:,u],minlength=16).tolist(),'mean_corner':a['signed'][:,:,u].mean(0).tolist(),'mean_alpha':aa[:,u].mean(0).tolist(),'sd_joint':aa[:,u,3].std()})
     rg=aa[:,refids,3];details[label+'_reference_joint']={'ids':refids,'mean':rg.mean(0),'individual_sd':rg.std(0),'sum_sd':rg.sum(1).std(),'sum_mean':rg.sum(1).mean(),'covariance':np.cov(rg,rowvar=False)}
   # Only a pre-named historical diagnostic pair, not a new search or proposed subset.
   if j==4 and split=='D':
    for method in ['guided','native_only']:
     target=tc['total'] if method=='guided' else truth
     for stage,ids in [('first_recorded_add',read(OLD/'selector_traces_4.json')[method]['20'][0]['set']),('frozen_k20',sets['4']['sets'][method]['20'])]:
      rr=target-(n['beta'][j]+v[:,:,ids].sum(-1));x,y=v[:,:,192],v[:,:,30];paircross=2*np.mean(x*y)
      barriers.append({'method':method,'stage':stage,'recorded_ids':ids,'MSE':np.mean(rr**2),'192_is_omitted':192 not in ids,'30_is_omitted':30 not in ids,'192_only_delta':loss_delta(rr,x),'30_only_delta':loss_delta(rr,y),'historical_192plus30_delta':loss_delta(rr,x+y),'twice_cross_192_30':paircross,'interpretation':'fixed historical contribution loss algebra only; no neurons changed, no replacement set selected'})
   caseinfo[str(j)][split]=details
  print('Diagnostic tables complete for output',j,flush=True)
 csvwrite(OUT/'fixed_set_errors.csv',allmetrics);csvwrite(OUT/'coefficient_accounting.csv',coefs);csvwrite(OUT/'omitted_signed_accounting.csv',omitted_stats);csvwrite(OUT/'unit_summary.csv',unitstats);csvwrite(OUT/'instantaneous_motifs.csv',motifs);csvwrite(OUT/'fixed_discovery_motif_groups.csv',fixedgroups);csvwrite(OUT/'omitted_error_alignment.csv',alignmentrows);csvwrite(OUT/'objective_trace_audit.csv',tracerows);csvwrite(OUT/'AND_historical_references.csv',reference);csvwrite(OUT/'AND_historical_pair_objective_algebra.csv',barriers)
 write(OUT/'worst_errors.json',worst);write(OUT/'diagnostic_summary.json',caseinfo);write(OUT/'objective_audit.json',{'definition':'L(S)=(1/(4N))*sum_c,t (beta + sum_r_in_S R[j,r]*ReLU(C1[r,:] a+b1[r]) - target)^2; guided target full student, native_only full teacher. No normalization of polynomial coefficient channels. Actual beta kept; no intercept or coefficient fit.','implementation':'each budget independent deterministic forward greedy then <=5 single improving swaps; tie index order; tolerance1e-12. Pure corner MSE neither explicitly preserves signs/motifs nor optimizes joint threshold.','intermediate_objectives_recomputed':len(tracerows),'original_source':str(BASE/'workflow_v04/core.py'),'source_hash':sha(BASE/'workflow_v04/core.py'),'selection_rerun':False,'native_group_memberships_changed':False,'readout_fitted':False,'historical_pair_algebra':barriers})
 figures(OUT,allmetrics,coefs,motifs,reference,caseinfo)
 from .report import report
 report(OUT,caseinfo,allmetrics,coefs,omitted_stats,unitstats,motifs,fixedgroups,alignmentrows,worst,reference,barriers)
 assert all(sha(path)==h for path,h in protected.items()),'Previous artifact changed'
 write(OUT/'preservation_verified.json',{'timestamp':now(),'protected_files_unchanged':len(protected),'reserved_population':'no scoring; no reserved archive arrays loaded; whole-file hash only for preservation','models_trained':0,'selectors_executed':0,'new_cases':0,'new_contexts':0})
 for p in LOG.iterdir():
  if p.is_file():shutil.copyfile(p,OUT/p.name)
 write(OUT/'complete.json',{'status':'complete','timestamp':now(),'cases':[4,6],'protected_files_verified':len(protected),'report_sha256':sha(OUT/'REPORT.md'),'identity_checks':8,'tests':'11 passed before diagnostics','no_new_groups_optimized':True});print('COMPLETE',OUT/'REPORT.md',flush=True)

def figures(out,ms,cs,motifs,refs,summary):
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,j in zip(axes,[4,6]):
  for method in ['guided','native_only']:
   for split,style in [('D','--'),('H','-')]:
    rows=[r for r in ms if r['output']==j and r['method']==method and r['split']==split];ax.plot([r['budget'] for r in rows],[r['corner_mae'] for r in rows],style,marker='o',label=method+' '+split)
  ax.axhline(.05,color='gray',ls=':');ax.set(title=f'Output {j}: existing fixed sets',xlabel='original budget',ylabel='corner MAE');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(out/'fixed_set_errors.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,j in zip(axes,[4,6]):
  for n,split in enumerate(['D','H']):
   a=summary[str(j)][split]['primary_remainder_mean_alpha'];ax.bar(np.arange(4)+n*.35,a,.35,label=split)
  ax.axhline(0,color='black',lw=.5);ax.set_xticks(np.arange(4)+.175,['baseline','first','second','joint']);ax.set(title=f'Output {j}: teacher − guided k20',ylabel='mean omitted coefficient');ax.legend()
 fig.tight_layout();fig.savefig(out/'omitted_coefficients.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,j in zip(axes,[4,6]):
  labels=['11-only','00-only','i-only / missing-k','k-only / missing-i','i-supported','k-supported','always-on affine','any-input-on hinge']
  for index,model in enumerate(['native','student']):
   values=[sum(r['mean_alpha'][3] for r in motifs if r['output']==j and r['model']==model and r['split']=='H' and r['label']==label) for label in labels];ax.bar(np.arange(len(labels))+index*.35,values,.35,label=model)
  ax.set_xticks(np.arange(len(labels))+.175,labels,rotation=40,ha='right');ax.set(title=f'Output {j}: conditional gate accounting',ylabel='mean signed joint contribution');ax.axhline(0,color='black',lw=.5);ax.legend()
 fig.tight_layout();fig.savefig(out/'conditional_motif_joint.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
