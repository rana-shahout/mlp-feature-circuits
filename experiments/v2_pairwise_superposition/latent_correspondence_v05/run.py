import argparse,csv,json,shutil,subprocess,sys,time
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from src.models import SparseTranscoder
from src.runtime import load_checkpoint
from ..stage2.core import frozen_teacher
from .core import *
BASE=Path('experiments/v2_pairwise_superposition');OLD=BASE/'reports/workflow-v04-20261030';DATA=BASE/'data/first-teacher-20261027/dataset.npz';MAN=BASE/'data/first-teacher-20261027/split_manifest.json'

def csvwrite(p,rows):
 rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def npz(p,**kw):np.savez_compressed(p,**{k:numpy(v) for k,v in kw.items()})
def models(run):
 prot=read(run/'protocol.json');tp=Path(prot['teacher']);sp=Path(prot['student']);assert sha(tp)==prot['hashes'][str(tp)];assert sha(sp)==prot['hashes'][str(sp)]
 t,cp=frozen_teacher(tp,'cuda');cp2=load_checkpoint(sp);assert cp2['width']==1024 and cp2['lambda']==1e-4;assert cp2['identity']['teacher_sha256']==sha(tp);assert cp2['identity']['dataset_sha256']==sha(DATA)
 s=SparseTranscoder(12,1024,12).cuda();s.load_state_dict(cp2['model']);s.eval().requires_grad_(False)
 with np.load(DATA) as a:p,q=tensor(a['P']),tensor(a['Q'])
 assert p.shape==(12,20) and q.shape==(12,9);assert raw_np(p.float().cpu().numpy())==prot['P_raw_sha256'];assert raw_np(q.float().cpu().numpy())==prot['Q_raw_sha256']
 return t.double(),s.double(),p,q

def raw_np(a):return __import__('hashlib').sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def choose_bank(eligible,pair,rng):
 pool=set(map(int,eligible));bases={3:[],4:[]};other=[i for i in range(20) if i not in pair]
 for size in [3,4]:
  for active in combinations(other,size):
   b=sum(1<<i for i in active);ks=[b,b|(1<<pair[1]),b|(1<<pair[0]),b|(1<<pair[0])|(1<<pair[1])]
   if all(k in pool for k in ks):bases[size].append(b)
 allbases=set(bases[3]+bases[4]);edges=[(b,t,b|(1<<t)) for b in bases[3] for t in range(15,20) if t not in pair and not (b&(1<<t)) and b|(1<<t) in allbases];rng.shuffle(edges);chosen=[];selected=set();usededges=[]
 for edge in edges:
  if len(usededges)>=32:break
  add=[b for b in [edge[0],edge[2]] if b not in selected]
  if len(selected)+len(add)>128:continue
  for b in add:chosen.append(b);selected.add(b)
  usededges.append(edge)
 for size in [3,4]:rng.shuffle(bases[size])
 for n in range(max(map(len,bases.values()),default=0)):
  for size in [3,4]:
   if n<len(bases[size]) and len(chosen)<128 and bases[size][n] not in selected:chosen.append(bases[size][n]);selected.add(bases[size][n])
 return {'pair':pair,'background_keys':chosen,'corner_keys':keys(corners(chosen,pair)).astype(int).tolist() if chosen else [],'nuisance_edges':usededges,'eligible_contexts':len(allbases),'eligible_edges':len(edges),'n_contexts':len(chosen),'density_counts':{str(n):sum(b.bit_count()==n for b in chosen) for n in [3,4]},'limited_support':len(chosen)<16}

def prepare(run):
 assert '**Version:** 0.5' in Path('SPEC.md').read_text();assert '**SPEC version 0.5**' in Path('CODEX_PROMPT.md').read_text();assert torch.cuda.is_available(),'CUDA required'
 run.mkdir(parents=True,exist_ok=False);log=run.parent/(run.name+'-logs');log.mkdir(exist_ok=True)
 files=[p for root in [BASE/'checkpoints',BASE/'data',BASE/'reports',BASE/'runs',BASE/'traces',BASE/'diagnostics',BASE/'execution_logs',BASE/'spec_archive',Path('results'),Path('logs')] for p in root.rglob('*') if p.is_file()]
 write(run/'protected_before.json',{str(p):sha(p) for p in files});snap=run/'source_snapshot';snap.mkdir()
 sources=[*Path(__file__).parent.glob('*.py'),BASE/'workflow_v04/core.py',BASE/'trace/core.py',BASE/'stage2/core.py',Path('src/models.py'),Path('src/runtime.py'),Path('SPEC.md'),Path('CODEX_PROMPT.md')]
 for f in sources:shutil.copy2(f,snap/str(f).replace('/','__'))
 prot0=read(OLD/'protocol.json');tp=Path(prot0['teacher']);choice=read(BASE/'reports/stage2-transcoders-20261027/selection.json')['best_validation_fidelity'];assert choice=='w1024_l0.0001';sp=BASE/'checkpoints/stage2-transcoders-20261027'/choice/'best.pt'
 for f in [tp,sp,DATA,MAN]:assert sha(f)==prot0['hashes'][str(f)]
 verify_record=read(BASE/'runs/stage2-transcoders-20261027/teacher_verification.json');assert sha(tp)==verify_record['checkpoint_sha256'];assert sha(DATA)==verify_record['dataset_sha256'];cp=load_checkpoint(tp);assert cp['identity']['config_sha256']==sha(BASE/'runs/capacity256-20261027/config.yaml');assert cp['epoch']==1987
 split0=read(MAN)['partitions'];alloriginal={int(k) for pp in split0.values() for k in pp['vector_keys']};train=np.asarray(split0['train']['vector_keys'],dtype=np.int64);assert len(train)==14000
 order=np.random.default_rng(6101).permutation(len(train));fit=train[order[:11200]].tolist();select=train[order[11200:]].tolist();assert set(fit).isdisjoint(select)
 protected_index=OLD/'protected_final_keys.npy';assert sha(protected_index)==read(OLD/'reservation_access.json')['index_sha256'];final=set(map(int,np.load(protected_index)));assert not (alloriginal&final)
 historic=set(map(int,np.load(OLD/'exclusion_keys.npz')['historical']));oldmanifest=read(OLD/'context_manifest.json')
 for r in oldmanifest.values():historic.update(int(k) for row in r['D']['corner_keys'] for k in row)
 for pair in PAIRS:
  r=oldmanifest[','.join(map(str,pair))];historic.update(int(k) for row in r['H']['corner_keys'] for k in row)
 # Account for later scored diagnostic records via persisted input keys only.
 extra_sources=[]
 for root in [BASE/'diagnostics']:
  for f in root.rglob('*manifest*.json'):
   extra_sources.append(str(f)) # Existing v1 reused v04 D/H, no new inputs.
 # Recovered keys from recorded stages are complete for this repo's two most recent analyses.
 excluded=alloriginal|final|historic;npz(run/'exclusion_keys.npz',original=np.array(sorted(alloriginal)),final=np.array(sorted(final)),historical=np.array(sorted(historic)))
 candidate=[sum(1<<i for i in active) for size in [3,4,5,6] for active in combinations(range(20),size)];assert len(candidate)==60249;allowed=[k for k in candidate if k not in excluded];rng=np.random.default_rng(6102);pools={n:[k for k in allowed if k.bit_count()==n] for n in [3,4,5,6]}
 for p0 in pools.values():rng.shuffle(p0)
 ordinary=[]
 for ix in range(max(map(len,pools.values()),default=0)):
  for n in [3,4,5,6]:
   if ix<len(pools[n]) and len(ordinary)<2000:ordinary.append(pools[n][ix])
 contexts={};legacy={};ordering=np.random.default_rng(6103);test=set(split0['test']['vector_keys']);forbidden=test|final
 for pair in PAIRS:
  name=','.join(map(str,pair));contexts[name]=choose_bank(allowed,pair,ordering)
  # Fixed legacy contexts from old manifest, never replace primary confirmation.
  rr=oldmanifest[name];bbs=list(dict.fromkeys(rr['D']['background_keys']+rr['H']['background_keys']));bbs=[b for b in bbs if all(int(k) not in forbidden for k in keys(corners([b],pair)).flat)]
  unseen=[b for b in bbs if all(int(k) not in set(fit)|set(select) for k in keys(corners([b],pair)).flat)];seen=[b for b in bbs if b not in set(unseen)];ordering.shuffle(unseen);ordering.shuffle(seen);bbs=(unseen+seen)[:128];pool=set(bbs);edges=[(b,t,b|(1<<t)) for b in bbs if b.bit_count()==3 for t in range(15,20) if t not in pair and not b&(1<<t) and b|(1<<t) in pool][:32]
  legacy[name]={'pair':pair,'background_keys':bbs,'corner_keys':keys(corners(bbs,pair)).astype(int).tolist() if bbs else [],'nuisance_edges':edges,'n_contexts':len(bbs),'source':str(OLD/'context_manifest.json'),'limited_support':len(bbs)<16}
 legacykeys=sorted({k for r in legacy.values() for row in r['corner_keys'] for k in row});confirmation_keys=sorted(set(ordinary)|{k for r in contexts.values() for row in r['corner_keys'] for k in row});assert all(k not in excluded for k in confirmation_keys);assert not set(confirmation_keys)&set(fit+select)
 known={int(k):name for name,part in split0.items() for k in part['vector_keys']};fset=set(fit);sset=set(select)
 membership={str(k):{'original':known.get(k,'unseen'),'readout':'fit' if k in fset else 'selection' if k in sset else 'unseen','historical_scored':k in historic} for k in set(legacykeys+confirmation_keys+fit+select)}
 jac=ordinary.copy();ordering.shuffle(jac);jac=jac[:256];worked={}
 for pair in PAIRS:
  name=','.join(map(str,pair));pop='primary' if contexts[name]['n_contexts'] else 'legacy';bk=(contexts if pop=='primary' else legacy)[name];edge=bk['nuisance_edges'][0] if bk['nuisance_edges'] else None;worked[name]={'population':pop,'background':edge[0] if edge else (bk['background_keys'][0] if bk['background_keys'] else None),'edge':edge,'rule':'metadata-only first declared edge else first declared complete background; never error-ranked'}
 manifest={'fit_keys':fit,'selection_keys':select,'ordinary_confirmation_keys':ordinary,'primary_unique_keys':confirmation_keys,'primary_contexts':contexts,'legacy_contexts':legacy,'legacy_unique_keys':legacykeys,'jacobian_keys':jac,'worked':worked,'membership':membership,'sampling':{'candidate_count':len(candidate),'allowed_count':len(allowed),'ordinary_density':{str(n):sum(k.bit_count()==n for k in ordinary) for n in [3,4,5,6]},'available_density':{str(n):len(pools[n]) for n in [3,4,5,6]},'condition':'uniform density-stratified metadata selection, excludes original/final/recoverable historical keys; non-IID','history_limitation':'Known stored diagnostic keys excluded. Unrecorded human/prior exposure cannot be ruled out. Diagnostic v1 used only previously scored v04 contexts.','reserved_access':'existing protected_final_keys.npy only; no reserved NPZ arrays opened','new_final_indexing':False}}
 write(run/'context_manifest.json',manifest)
 prot={'version':'0.5','created':now(),'teacher':str(tp),'student':str(sp),'hashes':{str(f):sha(f) for f in [tp,sp,DATA,MAN,Path('SPEC.md'),Path('CODEX_PROMPT.md'),protected_index]},'P_raw_sha256':prot0['P_raw_sha256'],'Q_raw_sha256':prot0['Q_raw_sha256'],'dimensions':{'H':256,'Z':1024,'x':12,'y':12,'semantic':9},'latent_numbering':'zero-based checkpoint encoder rows and decoder columns unchanged','grid':GRID,'rcond':1e-10,'seeds':{'partition':6101,'ordinary':6102,'contexts':6103,'shuffle':6104,'uncertainty':6105,'synthetic':6106},'fit_loss':'mean_examples sum_latents ||standardized_H U - centered_Z||^2 + alpha||U||^2; unpenalized intercept','primary_inputs':'post-ReLU H,Z only; no labels, Q, B, W2, output/response/Jacobian targets','readout_rule':'raw Mh+d; no clipping/ReLU; no fit+selection refit','eligibility':'v > max(1e-12,1e-8*median(strictly positive fit variances))','activity':'tau=max(1e-8,1e-3*sqrt(mean_fit_z_squared))','response_scale':'delta=0.1*sqrt(fitvar)','well_read':'NMSE<=.05 and active relative RMSE<=.20 with>=32 meaningful-active rows','finite_compatible':'relative RMSE<=.20 on >=32 nontrivial changes by type','jacobian_normalized_boundary_margin':1e-8,'jacobian_FD_initial_step':1e-4,'split_half':'fixed seeded partition order fit first5600 and last5600, own stats, same selected alpha','source_hashes':{str(f):sha(f) for f in sources},'archive':str(sorted((BASE/'spec_archive').glob('v05-adoption-*'))[-1]),'torch':torch.__version__,'CUDA':torch.version.cuda,'teacher_accepted_pilot':'failed original all-primitive >=99% balanced-accuracy gate','legacy_overlap':'tagged; never mixed with primary','no_network_training':True}
 write(run/'protocol.json',prot);(run/'git_status.txt').write_text(subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout);(run/'gpu.txt').write_text(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
 t,s,p,q=models(run);torch.testing.assert_close(p.norm(dim=0),torch.ones(20,device='cuda',dtype=DT),atol=2e-7,rtol=2e-7);torch.testing.assert_close(q.T@q,torch.eye(9,device='cuda',dtype=DT),atol=2e-7,rtol=2e-7)
 lock(run,'manifest_lock.json',['protocol.json','context_manifest.json','exclusion_keys.npz']);print('PREPARED',manifest['sampling'],{k:r['n_contexts'] for k,r in contexts.items()},flush=True)

@torch.no_grad()
def extract(run,ks):
 t,s,p,q=models(run);out={k:[] for k in ['x','h','z','y','scores']};errs=[]
 for start in range(0,len(ks),512):
  a=tensor(inputs(ks[start:start+512]));x=a@p.T;pre,h=t.hidden(x);z=s.latents(x);y=t.W2(h);scores=y@q
  C=t.W1.weight@p;K=s.encoder.weight@p;torch.testing.assert_close(h,(a@C.T+t.W1.bias).relu(),atol=1e-9,rtol=1e-8);torch.testing.assert_close(z,(a@K.T+s.encoder.bias).relu(),atol=1e-9,rtol=1e-8)
  for k,v in [('x',x),('h',h),('z',z),('y',y),('scores',scores)]:out[k].append(numpy(v))
  if start==0:
   t.float();s.float();x32=a.float()@p.float().T;y32=t(x32);z32=s.latents(x32);np.testing.assert_allclose(numpy(y32),numpy(y),atol=2e-6,rtol=2e-5);np.testing.assert_allclose(numpy(z32),numpy(z),atol=2e-6,rtol=2e-5);errs.append({'teacher_FP32max':float(np.max(np.abs(numpy(y32)-numpy(y)))),'latent_FP32max':float(np.max(np.abs(numpy(z32)-numpy(z))))});t.double();s.double()
 return {**{k:np.concatenate(v) for k,v in out.items()},'keys':np.array(ks,dtype=np.int64)},errs

def extract_fit_select(run):
 verify(run,'manifest_lock.json');man=read(run/'context_manifest.json');checks={}
 for label in ['fit','selection']:
  a,e=extract(run,man[label+'_keys']);npz(run/(label+'_activations.npz'),**a);checks[label]=e
 write(run/'fit_extraction_checks.json',checks)

@torch.no_grad()
def fit_controls_and_M(run):
 verify(run,'manifest_lock.json');f=dict(np.load(run/'fit_activations.npz'));s=dict(np.load(run/'selection_activations.npz'));z=tensor(f['z']);zs=tensor(s['z']);stats=target_statistics(z);npz(run/'fit_statistics.npz',**stats);write(run/'fit_statistics.json',{'v_floor':stats['v_floor'],'fit_eligible':int(stats['eligible'].sum()),'dormant':int((stats['strict_fit_activity']==0).sum()),'rare_meaningful_lt32':int((stats['meaningful_fit_activity']<32).sum())})
 results=[];modelsout={};diags={};rng=np.random.default_rng(6104);perm=rng.permutation(len(z));np.save(run/'shuffle_permutation.npy',perm)
 # Primary target arrays and predictor API remain independent of diagnostic outputs.
 for name,key in [('native','h'),('input','x'),('output','y'),('semantic','scores'),('shuffle','h')]:
  hf,hs=tensor(f[key]),tensor(s[key]);target=z[tensor(perm).long()] if name=='shuffle' else z;fit=design_fit(hf,target);gridrows=[];candidates={};projection=None
  for a in GRID:
   model=solve(fit,a);pf=predict(model,hf);ps=predict(model,hs);score=float(((ps-zs).square().mean(0)[stats['eligible']]/stats['v'][stats['eligible']]).mean());rr={'method':name,'alpha':a,'score':score,'fit_raw_MSE':float((pf-target).square().mean()),'selection_raw_MSE':float((ps-zs).square().mean()),'normal_equation_relative_residual':model['normal_equation_relative_residual']};gridrows.append(rr);candidates[a]=model
   if a==0:projection=(pf-target).square().mean(0)
  a=choose_alpha(gridrows);chosen=compact_model(candidates[a],fit);torch.save(to_cpu(chosen),run/(name+'_readout.pt'));modelsout[name]={'alpha':a,'rank':fit['rank'],'native_constant_columns':int(fit['constant_mask'].sum()),'selection_score':next(r['score'] for r in gridrows if r['alpha']==a)};results+=gridrows
  svals=numpy(fit['s']);diags[name]={'rank':fit['rank'],'retained_columns':int(fit['retained'].sum()),'condition_retained':float(svals[0]/svals[fit['rank']-1]) if fit['rank'] else None,'s_max':float(svals.max()) if len(svals) else None,'s_min':float(svals.min()) if len(svals) else None,'rcond':fit['rcond'],'cutoff':fit['cutoff'],'svd_reconstruction_max':fit['svd_reconstruction_max'],'selected_alpha':a};npz(run/(name+'_solver.npz'),singular_values=fit['s'],projection_residual_per_feature=projection,mu_h=fit['mu_h'],s_h=fit['s_h'],constant_mask=fit['constant_mask'])
  if name=='native':
   np.save(run/'M.npy',numpy(chosen['M']));np.save(run/'d.npy',numpy(chosen['d']));torch.save(to_cpu({**chosen,'target_statistics':to_cpu(stats),'hashes':read(run/'protocol.json')['hashes'],'input_manifest_sha256':sha(run/'context_manifest.json')}),run/'readout.pt');npz(run/'preprocessing.npz',mu_h=chosen['mu_h'],s_h=chosen['s_h'],mu_z=chosen['mu_z'],constant_mask=chosen['constant_mask'],eligible=stats['eligible'],v=stats['v'],tau=stats['tau'],U=chosen['U'])
   # Display membership fixed by fit-only centered RMS, not selected errors.
   rms=chosen['M'].abs()*chosen['s_h'][None,:];panels={str(m):sorted(range(256),key=lambda r:(-float(rms[m,r]),r))[:8] for m in FEATURES};write(run/'display_panels.json',panels)
   for half,indices in [('half1',np.arange(5600)),('half2',np.arange(5600,11200))]:
    dh=design_fit(hf[indices],z[indices]);halfmodel=compact_model(solve(dh,a),dh);torch.save(to_cpu(halfmodel),run/(half+'_readout.pt'))
  print('FITTED',name,modelsout[name],flush=True)
 modelsout['mean']={'alpha':None,'rank':0};torch.save({'M':torch.zeros((1024,0),dtype=DT),'d':numpy(stats['mu_z'])},run/'mean_readout.pt');csvwrite(run/'alpha_grid.csv',results);write(run/'solver_diagnostics.json',diags);write(run/'selected_models.json',modelsout)

def select_and_lock(run):
 # Candidates have been selected using selection latent error only; no confirmation cache exists.
 assert not (run/'confirmation_activations.npz').exists();assert not (run/'readout_lock.json').exists()
 files=['M.npy','d.npy','readout.pt','preprocessing.npz','fit_statistics.npz','fit_statistics.json','alpha_grid.csv','selected_models.json','solver_diagnostics.json','display_panels.json','shuffle_permutation.npy','manifest_lock.json']+[name+'_readout.pt' for name in ['native','input','output','semantic','shuffle','mean','half1','half2']]+[name+'_solver.npz' for name in ['native','input','output','semantic','shuffle']]
 lock(run,'readout_lock.json',files)
 (run/'HOW_TO_USE_M.md').write_text('# Raw-coordinate affine latent readout\n\n`M.npy` has shape (1024,256); `d.npy` has shape (1024,). Rows are zero-based frozen transcoder latents, columns are zero-based original hidden neurons.\n\n```python\nM = np.load("M.npy")\nd = np.load("d.npy")\n# H is RAW original post-ReLU hidden activation, shape (N,256).\nZhat = H @ M.T + d\n```\n\nDo not standardize H again, rectify/clip Zhat, permute latents, or use preactivations. Equivalent centered form: `mu_z + (H-mu_h) @ M.T`. Both networks remain frozen. The fitted map is neither ground-truth D2 nor a unique or causal hidden pathway.\n')
 print('READOUT LOCKED',read(run/'selected_models.json')['native'],flush=True)

def extract_confirmation(run):
 verify(run,'readout_lock.json');verify(run,'manifest_lock.json');man=read(run/'context_manifest.json');checks={}
 for label,ks in [('confirmation',man['primary_unique_keys']),('legacy',man['legacy_unique_keys'])]:
  if ks:
   a,e=extract(run,ks);npz(run/(label+'_activations.npz'),**a);checks[label]=e
 write(run/'confirmation_extraction_checks.json',checks)

def load_models(run):
 modelsout={}
 for name in ['native','input','output','semantic','shuffle','mean','half1','half2']:
  cp=torch.load(run/(name+'_readout.pt'),map_location='cuda',weights_only=False);modelsout[name]={k:(tensor(v) if isinstance(v,(np.ndarray,torch.Tensor)) else v) for k,v in cp.items()}
 return modelsout

def predictions(modelsout,arrays):
 out={}
 for name,model in modelsout.items():
  key={'native':'h','input':'x','output':'y','semantic':'scores','shuffle':'h','half1':'h','half2':'h'}.get(name)
  out[name]=numpy(predict(model,tensor(arrays[key]))) if key else np.broadcast_to(numpy(model['d']),(len(arrays['z']),1024)).copy()
 return out

def metrics_summary(rows):
 nm=np.array([r['NMSE_fitvar'] for r in rows if r['NMSE_fitvar'] is not None]);return {'well_read':sum(r['well_read'] for r in rows),'assessed_features':len(nm),'median_NMSE':float(np.median(nm)) if len(nm) else None,'p90_NMSE':float(np.quantile(nm,.9)) if len(nm) else None,'worst_decile_mean_NMSE':float(np.sort(nm)[int(.9*len(nm)):].mean()) if len(nm) else None,'mean_NMSE':float(nm.mean()) if len(nm) else None,'pooled_raw_MSE':float(np.mean([r['mse'] for r in rows])),'unsupported_meaningful_lt32':sum(r['meaningful_activity']<32 for r in rows),'newly_activating':sum(r['new_activity'] for r in rows),'constant_eval':sum(r['R2_eval'] is None for r in rows)}

def evaluate_latents(run):
 verify(run,'readout_lock.json');man=read(run/'context_manifest.json');modelsout=load_models(run);stats=dict(np.load(run/'fit_statistics.npz'));summaries={};allrows=[]
 for label in ['fit','selection','confirmation','legacy']:
  path=run/(label+'_activations.npz')
  if not path.exists():continue
  arrays=dict(np.load(path));preds=predictions(modelsout,arrays);npz(run/(label+'_predictions.npz'),**preds);lookup={int(k):i for i,k in enumerate(arrays['keys'])}
  populations=[(label,np.arange(len(arrays['z'])))]
  if label=='confirmation':
   populations=[('primary_ordinary',np.array([lookup[k] for k in man['ordinary_confirmation_keys']])),('primary_union',np.arange(len(arrays['z'])))]
   for key,bank in man['primary_contexts'].items():
    ids=sorted({lookup[k] for row in bank['corner_keys'] for k in row})
    if ids:populations.append(('primary_context_'+key,np.array(ids)))
  if label=='legacy':
   populations=[('legacy_union',np.arange(len(arrays['z'])))]
   for key,bank in man['legacy_contexts'].items():
    ids=sorted({lookup[k] for row in bank['corner_keys'] for k in row})
    if ids:populations.append(('legacy_context_'+key,np.array(ids)))
  for population,idx in populations:
   summaries[population]={}
   for name,pred in preds.items():
    rows=feature_metrics(arrays['z'][idx],pred[idx],stats)
    for r in rows:r.update({'population':population,'method':name})
    summaries[population][name]=metrics_summary(rows);allrows+=rows
   print('EVALUATED',population,summaries[population]['native'],flush=True)
 csvwrite(run/'all_feature_metrics.csv',allrows);write(run/'latent_metric_summary.json',summaries)
 # Coefficient direction/sign stability versus both fixed-half diagnostic fits.
 base=modelsout['native'];rr=[]
 for half in ['half1','half2']:
  other=modelsout[half];u=base['M']*base['s_h'];v=other['M']*other['s_h'];den=u.norm(dim=1)*v.norm(dim=1);cos=(u*v).sum(1)/den.clamp_min(1e-30);relevant=(u.abs()>1e-12)|(v.abs()>1e-12);signagree=((u.sign()==v.sign())&relevant).sum(1)/relevant.sum(1).clamp_min(1)
  for m in range(1024):rr.append({'feature':m,'comparison':half,'standardized_direction_cosine':float(cos[m]) if den[m]>1e-20 else None,'primary_standardized_norm':float(u[m].norm()),'half_standardized_norm':float(v[m].norm()),'nonzero_contribution_sign_agreement':float(signagree[m]),'raw_coefficient_difference_norm':float((base['M'][m]-other['M'][m]).norm())})
 csvwrite(run/'coefficient_stability.csv',rr)
 if (run/'confirmation_predictions.npz').exists():
  pp=dict(np.load(run/'confirmation_predictions.npz'));zz=np.load(run/'confirmation_activations.npz')['z'];rows=[]
  for m in range(1024):
   rows.append({'feature':m,**{name+'_prediction_difference_RMSE':float(np.mean((pp[name][:,m]-pp['native'][:,m])**2)**.5) for name in ['half1','half2']},'half1_vs_half2_RMSE':float(np.mean((pp['half1'][:,m]-pp['half2'][:,m])**2)**.5)})
  csvwrite(run/'prediction_stability.csv',rows)
def test(run):
 verify(run,'manifest_lock.json');cmd=[sys.executable,'-m','pytest','-q',str(Path(__file__).parent/'test_v05.py')];result=subprocess.run(cmd,capture_output=True,text=True);(run/'tests.log').write_text(result.stdout+result.stderr);(run/'tests.exit_status').write_text(str(result.returncode)+'\n');assert result.returncode==0,'Correctness tests failed; see tests.log'
 from .test_v05 import fixture
 h,m,d,z=fixture();r=solve(design_fit(h,z),0);x=tensor([[-2.],[-1.],[1.],[2.]]);hh=x.relu();zz=torch.cat([x.relu(),(-x).relu(),(-x).relu()],1);bb=tensor([[1.,1.,-1.]]);negative=solve(design_fit(hh,zz),0);zn=predict(negative,hh)
 write(run/'synthetic_control_results.json',{'known_affine_max_coefficient_error':float((r['M']-m).abs().max()),'known_affine_max_intercept_error':float((r['d']-d).abs().max()),'known_affine_prediction_max':float((predict(r,h)-z).abs().max()),'perfect_output_negative_fixture_output_max':float((zz@bb.T-hh).abs().max()),'perfect_output_negative_fixture_latent_MSE':float((zn-zz).square().mean()),'negative_fixture_true_latents':numpy(zz),'negative_fixture_readout':numpy(zn),'native_copy_permutation_rescaling':'passed coefficient/prediction and output/Jacobian identities in tests.log','no_models_trained':True});print('TESTS PASSED',flush=True)


def preservation_audit(run):
 verify(run,'manifest_lock.json');verify(run,'readout_lock.json');man=read(run/'context_manifest.json');ex=dict(np.load(run/'exclusion_keys.npz'));exclude=set(map(int,np.concatenate(list(ex.values()))));assert set(man['fit_keys']).isdisjoint(man['selection_keys']);assert not set(man['primary_unique_keys'])&exclude
 # Actual complete-corner protection and dependency checks, no population scored here.
 for bank in man['primary_contexts'].values():
  if bank['background_keys']:
   ks=keys(corners(bank['background_keys'],bank['pair']));np.testing.assert_array_equal(ks,bank['corner_keys']);assert not set(map(int,ks.flat))&exclude
 cp=torch.load(run/'readout.pt',map_location='cuda',weights_only=False);M=tensor(np.load(run/'M.npy'));d=tensor(np.load(run/'d.npy'));torch.testing.assert_close(M,cp['M']);torch.testing.assert_close(d,cp['d']);fit=np.load(run/'fit_activations.npz');hh=tensor(fit['h'][:64]);raw=hh@M.T+d;center=cp['mu_z']+(hh-cp['mu_h'])@M.T;torch.testing.assert_close(raw,center,atol=1e-9,rtol=1e-8)
 a={};
 if (run/'confirmation_activations.npz').exists():
  arr=dict(np.load(run/'confirmation_activations.npz'))
  for name,key in [('native','h'),('input','x'),('output','y'),('semantic','scores')]:
   model=load_models(run)[name];const=numpy(model['constant_mask']).astype(bool);a[name]={'fit_constant_count':int(const.sum()),'varying_on_confirmation':np.flatnonzero(const&(arr[key].std(0)>1e-12*max(1.,float(numpy(model['s_h']).max())))).tolist()}
 write(run/'constant_predictor_confirmation.json',a)
 previous=read(run/'protected_before.json');assert all(sha(path)==h for path,h in previous.items()),'Previous artifact changed'
 source_record=read(run/'protocol.json')['source_hashes'];assert all(sha(path)==h for path,h in source_record.items()),'Source changed during execution'
 write(run/'preservation_verified.json',{'timestamp':now(),'files_verified':len(previous),'split_disjoint':True,'primary_complete_corner_exclusions_verified':True,'readout_cuda_reload_and_centering_max':float((raw-center).abs().max()),'protected_final_scored':False,'reserved_NPZ_opened':False,'no_network_training':True,'no_clipping':True,'source_hashes_unchanged':True})
 write(run/'complete.json',{'status':'complete','timestamp':now(),'M':str((run/'M.npy').resolve()),'d':str((run/'d.npy').resolve()),'readout_sha256':sha(run/'readout.pt'),'M_sha256':sha(run/'M.npy'),'d_sha256':sha(run/'d.npy'),'summary_sha256':sha(run/'SUMMARY.md'),'feature_reports':{str(m):sha(run/f'feature_{m}.md') for m in FEATURES},'protected_files_verified':len(previous),'tests_exit_status':0,'networks_modified':False,'final_scored':False});print('COMPLETE',run/'SUMMARY.md',flush=True)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['all','prepare','test','extract_fit_select','fit_controls_and_M','select_and_lock','extract_confirmation','evaluate_latents','evaluate_outputs','evaluate_responses','evaluate_jacobians','explain_fixed_features','report','preservation_audit']);ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stages=['prepare','test','extract_fit_select','fit_controls_and_M','select_and_lock','extract_confirmation','evaluate_latents','evaluate_outputs','evaluate_responses','evaluate_jacobians','explain_fixed_features','report','preservation_audit'] if args.stage=='all' else [args.stage]
 timings=[]
 for name in stages:
  start=time.monotonic();print(now(),name,flush=True)
  if name in ['evaluate_outputs','evaluate_responses','evaluate_jacobians']:
   from . import evaluate as evaluation
   getattr(evaluation,name)(args.run)
  elif name in ['explain_fixed_features','report']:
   from . import report as reporting
   getattr(reporting,name)(args.run)
  else:globals()[name](args.run)
  timings.append({'stage':name,'elapsed_seconds':time.monotonic()-start,'status':0,'completed':now()});write(args.run/'stage_timings.json',timings)
if __name__=='__main__':main()
