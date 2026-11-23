import os,sys,time,shutil,subprocess,copy
from pathlib import Path
import yaml,numpy as np,torch
from .core import *
from ..native_features_v07.run import calibration
from ..latent_correspondence_v05.core import verify
from ..latent_correspondence_v05.run import models,DATA,MAN
STAGE=BASE/'runs/stage2-transcoders-20261027'

def prepare(root):
 assert 'SPEC v0.8' in Path('SPEC.md').read_text();assert 'SPEC v0.8' in Path('CODEX_PROMPT.md').read_text();root.mkdir(parents=True,exist_ok=False)
 for n in ['phase_I','phase_II','source_snapshot']:(root/n).mkdir()
 for n in ['paths','static_references','worked_cases','figures']:(root/'phase_I'/n).mkdir()
 for n in ['training_runs','readout_runs','validation_and_factorization','figures']:(root/'phase_II'/n).mkdir()
 conf=yaml.safe_load((STAGE/'config.yaml').read_text());splits=read(MAN)['partitions'];man=read(V07/'input_manifests.json');tc_keys=set(splits['train']['vector_keys']+splits['validation']['vector_keys']);data={}
 for name in ['V','A']:
  ks=[k for k in man['banks'][name]['keys'] if k not in tc_keys];edges,contexts=rebuild(ks);assert len(ks)>=512;data[name]=dict(keys=ks,edges=edges,contexts=contexts,rows=len(ks),n_edges=len(edges),density={str(n):sum(k.bit_count()==n for k in ks) for n in [3,4,5,6]});assert not set(ks)&tc_keys
 write(root/'phase_II/data_manifests.json',data)
 grid=dict(version='0.8',widths=[128,256,512,1024,2048],student_seeds=[4105,5105,6105],shuffle_seeds=[4106,5106,6106],lambda_L1=1e-4,original_config=conf,original_config_sha256=sha(STAGE/'config.yaml'),source_training_sha256=sha(BASE/'stage2/run.py'),native_model_frozen=True,readout_alpha_grid=[0,1e-8,1e-6,1e-4,1e-2,1],budgets=[16,32,64,128],data_manifest_sha256=sha(root/'phase_II/data_manifests.json'))
 write(root/'phase_II/training_grid_lock.json',grid);write(root/'protocol_lock.json',dict(version='0.8',phase_II_grid_sha256=sha(root/'phase_II/training_grid_lock.json'),phase_I_guards=dict(innovation=.25,condition=1e6,cap='1.10*rho_L(S0,V)+.01'),reference_script='requested bundled REFERENCE_CHECKS.py not present; equivalent repository fixtures implemented',active_spec_location='root SPEC.md and CODEX_PROMPT.md; requested subdirectory absent',seeds=list(range(8200,8210)),max_additions=128,phase_order=['phase_I_report','phase_II_training','all_width_locks','A_width_audit']))
 previous={str(p):sha(p) for folder in [BASE,Path('results'),Path('logs')] for p in folder.rglob('*') if p.is_file() and root not in p.parents and '__pycache__' not in str(p)};write(root/'preservation_before.json',previous)
 for p in [Path('SPEC.md'),Path('CODEX_PROMPT.md'),*Path(__file__).parent.glob('*.py'),STAGE/'config.yaml',BASE/'stage2/run.py',BASE/'stage2/core.py',Path('src/models.py'),Path('src/runtime.py')]:shutil.copy2(p,root/'source_snapshot'/str(p).replace('/','__'))
 verify(OLD,'readout_lock.json');verify(OLD,'manifest_lock.json')
 for f,h in read(V07/'feature_lock.json').items():assert sha(V07/f)==h
 for f,h in read(V07/'protocol_lock.json')['source_hashes'].items():
  if f not in ['SPEC.md','CODEX_PROMPT.md']:assert sha(f)==h
 # Historical training implementation and model semantics must match archived code byte-for-byte.
 for p in [BASE/'stage2/run.py',BASE/'stage2/core.py',Path('src/models.py'),Path('src/runtime.py')]:assert sha(p)==sha(STAGE/'source_snapshot'/str(p).replace('/','__'))
 reuse={}
 for w in [512,1024]:
  p=BASE/f'checkpoints/stage2-transcoders-20261027/w{w}_l0.0001/best.pt';cp=torch.load(p,map_location='cpu',weights_only=False);assert cp['config']==conf and cp['width']==w and cp['lambda']==1e-4;assert cp['identity']['teacher_sha256']=='47a61c86e48e58f09bd1d2da167a5a3a904e8fdb5a39c171bf542b6994d49c70';assert cp['identity']['config_sha256']==sha(STAGE/'config.yaml');assert cp['identity']['target_cache_sha256']==sha(STAGE/'teacher_targets.npz');reuse[f'w{w}_s4105']=dict(path=str(p),sha256=sha(p),training_complete=read(p.parent/'training_complete.json'),verified='config, seeds, data/targets, architecture source, initialization/optimizer/checkpoint policy')
 write(root/'phase_II/reuse_verified.json',reuse);write(root/'source_manifest.json',{p:previous[p] for p in previous if p.endswith(('.py','.yaml','.pt')) or 'v07-' in p})
 (root/'tests.log').write_text(Path('/tmp/v08-tests.log').read_text());assert '7 passed' in (root/'tests.log').read_text();(root/'gpu.txt').write_text(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout);calibration(root/'phase_I');write(root/'prepared.json',dict(status='complete',CUDA=torch.cuda.is_available(),device=torch.cuda.get_device_name()));print('PREPARED; PHASE II LOCKED',flush=True)

@torch.no_grad()
def phase1(root):
 torch.set_num_threads(4);assert read(root/'prepared.json')['CUDA'];p1=root/'phase_I';prot=read(OLD/'protocol.json');t,s,p,q=models(OLD);W1=t.W1.weight@p;b1=t.W1.bias;K=s.encoder.weight@p;e=s.encoder.bias;W=q.T@t.W2.weight;b=q.T@t.W2.bias;M=tensor(np.load(OLD/'M.npy'));d=tensor(np.load(OLD/'d.npy'));S0=read(V07/'accepted_feature_ids.json');man=read(V07/'input_manifests.json')['banks'];oldrows=__import__('csv').DictReader(open(V07/'candidate_metrics_validation.csv'));oldrows=list(oldrows);statuses=[r['status'] for r in oldrows]
 native=p1/'native_matrices.npz';np.savez(native,D1=np.eye(20),W1=arr(W1),C1=arr(W1),b1=arr(b1),W2=arr(W),b2=arr(b),K=arr(K),e=arr(e),M=arr(M),d=arr(d))
 hf,zf=infer_canonical(read(OLD/'context_manifest.json')['fit_keys'],W1,b1,K,e);v=zf.var(0,unbiased=False);r=zf.square().mean(0).sqrt();hv,zv=infer_canonical(man['V']['keys'],W1,b1,K,e);edges=np.load(V07/'V_indices.npz')['edges'];rows,f=metrics(hv,zv,M,d,(v,r),edges);assert [r['feature'] for r in rows if r['status']=='ACCEPT']==S0
 c=cache(hv,zv,M,d,W,b,edges);c['native_W_norm']=W.norm();F0=factor(M,d,W,b,S0);base=score(F0,c,True);assert F0['rank']==184
 L=[r['feature'] for r in rows if r['status']=='REJECT_THIS_READOUT' and r['failed']==['inactive_leakage'] and not r['missing'] and r['inactive_leakage']<=.10];cap=1.1*base['rho_L']+.01;Bmax=min(128,len(L),256-F0['rank']);csvwrite(p1/'pools_and_statuses.csv',[dict(**r,reserve=r['feature'] in L,immutable_status=statuses[r['feature']]) for r in rows]);write(p1/'baseline_reproduction.json',dict(accepted_ids_match=True,rank=F0['rank'],V=base,reserve=L,Bmax=Bmax,cap_L=cap))
 gains=[]
 for m in range(1024):
  if m not in S0:
   qq,nu,gain=geometry(F0,M[m]);gains.append(dict(feature=m,in_reserve=m in L,strict_status=statuses[m],novelty=float(nu),gain_W=float(gain.sum()),per_output=arr(gain)))
 csvwrite(p1/'all_candidate_initial_gains.csv',gains)
 print('BASELINE reproduced; reserve',len(L),'budget',Bmax,'cap',cap,flush=True)
 def trial(F,current,m):
  inn=innovation(F,m,M,d,c);new=factor(M,d,W,b,F['ids']+[m]);sc=score(new,c);nest=float((new['J']@new['D']@F['J']@F['D']-F['J']@F['D']).norm());fails=[]
  if inn['nu']<=max(1e-12,1e-10*float(M[m].norm())):fails.append('novelty')
  if new['rank']!=F['rank']+1:fails.append('rank')
  if nest>1e-7:fails.append('nesting')
  if new['condition']>1e6:fails.append('condition')
  if not inn['assessable']:fails.append('innovation_support')
  elif inn['value']>.25 or inn['edge']>.25:fails.append('innovation_error')
  if sc['rho_L']>cap:fails.append('amplification_cap')
  rec=dict(feature=m,strict_status=statuses[m],provisional_label='PROVISIONALLY_INCLUDED_IF_SELECTED',innovation=inn,failures=fails,nesting=nest,actual_matrix_gain=float(F['R'].square().sum()-new['R'].square().sum()),before=current,after=sc,per_output_delta_N=np.array(current['per_output_N'])**2-np.array(sc['per_output_N'])**2)
  return new,sc,rec
 paths={}
 reliability=sorted(L,key=lambda m:(max(rows[m][k]/lim for k,lim in [('NMSE',.05),('active_RRMSE',.20),('inactive_leakage',.10),('finite_change_RRMSE',.20)]),m))
 for name in ['guided','reliability']+[f'random_{n}' for n in range(8200,8210)]:
  started=time.monotonic();folder=p1/'paths'/name;folder.mkdir();(folder/'checkpoints').mkdir();F=F0;current=base;seq=[S0.copy()];scores=[dict(step=0,**base)];ledger=[];add=[];priority=reliability if name=='reliability' else np.random.Generator(np.random.PCG64(int(name.split('_')[1]))).permutation(L).tolist() if name.startswith('random') else []
  for step in range(1,Bmax+1):
   order=sorted([m for m in L if m not in F['ids']],key=lambda m:(-float(geometry(F,M[m])[2].sum()),m)) if name=='guided' else [m for m in priority if m not in F['ids']];best=None
   for m in order:
    new,sc,rec=trial(F,current,m);rec.update(step=step);ledger.append(rec)
    ok=not rec['failures']
    if name=='guided':
     if sc['rho_N']**2>current['rho_N']**2+1e-12:rec['failures'].append('native_regression');ok=False
     if current['objective']-sc['objective']<max(1e-8,1e-4*current['objective']):rec['failures'].append('no_objective_improvement');ok=False
    if not ok:continue
    if best is None or sc['objective']<best[1]['objective']-1e-12 or (abs(sc['objective']-best[1]['objective'])<=1e-12 and (-rec['innovation']['gain_W'],m)<(-best[2]['innovation']['gain_W'],best[2]['feature'])):best=(new,sc,rec)
    if name!='guided':break
   if best is None:break
   new,sc,rec=best;score(new,c,True);rec['selected']=True;rec['coefficient_changes_shared']=arr(new['C'][:,:-1]-F['C']);add.append(rec);F,current=new,sc;seq.append(F['ids'].copy());scores.append(dict(step=step,added_feature=rec['feature'],**sc));print('ADD',name,step,rec['feature'],sc['objective'],flush=True)
  minimum=min(sc['objective'] for sc in scores);represent=next(i for i,sc in enumerate(scores) if sc['objective']<=minimum*1.01+ (1e-12 if minimum==0 else 0)) if name=='guided' else len(seq)-1
  selected={0,1,2,4,8,16,32,64,len(seq)-1,represent};selected={i for i in selected if i<len(seq)}
  for step in sorted(selected):savefactor(folder/'checkpoints'/f'add_{step:03}',factor(M,d,W,b,seq[step]),native,statuses)
  csvwrite(folder/'trial_ledger.csv',ledger);csvwrite(folder/'additions.csv',add);csvwrite(folder/'search_metrics.csv',scores);write(folder/'sequence_lock.json',dict(sequence=seq,checkpoint_steps=sorted(selected),representative=represent,endpoint=len(seq)-1,stop='budget' if len(seq)-1==Bmax else 'no_admissible_single_addition',trials=len(ledger),seconds=time.monotonic()-started,strict_labels_sha256=sha(V07/'candidate_metrics_validation.csv')));paths[name]=dict(seq=seq,scores=scores,representative=represent,additions=add)
  print('PATH LOCKED',name,len(seq)-1,'trials',len(ledger),flush=True)
 static={'strict':S0,'all_reserve':S0+L,'all_candidates':list(range(1024))}
 if 266 in L:static['singleton_TC266']=S0+[266];_,_,rec=trial(F0,base,266);write(p1/'static_references/TC266_guard_diagnostic.json',rec)
 for name,ids in static.items():
  F=factor(M,d,W,b,ids);folder=p1/'static_references'/name;folder.mkdir();savefactor(folder/'matrices',F,native,statuses);write(folder/'V_metrics.json',score(F,c,True))
 write(p1/'all_paths_lock.json',dict(paths={name:sha(p1/'paths'/name/'sequence_lock.json') for name in paths},static=static,phase_II_protocol_sha256=sha(root/'phase_II/training_grid_lock.json')))
 # Audit begins only after all search paths and representative states are locked.
 ha,za=infer_canonical(man['A']['keys'],W1,b1,K,e);ea=np.load(V07/'A_indices.npz')['edges'];ca=cache(ha,za,M,d,W,b,ea);ca['native_W_norm']=W.norm()
 for name,pth in paths.items():csvwrite(p1/'paths'/name/'reused_audit_metrics.csv',[dict(step=i,**score(factor(M,d,W,b,ids),ca,True)) for i,ids in enumerate(pth['seq'])])
 for name,ids in static.items():write(p1/'static_references'/name/'A_metrics.json',score(factor(M,d,W,b,ids),ca,True))
 reproduced=score(F0,ca,True);old=read(V07/'numerical_checks.json');assert abs(reproduced['rho_N']-old['aggregate_feature_RRMSE'])<1e-8;assert abs(reproduced['rho_T']-old['aggregate_TC_RRMSE'])<1e-8;write(p1/'audit_baseline_reproduction.json',reproduced)
 # Fixed complete V corners for first three guided additions plus TC266 diagnostic.
 cases=[r['feature'] for r in paths['guided']['additions'][:3]]+([266] if 266 in L else [])
 for m in cases:
  pair=(7,8) if m==266 else (0,1);ctx=man['V']['contexts'][f'{pair[0]},{pair[1]}']['corners'];ks=sorted(ctx,key=lambda c:tuple(bits([c[0]])[0]))[0];h,z=infer_canonical(ks,W1,b1,K,e);con=h*M[m];strength=arr((hf*M[m]).abs().mean(0));panel=sorted(range(256),key=lambda r:(-strength[r],r))[:8];rem=con.sum(1)-con[:,panel].sum(1);afterids=(S0+[m]) if m==266 else paths['guided']['seq'][1+[r['feature'] for r in paths['guided']['additions']].index(m)];beforeids=afterids[:-1]
  before=factor(M,d,W,b,beforeids);after=factor(M,d,W,b,afterids);np.savez(p1/'worked_cases'/f'TC_{m}.npz',keys=ks,panel=panel,W1=arr(W1),b1=arr(b1),M_row=arr(M[m]),d=arr(d[m]),pre=arr(tensor(bits(ks))@W1.T+b1),h=arr(h),TC=arr(z[:,m]),native=arr(h@M[m]+d[m]),contributions=arr(con),remainder=arr(rem),before_C=arr(before['C']),after_C=arr(after['C']),before_ids=beforeids,after_ids=afterids,before_output=arr((h@before['D'].T+before['d'])@before['C'].T+before['bF']),after_output=arr((h@after['D'].T+after['d'])@after['C'].T+after['bF']),before_residual=arr(h@before['R'].T),after_residual=arr(h@after['R'].T),full_native=arr(h@W.T+b))
 from .reports import phase1_report
 phase1_report(root,paths,static,L,base,rows)
 write(p1/'complete.json',dict(status='complete',report_sha256=sha(p1/'PHASE_I_SUMMARY.md'),paths_locked=True));print('PHASE I COMPLETE',flush=True)

if __name__=='__main__':
 root=Path(sys.argv[2]);{'prepare':prepare,'phase1':phase1}[sys.argv[1]](root)
