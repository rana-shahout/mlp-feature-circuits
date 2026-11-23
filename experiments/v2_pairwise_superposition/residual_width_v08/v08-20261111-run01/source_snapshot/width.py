"""Bounded width cells using unchanged archived Stage 2 training procedure."""
import os,sys,copy,time,shutil
from pathlib import Path
import numpy as np,torch
from .core import *
from .run import STAGE
from ..latent_correspondence_v05.run import models,MAN
from ..latent_correspondence_v05.core import design_fit,solve,choose_alpha,target_statistics,GRID
from src.models import SparseTranscoder
from src.runtime import load_checkpoint

def cell(root,width,seed,shuffle):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert (root/'phase_I/complete.json').exists();grid=read(root/'phase_II/training_grid_lock.json');assert width in grid['widths'] and seed in grid['student_seeds'];name=f'w{width}_s{seed}';train=root/'phase_II/training_runs'/name;train.mkdir(exist_ok=False);reuse=read(root/'phase_II/reuse_verified.json');t,old,p,q=models(OLD);conf=copy.deepcopy(grid['original_config']);conf['student_seed']=seed;conf['shuffle_seed']=shuffle;conf['widths']=[width];conf['lambdas']=[1e-4]
 if name in reuse:
  ck=Path(reuse[name]['path']);assert sha(ck)==reuse[name]['sha256'];write(train/'reuse.json',reuse[name]);info=reuse[name]['training_complete'];print('REUSE',name,flush=True)
 else:
  from ..stage2 import run as stage
  # Exact training implementation; configuration only changes authorized width and seeds.
  stage.C=conf;splits=read(MAN)['partitions'];xt=tensor(bits(splits['train']['vector_keys'])).float()@p.float().T;xv=tensor(bits(splits['validation']['vector_keys'])).float()@p.float().T
  with np.load(STAGE/'teacher_targets.npz') as data:yt=torch.as_tensor(data['train'],device='cuda');yv=torch.as_tensor(data['validation'],device='cuda')
  identity=dict(teacher_sha256=read(OLD/'protocol.json')['hashes'][read(OLD/'protocol.json')['teacher']],dataset_sha256=sha(Path(conf['dataset'])),target_cache_sha256=sha(STAGE/'teacher_targets.npz'),width=width,lambda_L1=1e-4,grid_sha256=sha(root/'phase_II/training_grid_lock.json'))
  stage.fit(width,1e-4,xt,yt,xv,yv,train,identity);ck=train/'best.pt';info=read(train/'training_complete.json');del xt,xv,yt,yv
 cp=load_checkpoint(ck);s=SparseTranscoder(12,width,12).cuda().double();s.load_state_dict(cp['model']);s.eval().requires_grad_(False);W1=t.W1.weight@p;b1=t.W1.bias;W=q.T@t.W2.weight;b=q.T@t.W2.bias;K=s.encoder.weight@p;e=s.encoder.bias
 native=root/'phase_I/native_matrices.npz';ro=root/'phase_II/readout_runs'/name;ro.mkdir();man=read(OLD/'context_manifest.json');h,z=infer_canonical(man['fit_keys'],W1,b1,K,e);hs,zs=infer_canonical(man['selection_keys'],W1,b1,K,e);stats=(z.var(0,unbiased=False),z.square().mean(0).sqrt());np.savez(ro/'fit_statistics.npz',v=arr(stats[0]),r=arr(stats[1]))
 if width==1024 and seed==4105:
  M=tensor(np.load(OLD/'M.npy'));d=tensor(np.load(OLD/'d.npy'));alpha=read(OLD/'selected_models.json')['native']['alpha'];write(ro/'reuse.json',dict(source=str(OLD),M_sha256=sha(OLD/'M.npy'),d_sha256=sha(OLD/'d.npy')))
 else:
  fit=design_fit(h,z);positive=stats[0][stats[0]>0];floor=max(1e-12,1e-8*float(torch.quantile(positive,.5)));eligible=stats[0]>floor;mods={};scores=[]
  for a in GRID:
   mod=solve(fit,a);err=(hs@mod['M'].T+mod['d']-zs).square().mean(0);sc=float((err[eligible]/stats[0][eligible]).mean());scores.append(dict(alpha=a,score=sc));mods[a]=mod
  alpha=choose_alpha(scores);M,d=mods[alpha]['M'],mods[alpha]['d'];csvwrite(ro/'alpha_grid.csv',scores);np.savez(ro/'preprocessing.npz',mu_H=arr(fit['mu_h']),sd_H=arr(fit['s_h']),constant=arr(fit['constant_mask']),mu_Z=arr(fit['mu_z']));write(ro/'fit_diagnostics.json',dict(alpha=alpha,rank=fit['rank'],normal_equation_relative_residual=mods[alpha]['normal_equation_relative_residual'],constant_rule='sd <=1e-12*max(1,max(sd))'))
 np.save(ro/'M.npy',arr(M));np.save(ro/'d.npy',arr(d));np.savez(ro/'TC_canonical.npz',K=arr(K),e=arr(e));write(ro/'readout_lock.json',dict(M=sha(ro/'M.npy'),d=sha(ro/'d.npy'),TC_checkpoint=str(ck),TC_checkpoint_sha256=sha(ck),alpha=alpha,fit_manifest_sha256=sha(OLD/'context_manifest.json')))
 banks=read(root/'phase_II/data_manifests.json');hv,zv=infer_canonical(banks['V']['keys'],W1,b1,K,e);rows,f=metrics(hv,zv,M,d,stats,np.asarray(banks['V']['edges']));out=root/'phase_II/validation_and_factorization'/name;out.mkdir();csvwrite(out/'candidate_metrics_validation.csv',rows);ids=[r['feature'] for r in rows if r['status']=='ACCEPT'];statuses=[r['status'] for r in rows];F=factor(M,d,W,b,ids);savefactor(out/'full_set',F,native,statuses);c=cache(hv,zv,M,d,W,b,np.asarray(banks['V']['edges']));c['native_W_norm']=W.norm();write(out/'full_V_metrics.json',score(F,c,True));write(out/'accepted_ids.json',ids)
 seq=[[]];current=factor(M,d,W,b,[]);gains=[]
 for step in range(min(128,F['rank'])):
  candidates=[]
  for m in ids:
   if m not in current['ids']:candidates.append((float(geometry(current,M[m])[2].sum()),m))
  chosen=None
  for gain,m in sorted(candidates,key=lambda x:(-x[0],x[1])):
   if gain<=1e-12*float(W.square().sum()):break
   trial=factor(M,d,W,b,current['ids']+[m])
   if trial['rank']==current['rank']+1:chosen=trial;gains.append(gain);break
  if chosen is None:break
  current=chosen;seq.append(current['ids'].copy())
 keep=sorted({0,16,32,64,128,len(seq)-1}&set(range(len(seq))));bud=[]
 for step in keep:
  f=factor(M,d,W,b,seq[step]);savefactor(out/f'budget_{step:03}',f,native,statuses);bud.append(dict(budget=step,**score(f,c,True)))
 csvwrite(out/'budget_V_metrics.csv',bud)
 # Fidelity uses TC decoder only as an ordinary training/fidelity measurement, never as a C2 target.
 def fidelity(ks,z):
  hh,_=infer_canonical(ks,W1,b1,K,e);a=tensor(bits(ks));x=a@p.T;full=t.W2(hh);pred=s.decoder(z);sem=pred@q;target=hh@W.T+b;return dict(full_output_MSE=float((pred-full).square().mean()),semantic_MSE=float((sem-target).square().mean()),exact_positive_L0=float((z>0).sum(1).double().mean()),meaningful_L0=float((z>torch.maximum(torch.full_like(stats[1],1e-8),1e-3*stats[1])).sum(1).double().mean()),fraction_active=float((z>0).double().mean()),dead=int((~(z>0).any(0)).sum()))
 inf=dict(width=width,seed=seed,shuffle_seed=shuffle,accepted=len(ids),rejected=sum(r['status']=='REJECT_THIS_READOUT' for r in rows),unassessed=sum(r['status']=='UNASSESSED' for r in rows),accepted_fraction=len(ids)/width,assessed_fraction=sum(r['status']!='UNASSESSED' for r in rows)/width,accepted_rank=F['rank'],smallest_retained=float(F['s'][F['rank']-1]) if F['rank'] else None,condition=F['condition'],J_norm=F['J_norm'],training=info,reused=name in reuse,selected_epoch=info['selected_epoch'],epochs=info['epochs'],updates=info['epochs']*((len(read(MAN)['partitions']['train']['vector_keys'])+255)//256),cap_hit=info['epochs']==2000,alpha=alpha,validation_fidelity=fidelity(banks['V']['keys'],zv))
 write(out/'cell_info.json',inf);write(out/'sequence_lock.json',dict(sequence=seq,evaluation_budgets=keep,geometric_gains=gains,checkpoint_sha256=sha(ck),accepted_ids_sha256=sha(out/'accepted_ids.json'),readout_lock_sha256=sha(ro/'readout_lock.json'),grid_sha256=sha(root/'phase_II/training_grid_lock.json')));print('CELL V LOCKED',name,len(ids),F['rank'],flush=True)

def worker(root,worker):
 grid=read(root/'phase_II/training_grid_lock.json');cells=[(w,s,sh) for w in grid['widths'] for s,sh in zip(grid['student_seeds'],grid['shuffle_seeds'])]
 for i,(w,s,sh) in enumerate(cells):
  if i%2==worker:cell(root,w,s,sh)
 write(root/f'phase_II/worker_{worker}_complete.json',dict(status='complete'))

@torch.no_grad()
def audit(root):
 torch.set_num_threads(4);grid=read(root/'phase_II/training_grid_lock.json');folders=[root/f'phase_II/validation_and_factorization/w{w}_s{s}' for w in grid['widths'] for s in grid['student_seeds']];assert all((f/'sequence_lock.json').exists() for f in folders);write(root/'phase_II/all_primary_locks.json',{f.name:sha(f/'sequence_lock.json') for f in folders})
 t,old,p,q=models(OLD);W1=t.W1.weight@p;b1=t.W1.bias;W=q.T@t.W2.weight;b=q.T@t.W2.bias;bks=read(root/'phase_II/data_manifests.json');ks=bks['A']['keys'];edges=np.asarray(bks['A']['edges']);summary=[];budgets=[]
 for folder in folders:
  inf=read(folder/'cell_info.json');name=folder.name;ro=root/'phase_II/readout_runs'/name;tc=np.load(ro/'TC_canonical.npz');M=tensor(np.load(ro/'M.npy'));d=tensor(np.load(ro/'d.npy'));h,z=infer_canonical(ks,W1,b1,tensor(tc['K']),tensor(tc['e']));ss=np.load(ro/'fit_statistics.npz');rows,f=metrics(h,z,M,d,(tensor(ss['v']),tensor(ss['r'])),edges);csvwrite(folder/'candidate_metrics_audit.csv',rows);ids=read(folder/'accepted_ids.json');c=cache(h,z,M,d,W,b,edges);c['native_W_norm']=W.norm();met=score(factor(M,d,W,b,ids),c,True);write(folder/'full_A_metrics.json',met)
  ck=read(ro/'readout_lock.json')['TC_checkpoint'];cp=load_checkpoint(ck);s=SparseTranscoder(12,inf['width'],12).cuda().double();s.load_state_dict(cp['model']);s.eval().requires_grad_(False);pred=s.decoder(z);full=t.W2(h);se=pred@q;native=h@W.T+b;tau=torch.maximum(torch.full_like(tensor(ss['r']),1e-8),1e-3*tensor(ss['r']));fidelity=dict(full_output_MSE=float((pred-full).square().mean()),semantic_MSE=float((se-native).square().mean()),exact_positive_L0=float((z>0).sum(1).double().mean()),meaningful_L0=float((z>tau).sum(1).double().mean()),fraction_active=float((z>0).double().mean()),dead=int((~(z>0).any(0)).sum()))
  info={**inf,'confirmed':sum(rows[m]['status']=='ACCEPT' for m in ids),**met,'audit_fidelity':fidelity};write(folder/'cell_summary.json',info);summary.append(info)
  seq=read(folder/'sequence_lock.json');bm=[]
  for budget in seq['evaluation_budgets']:
   met=score(factor(M,d,W,b,seq['sequence'][budget]),c,True);bm.append(dict(width=inf['width'],seed=inf['seed'],budget=budget,**met));budgets.append(bm[-1])
  csvwrite(folder/'budget_A_metrics.csv',bm);print('WIDTH AUDIT',name,info['accepted'],info['rho_N'],info['rho_L'],flush=True)
 csvwrite(root/'phase_II/width_summary.csv',summary);csvwrite(root/'phase_II/equal_budget_metrics.csv',budgets);write(root/'phase_II/audit_exposure.json',dict(reused_development_audit=True,all_primary_locks_before_audit=True,counts={k:dict(rows=v['rows'],edges=v['n_edges']) for k,v in bks.items()},TC_train_validation_disjoint=True))
 from .reports import phase2_report,finish
 phase2_report(root,summary,budgets);finish(root)

if __name__=='__main__':
 root=Path(sys.argv[2])
 if sys.argv[1]=='worker':worker(root,int(sys.argv[3]))
 else:audit(root)
