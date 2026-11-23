"""Locked v0.6 reverse embedding; no training or new input selection."""
import argparse,sys,shutil,subprocess,json,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ..latent_correspondence_v05.core import *
from ..latent_correspondence_v05.run import models,csvwrite,raw_np,DATA,MAN
BASE=Path('experiments/v2_pairwise_superposition')
OLD=BASE/'latent_correspondence_runs/v05-20261102T210650Z'

def arr(x):return numpy(x)
def savejson(p,x):
 def clean(v):
  if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
  if isinstance(v,(list,tuple)):return [clean(a) for a in v]
  if torch.is_tensor(v):return clean(arr(v))
  if isinstance(v,np.ndarray):return clean(v.tolist())
  if isinstance(v,np.generic):return clean(v.item())
  if isinstance(v,float) and not np.isfinite(v):return None
  return v
 write(p,clean(x))
def error(p,t,mu=None):
 r=p-t
 return dict(mse=float(np.mean(r*r)),relative_RMSE=float(np.linalg.norm(r)/np.linalg.norm(t)),centered_relative_error=float(np.linalg.norm(r)/np.linalg.norm(t-mu)) if mu is not None else None)
def check(a,b):
 a=np.asarray(a);b=np.asarray(b);d=np.linalg.norm(a-b);scale=np.linalg.norm(a)+np.linalg.norm(b);rel=d/max(1.,scale);assert rel<=1e-8,(d,rel)
 return dict(max_abs=float(np.max(np.abs(a-b))),relative=float(rel))
def export(run,name,a,meta,status='derived',definition=None):
 a=np.asarray(a,dtype=np.float64);np.save(run/(name+'.npy'),a)
 with open(run/(name+'.csv'),'w') as f:
  if a.ndim==1:np.savetxt(f,np.column_stack([np.arange(len(a)),a]),delimiter=',',header='index,value',comments='')
  else:np.savetxt(f,np.column_stack([np.arange(len(a)),a]),delimiter=',',header='row,'+','.join(map(str,range(a.shape[1]))),comments='')
 meta[name]=dict(shape=list(a.shape),dtype=str(a.dtype),axes=['original row index','original column index'][:a.ndim],original_indices=[list(range(n)) for n in a.shape],construction_status=status,definition=definition or name,fitting_population='original 11200 fit keys' if 'fit' in name else None)

def main(run):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available();run.mkdir(parents=True,exist_ok=False);(run/'figures').mkdir();(run/'source_snapshot').mkdir()
 prior={str(p):sha(p) for p in BASE.rglob('*') if p.is_file() and run not in p.parents and '__pycache__' not in str(p)}
 savejson(run/'protected_before.json',prior)
 archive=BASE/'spec_archive'/('v06-adoption-'+run.name[4:]);archive.mkdir(exist_ok=False)
 for label,source in [('previous_v05',OLD/'source_snapshot'),('active_v06',Path('.'))]:
  dest=archive/label;dest.mkdir()
  for name in ['SPEC.md','CODEX_PROMPT.md']:
   shutil.copy2(source/name,dest/name)
 savejson(archive/'hashes.json',{str(p.relative_to(archive)):sha(p) for p in archive.rglob('*.md')})
 verify(OLD,'readout_lock.json');verify(OLD,'manifest_lock.json');prot=read(OLD/'protocol.json');man=read(OLD/'context_manifest.json')
 assert sha(prot['teacher'])=='47a61c86e48e58f09bd1d2da167a5a3a904e8fdb5a39c171bf542b6994d49c70'
 assert sha(DATA)=='748ac0ba5e3fd7de981802af8f11ca24df3f385d0670d86ea40ae4993a576f0f'
 for f,h in prot['hashes'].items():
  if f not in ['SPEC.md','CODEX_PROMPT.md']:assert sha(f)==h,(f,h)
 shutil.copy2(OLD/'context_manifest.json',run/'context_manifest.json');shutil.copy2(OLD/'readout_lock.json',run/'source_readout_lock.json')
 t,s,p,q=models(OLD);M=np.load(OLD/'M.npy');d=np.load(OLD/'d.npy');cp=torch.load(OLD/'readout.pt',weights_only=False,map_location='cpu');np.testing.assert_array_equal(M,arr(cp['M']));np.testing.assert_array_equal(d,arr(cp['d']))
 sourcefiles=[Path(__file__),Path('SPEC.md'),Path('CODEX_PROMPT.md'),OLD/'protocol.json',OLD/'context_manifest.json',OLD/'readout.pt',DATA,MAN,Path(prot['teacher']),Path(prot['student'])]
 hashes={str(f):sha(f) for f in sourcefiles}
 protocol=dict(version='0.6',created=now(),source_hashes=hashes,git_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),device=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,CUDA=torch.version.cuda,inference_dtype='float64 (same as v0.5)',solver_dtype='float64',D1_convention='FP64 SVD pseudoinverse rcond=1e-10; algebraic input coordinates',seeds=prot['seeds'],population_label='reused development evaluation',final_set_accessed=False,half_rule='reuse v0.5 first5600 and last5600 fit-key order',archive=str(archive))
 savejson(run/'protocol.json',protocol)
 (run/'git.diff').write_text(subprocess.run(['git','diff','HEAD'],capture_output=True,text=True).stdout)
 for f in [Path(__file__),Path('SPEC.md'),Path('CODEX_PROMPT.md')]:shutil.copy2(f,run/'source_snapshot'/f.name)
 A={k:arr(v) for k,v in dict(P=p,Q=q,W1=t.W1.weight,b1=t.W1.bias,W2=t.W2.weight,b2=t.W2.bias,E=s.encoder.weight,e=s.encoder.bias,B=s.decoder.weight,c=s.decoder.bias).items()}
 A.update(S1=A['P'],D1=np.linalg.pinv(A['P'],rcond=1e-10),C1=A['W1']@A['P'],D2=M,d2=d,K=A['E']@A['P'],A=A['Q'].T@A['B'])
 # Recompute frozen CUDA inference only on exact existing keys; never open final arrays.
 pops={};inferencechecks={}
 for pop,ks in [('fit',man['fit_keys']),('selection',man['selection_keys']),('development',man['ordinary_confirmation_keys']),('primary',man['primary_unique_keys']),('legacy',man['legacy_unique_keys'])]:
  bits=tensor(inputs(ks));x=bits@p.T
  with torch.no_grad():u,h=t.hidden(x);z=s.latents(x);y=t.W2(h)
  vals={k:arr(v) for k,v in dict(bits=bits,u=u,h=h,z=z,y=y).items()};vals['keys']=np.array(ks);vals['zn']=vals['h']@M.T+d;vals['y_alg']=vals['h']@A['W2'].T+A['b2'];pops[pop]=vals
  inferencechecks[pop]=check(vals['y'],vals['y_alg']);np.savez_compressed(run/(pop+'_activations.npz'),**vals)
  print('EXTRACTED',pop,len(ks),flush=True)
 f=pops['fit'];sel=pops['selection'];dev=pops['development'];hs=target_statistics(tensor(f['h']));zs=target_statistics(tensor(f['z']))
 lm=feature_metrics(dev['z'],dev['zn'],dict(np.load(OLD/'fit_statistics.npz')));assert sum(r['well_read'] for r in lm)==493
 for m in FEATURES:assert lm[m]['well_read']
 csvwrite(run/'metrics_latent.csv',lm);savejson(run/'source_verification.json',dict(hashes=hashes,readout_matches_checkpoint=True,historical_well_read=493,density_counts={str(n):int((dev['bits'].sum(1)==n).sum()) for n in [3,4,5,6]},inference_rounding=inferencechecks))
 print('SOURCES VERIFIED; fitting reverse design',flush=True)
 fit=design_fit(tensor(f['z']),tensor(f['h']));grid=[];candidates={}
 for alpha in GRID:
  mod=solve(fit,alpha);pred=arr(predict(mod,tensor(sel['z'])));score=np.mean(np.mean((pred-sel['h'])**2,0)[arr(hs['eligible'])]/arr(hs['v'])[arr(hs['eligible'])]);grid.append(dict(alpha=alpha,score=score,normal_equation_relative_residual=mod['normal_equation_relative_residual']));candidates[alpha]=mod
 alpha=choose_alpha(grid);mod=candidates[alpha];S=arr(mod['M']);qq=arr(mod['d']);csvwrite(run/'alpha_grid.csv',grid)
 torch.save(to_cpu(compact_model(mod,fit)),run/'embedding.pt');np.savez_compressed(run/'fit_statistics.npz',**{k:arr(v) for k,v in hs.items()},mu_predictor=arr(fit['mu_h']),std_predictor=arr(fit['s_h']),constant_predictor=arr(fit['constant_mask']))
 np.save(run/'reverse_design_singular_values.npy',arr(fit['s']));np.save(run/'M_singular_values.npy',np.linalg.svd(M,compute_uv=False))
 savejson(run/'embedding_lock.json',dict(alpha=alpha,rank=fit['rank'],grid=grid,embedding_sha256=sha(run/'embedding.pt'),fit_keys_sha256=raw_np(f['keys']),selection_keys_sha256=raw_np(sel['keys']),svd_reconstruction_max=fit['svd_reconstruction_max'],condition=float(fit['s'][0]/fit['s'][fit['rank']-1])))
 pinv=np.linalg.pinv(M,rcond=1e-10)
 for name,ss,bb in [('fit',S,qq),('pinv',pinv,-pinv@d)]:
  cc=A['W2']@ss;offset=A['b2']+A['W2']@bb
  A.update({f'S2_{name}':ss,f'q2_{name}':bb,f'C2_{name}':cc,f'b2_feature_{name}':offset,f'L2_{name}':A['Q'].T@cc,f'beta2_feature_{name}':A['Q'].T@offset})
 A.update(S2=S,C2=A['C2_fit']);metadata={}
 for name,a in A.items():export(run,name,a,metadata)
 for rec in metadata.values():rec['source_hashes']=hashes
 savejson(run/'matrix_metadata.json',metadata);np.savez_compressed(run/'matrices.npz',**A)
 print('PRIMARY LOCKED',alpha,'rank',fit['rank'],flush=True)
 halves={};stability=[]
 for name,idx in [('half1',np.arange(5600)),('half2',np.arange(5600,11200))]:
  hf=design_fit(tensor(f['z'][idx]),tensor(f['h'][idx]));hm=solve(hf,alpha);torch.save(to_cpu(compact_model(hm,hf)),run/(name+'_embedding.pt'));savejson(run/(name+'_manifest.json'),f['keys'][idx]);ss=arr(hm['M']);bb=arr(hm['d']);halves[name]=(ss,bb)
  for space,base,other in [('S2',S,ss),('C2',A['C2_fit'],A['W2']@ss)]:
   for m in range(1024):
    v,w=base[:,m],other[:,m];nv,nw=np.linalg.norm(v),np.linalg.norm(w);stability.append(dict(half=name,space=space,feature=m,primary_norm=nv,half_norm=nw,cosine=np.dot(v,w)/(nv*nw) if nv*nw else None,relative_difference=np.linalg.norm(v-w)/nv if nv else None))
 csvwrite(run/'coefficient_stability.csv',stability)
 stab={name:dict(hidden=error(dev['z']@ss.T+bb,dev['z']@S.T+qq,arr(hs['mu_z'])),normalized_prediction_MSE=float(np.mean(np.mean(((dev['z']@ss.T+bb)-(dev['z']@S.T+qq))**2,0)[arr(hs['eligible'])]/arr(hs['v'])[arr(hs['eligible'])]))) for name,(ss,bb) in halves.items()};savejson(run/'prediction_stability.json',stab)
 summaries={};hiddenrows=[];outrows=[];identities={};diagnostics={};decomp={};predictions={}
 for pop,v in pops.items():
  summaries[pop]={};predictions[pop]={}
  for name,ss,bb in [('fit',S,qq),('pinv',pinv,A['q2_pinv']),('mean',np.zeros_like(S),f['h'].mean(0))]:
   for code in ['z','zn']:
    hh=v[code]@ss.T+bb;yy=hh@A['W2'].T+A['b2'];label=name+'_'+code;predictions[pop][label]=(hh,yy)
    rows=feature_metrics(v['h'],hh,hs)
    for r in rows:r.update(population=pop,method=label,neuron=r.pop('feature'),status='unassessable' if not r['fit_eligible'] or r['meaningful_activity']<32 else 'passing' if r['well_read'] else 'failing')
    hiddenrows+=rows;summaries[pop][label]=dict(hidden=error(hh,v['h'],f['h'].mean(0)),output=error(yy,v['y']),passing=sum(r['status']=='passing' for r in rows),failing=sum(r['status']=='failing' for r in rows),unassessable=sum(r['status']=='unassessable' for r in rows))
    for space,target,pred in [('raw',v['y'],yy),('semantic',v['y']@A['Q'],yy@A['Q']),('orthogonal',v['y']-(v['y']@A['Q'])@A['Q'].T,yy-(yy@A['Q'])@A['Q'].T)]:
     for j in range(target.shape[1]):outrows.append(dict(population=pop,method=label,space=space,index=j,**error(pred[:,j],target[:,j])))
  tc=v['z']@A['B'].T+A['c'];
  for space,target,pred in [('raw',v['y'],tc),('semantic',v['y']@A['Q'],tc@A['Q']),('orthogonal',v['y']-(v['y']@A['Q'])@A['Q'].T,tc-(tc@A['Q'])@A['Q'].T)]:
   for j in range(target.shape[1]):outrows.append(dict(population=pop,method='TC',space=space,index=j,**error(pred[:,j],target[:,j])))
  summaries[pop]['TC']=error(tc,v['y']);summaries[pop]['fit_vs_TC']=error(predictions[pop]['fit_z'][1],tc)
  for name in ['fit','pinv']:
   ss=A['S2_'+name];bb=A['q2_'+name];cc=A['C2_'+name];off=A['b2_feature_'+name];h=v['h'];z=v['z'];zn=v['zn'];ez=z-zn;rN=zn@ss.T+bb-h;se=ez@ss.T;rT=z@ss.T+bb-h;y=v['y_alg'];yn=zn@cc.T+off;yt=z@cc.T+off;rz=(rT+h)@M.T+d-z
   tests=[(rN,h@(ss@M-np.eye(256)).T+ss@d+bb),(rT,rN+se),(yn-y,rN@A['W2'].T),(yt-y,rT@A['W2'].T),(yt-yn,ez@cc.T),(yn-y,h@(cc@M-A['W2']).T+cc@d+off-A['b2']),(rz,rT@M.T-ez)]
   identities[pop+'_'+name]=[check(a,b) for a,b in tests]
   for space,rn,rt,ee in [('hidden',rN,rT,se),('output',rN@A['W2'].T,rT@A['W2'].T,se@A['W2'].T)]:decomp[pop+'_'+name+'_'+space]=dict(rT_energy=np.sum(rt**2),rN_energy=np.sum(rn**2),Se_energy=np.sum(ee**2),cross_term=2*np.sum(rn*ee))
   if pop=='development':
    ep=(z-d)-(z-d)@pinv.T@M.T;diagnostics[name]=dict(outside_image_energy_fraction=np.sum(ep**2)/np.sum((z-d)**2),latent_roundtrip=error(rz+z,z),S2_operator_norm=np.linalg.norm(ss,2))
    for rn,rv in [('S2D2_minus_I',ss@M-np.eye(256)),('hidden_bias',ss@d+bb),('C2D2_minus_W2',cc@M-A['W2']),('output_bias',cc@d+off-A['b2'])]:
     export(run,name+'_'+rn,rv,metadata);diagnostics[name][rn]=dict(frobenius_norm=np.linalg.norm(rv),operator_norm=np.linalg.norm(rv,2) if rv.ndim==2 else np.linalg.norm(rv))
 csvwrite(run/'metrics_hidden.csv',hiddenrows);csvwrite(run/'metrics_output.csv',outrows);savejson(run/'metric_summary.json',summaries);savejson(run/'numerical_validation.json',identities);savejson(run/'matrix_diagnostics.json',diagnostics);savejson(run/'error_decomposition.json',decomp)
 # Worked examples: exact historical keys and display neurons, plus all 256/1024 terms.
 worked={};worktables=[];slices=[];gatechecks={};workinfo={}
 for m in FEATURES:
  old=np.load(OLD/f'feature_{m}_worked.npz');ks=old['keys'].reshape(-1);bits=tensor(inputs(ks));x=bits@p.T
  with torch.no_grad():u,h=t.hidden(x);z=s.latents(x)
  u,h,z=map(arr,[u,h,z]);bits=arr(bits);zn=h@M.T+d;panel=old['panel'];con=h*M[m];rem=con.sum(1)-con[:,panel].sum(1);ua=bits@A['C1'].T+A['b1'];ga=ua>0;g=u>0;L=(M[None,:,:]*ga[:,None,:])@A['C1'];o=(M[None,:,:]*ga[:,None,:])@A['b1']+d
  gatechecks[str(m)]=dict(affine=check(np.einsum('nki,ni->nk',L,bits)+o,np.maximum(ua,0)@M.T+d),pre=check(ua,u),gate_changes=int(np.sum(ga!=g)))
  population=man['worked'][','.join(map(str,FEATURE_CASE[m][1]))]['population'];workinfo[m]=dict(population=population,keys=ks.tolist(),panel=panel.tolist(),TC_response=z[:,m].tolist(),native_response=zn[:,m].tolist(),active_gates=g.sum(1).tolist(),TC_pre=(bits@A['K'][m]+A['e'][m]).tolist())
  vals=dict(keys=ks,bits=bits,u=u,gates=g,h=h,z=z,zn=zn,contributions=con,remainder=rem,L1=L,o1=o,panel=panel,algebraic_input_coordinates=bits@A['P'].T@A['D1'].T)
  for code,zz in [('TC',z),('native',zn)]:
   allc=zz[:,:,None]*A['C2_fit'].T[None,:,:];vals[code+'_all_output_contributions']=allc;vals[code+'_remaining_output_sum']=allc.sum(1)-allc[:,m];vals[code+'_offset']=A['b2_feature_fit']
  for k,v in vals.items():worked[f'f{m}_{k}']=v
  for n,key in enumerate(ks):
   for r in range(256):worktables.append(dict(feature=m,population=population,key=int(key),corner=['00','01','10','11'][n%4],neuron=r,displayed=bool(r in panel),pre=u[n,r],gate=int(g[n,r]),h=h[n,r],D2=M[m,r],signed_contribution=con[n,r],undisplayed_remainder=rem[n],intercept=d[m],zN=zn[n,m],zT=z[n,m],error=zn[n,m]-z[n,m]))
  for name,vec in [('D2_row',M[m]),('S2_column',S[:,m]),('C2_column',A['C2_fit'][:,m]),('semantic_C2',A['L2_fit'][:,m]),('B_column',A['B'][:,m]),('A_column',A['A'][:,m])]:
   slices += [dict(feature=m,matrix=name,index=i,value=value) for i,value in enumerate(vec)]
 np.savez_compressed(run/'worked_examples.npz',**worked);csvwrite(run/'worked_native_tables.csv',worktables);csvwrite(run/'feature_matrix_slices.csv',slices);savejson(run/'worked_contexts.json',workinfo);savejson(run/'gate_validation.json',gatechecks)
 # Finite response diagnostics on all existing contexts; primary and legacy remain separate.
 responses=[];changes={}
 for population,banks in [('primary',man['primary_contexts']),('legacy',man['legacy_contexts'])]:
  v=pops[population];lookup={int(k):i for i,k in enumerate(v['keys'])}
  for pair,bank in banks.items():
   ids=np.array([[lookup[k] for k in row] for row in bank['corner_keys']],dtype=int)
   if not len(ids):continue
   for method in ['fit_z','fit_zn','pinv_z','pinv_zn','mean_z']:
    for space,tt,pp in [('hidden',v['h'],predictions[population][method][0]),('output',v['y'],predictions[population][method][1])]:
     target=tt[ids];pred=pp[ids];ops={'first_bit':np.array([0,-1,0,1]),'second_bit':np.array([0,0,-1,1]),'interaction':np.array([1,-1,-1,1])}
     change_pairs=[(typ,np.einsum('c,ncr->nr',op,target),np.einsum('c,ncr->nr',op,pred)) for typ,op in ops.items()]
     edge_ids={int(b):i for i,b in enumerate(bank['background_keys'])}
     if bank['nuisance_edges']:
      ei=np.array([[edge_ids[int(e[0])],edge_ids[int(e[2])]] for e in bank['nuisance_edges']]);change_pairs.append(('nuisance',(target[ei[:,1]]-target[ei[:,0]]).reshape(-1,target.shape[-1]),(pred[ei[:,1]]-pred[ei[:,0]]).reshape(-1,pred.shape[-1])))
     for typ,dt,dp in change_pairs:
      norm=np.linalg.norm(dt,axis=1);ref=f['h'] if space=='hidden' else f['y'];threshold=1e-8*max(1,np.sqrt(np.mean(np.sum(ref**2,1))));eligible=norm>threshold;count=int(eligible.sum());key=f'{population}_{pair}_{method}_{space}_{typ}';changes[key+'_target']=dt;changes[key+'_prediction']=dp;changes[key+'_eligible']=eligible;changes[key+'_target_norm']=norm
      responses.append(dict(population=population,pair=pair,method=method,space=space,type=typ,n=len(dt),eligible=count,status='unassessable' if count<32 else 'supported descriptive summary',relative_RMSE=float(np.linalg.norm((dp-dt)[eligible])/np.linalg.norm(dt[eligible])) if count else None,overlap='recorded membership in context_manifest.json',min_target_norm=float(norm.min()),median_target_norm=float(np.median(norm)),max_target_norm=float(norm.max())))
 csvwrite(run/'finite_response_metrics.csv',responses);np.savez_compressed(run/'finite_changes.npz',**changes)
 savejson(run/'matrix_metadata.json',metadata)
 comparisons=[]
 for name,cc in [('pinv',A['C2_pinv']),('TC',A['B'])]:
  for m in range(1024):
   aa=A['C2_fit'][:,m];bb=cc[:,m];den=np.linalg.norm(aa)*np.linalg.norm(bb);delta=np.linalg.norm(aa-bb)
   comparisons.append(dict(comparison=name,feature=m,difference_norm=delta,activity_weighted_difference=delta*np.std(f['z'][:,m]),cosine=np.dot(aa,bb)/den if den else None))
 csvwrite(run/'output_map_column_comparison.csv',comparisons)
 savejson(run/'input_boundary_diagnostics.json',dict(P_rank=int(np.linalg.matrix_rank(A['P'])),P_singular_values=np.linalg.svd(A['P'],compute_uv=False),PD1=check(A['P']@A['D1'],np.eye(12)),C1D1=check(A['C1']@A['D1'],A['W1']),projected_coordinate_error_norms=np.linalg.norm(dev['bits']@A['P'].T@A['D1'].T-dev['bits'],axis=1),dropped_predictors=np.flatnonzero(arr(fit['constant_mask'])),dropped_varying_development=np.flatnonzero(arr(fit['constant_mask'])&(dev['z'].std(0)>1e-12))))
 # Reload equality is an explicit contract.
 re=dict(np.load(run/'matrices.npz'));np.testing.assert_array_equal(re['S2'],np.load(run/'S2_fit.npy'));np.testing.assert_array_equal(re['C2'],np.load(run/'C2_fit.npy'));check(dev['z']@re['C2'].T+re['b2_feature_fit'],predictions['development']['fit_z'][1])
 print('METRICS COMPLETE',summaries['development'],flush=True)
 report(run,A,fit,summaries,diagnostics,stability,workinfo,worked,man,decomp)
 (run/'HOW_TO_USE_MATRICES.md').write_text('S2.npy and C2.npy alias the primary fitted matrices. Original indices; all arrays FP64. D1 gives algebraic input coordinates, not recovered Boolean bits.\n```python\nimport numpy as np\na=np.load("matrices.npz")\nP,W1,b1,W2,b2=[a[k] for k in ["P","W1","b1","W2","b2"]]\nX=bits@P.T\nH=np.maximum(X@W1.T+b1,0)\nY=H@W2.T+b2\nZ_T=np.maximum(X@a["E"].T+a["e"],0)\nZ_N=H@a["D2"].T+a["d2"]\nH_from_T=Z_T@a["S2_fit"].T+a["q2_fit"]\nH_from_N=Z_N@a["S2_fit"].T+a["q2_fit"]\nY_from_T=Z_T@a["C2_fit"].T+a["b2_feature_fit"]\nY_from_N=Z_N@a["C2_fit"].T+a["b2_feature_fit"]\nsemantic_Y_from_T=Y_from_T@a["Q"]\n```\n')
 assert all(sha(path)==h for path,h in prior.items()),'Historical artifact modified'
 savejson(run/'preservation_verified.json',dict(files_verified=len(prior),final_scored=False,networks_trained=False))
 savejson(run/'complete.json',dict(status='numerical run complete; pending external figure inspection and test log',timestamp=now()))

def report(run,A,fit,summaries,diag,stability,info,worked,man,decomp):
 plt.rcParams.update({'font.size':9});pages=PdfPages(run/'matrix_report.pdf');figs=[]
 def finish(fig,name):
  fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(run/'figures'/(name+'.png'),dpi=160);fig.savefig(run/'figures'/(name+'.pdf'));pages.savefig(fig);figs.append(name);plt.close(fig)
 def heat(ax,a,title,limit=None):
  a=np.atleast_2d(a);lim=limit or max(float(np.abs(a).max()),1e-12);im=ax.imshow(a,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest');ax.set_title(title);ax.set_xlabel('original column index');ax.set_ylabel('original row index');plt.colorbar(im,ax=ax,fraction=.035)
 fig,ax=plt.subplots(figsize=(12,8));ax.axis('off');text='Native computation and matrix inventory\n\na (20) → P (12×20) → x (12) → W1,b1 → ReLU → h (256) → W2,b2 → y (12)\n\nzN = D2 ReLU(C1 a+b1)+d2; C1=W1 P; D2=M (frozen v0.5 readout)\n\nzT = ReLU(E P a+e) → S2,q2 → reconstructed h → native W2,b2 → reconstructed y\n\nC2=W2 S2; b2_feature=b2+W2 q2; semantic map L2=Qᵀ C2\n\nD1=P† (20×12): algebraic input coordinates D1 x. P D1=I12; D1 P is a rank-12 projector.\n\nKnown: P,Q,W1,b1,W2,b2,E,e,B,c. Frozen estimate: D2,d2.\nNew estimate: S2 (256×1024),q2 (256); induced C2 (12×1024).\n\nAll 1024 TC coordinates enter the fit; no rectification of reconstructed h.\nFit 11200 / selection 2800 / reused development evaluation 2000.\nNo retraining, new bank selection, hidden interventions, or final-test scoring.';ax.text(.03,.95,text,va='top',fontsize=13,linespacing=1.7);finish(fig,'01_architecture')
 for suffix,names in [('input',['P','D1','W1','C1']),('hidden',['D2','S2_fit','S2_pinv']),('output',['W2','C2_fit','C2_pinv','B']),('offsets',['b1','b2','d2','q2_fit','q2_pinv','b2_feature_fit','b2_feature_pinv','c'])]:
  cols=2;rows=(len(names)+1)//2;fig,axs=plt.subplots(rows,cols,figsize=(14,3.3*rows),squeeze=False);common=max(np.abs(A[n]).max() for n in names if n in ['C2_fit','C2_pinv','B']) if suffix=='output' else None
  for ax,name in zip(axs.flat,names):heat(ax,A[name],name+' '+str(A[name].shape),max(np.abs(A['S2_fit']).max(),np.abs(A['S2_pinv']).max()) if name in ['S2_fit','S2_pinv'] else max(np.abs(A['q2_fit']).max(),np.abs(A['q2_pinv']).max()) if name in ['q2_fit','q2_pinv'] else max(np.abs(A['b2_feature_fit']).max(),np.abs(A['b2_feature_pinv']).max()) if name in ['b2_feature_fit','b2_feature_pinv'] else common if name in ['C2_fit','C2_pinv','B'] else None)
  for ax in list(axs.flat)[len(names):]:ax.axis('off')
  fig.suptitle('Full indexed matrix atlas — '+suffix+'; full ranges, no clipping');finish(fig,'02_atlas_'+suffix)
 fig,axs=plt.subplots(3,2,figsize=(15,10));lim=max(np.abs(A[n]).max() for n in ['L2_fit','L2_pinv','A'])
 for row,n in enumerate(['L2_fit','L2_pinv','A']):
  heat(axs[row,0],A[n],n+' full semantic map',lim);sub=A[n][:,FEATURES];heat(axs[row,1],sub,n+' fixed features '+str(FEATURES),lim)
  for i in range(9):
   for j in range(6):axs[row,1].text(j,i,f'{sub[i,j]:.3g}',ha='center',va='center',fontsize=7)
 finish(fig,'03_semantic_maps')
 fig,axs=plt.subplots(2,3,figsize=(15,8));summary=summaries['development'];methods=['fit_z','pinv_z','mean_z','fit_zn','pinv_zn'];axs[0,0].bar(methods,[summary[k]['hidden']['relative_RMSE'] for k in methods]);axs[0,0].set_title('Hidden relative RMSE; actual TC / native codes');axs[0,0].tick_params(axis='x',rotation=35);axs[0,1].bar(methods+['TC'],[summary[k]['output']['relative_RMSE'] for k in methods]+[summary['TC']['relative_RMSE']]);axs[0,1].tick_params(axis='x',rotation=35);axs[0,1].set_title('Output relative RMSE');axs[0,2].semilogy(np.linalg.svd(A['D2'],compute_uv=False),label='M');axs[0,2].semilogy(arr(fit['s']),label='standardized Z design');axs[0,2].legend();axs[0,2].set_title('Singular spectra')
 heat(axs[1,0],A['S2_fit']@A['D2']-np.eye(256),'S2 D2 − I');heat(axs[1,1],A['C2_fit']@A['D2']-A['W2'],'C2 D2 − W2');ss=[r for r in stability if r['half']=='half1' and r['space']=='C2'];axs[1,2].plot([r['cosine'] for r in ss],'.',ms=2);axs[1,2].set_title('C2 primary vs half1 column cosine');fig.suptitle('Reused development evaluation n=2000 — reconstruction and identifiability');finish(fig,'04_reconstruction')
 for m in FEATURES:
  w={k[len(f'f{m}_'):]:v for k,v in worked.items() if k.startswith(f'f{m}_')};panel=w['panel'];fig,axs=plt.subplots(3,3,figsize=(16,11));heat(axs[0,0],w['bits'],'Actual Boolean inputs; corners 00,01,10,11');heat(axs[0,1],A['C1'][panel],'Native C1 rows; neurons '+str(panel.tolist()));heat(axs[0,2],np.column_stack([A['b1'][panel],A['D2'][m,panel]]),'b1 and D2 weights; same displayed neurons');heat(axs[1,0],w['u'][:,panel],'Native preactivation u');heat(axs[1,1],w['gates'][:,panel].astype(float),'Native ReLU gates');ax=axs[1,2];ax.axis('off');tab=np.column_stack([w['contributions'][:,panel],w['remainder'],np.full(len(w['keys']),A['d2'][m]),w['zn'][:,m],w['z'][:,m]]);ax.table(cellText=[[f'{v:.4f}' for v in row] for row in tab],colLabels=[str(r) for r in panel]+['rest','d2','zN','zT'],rowLabels=[['00','01','10','11'][i%4] for i in range(len(tab))],loc='center');[table.auto_set_font_size(False) for table in ax.tables];[table.set_fontsize(5) for table in ax.tables];ax.set_title('D2 h terms + exact undisplayed remainder + d2')
  axs[2,0].plot(A['D2'][m],label='D2 row');axs[2,0].plot(A['S2_fit'][:,m],label='S2 column');axs[2,0].legend();axs[2,0].set_title('All 256 original native neuron indices');axs[2,1].plot(A['L2_fit'][:,m],'-o',label='Qᵀ C2 fit');axs[2,1].plot(A['A'][:,m],'-o',label='Qᵀ B');axs[2,1].legend();axs[2,1].set_title('All 9 semantic output coefficients');heat(axs[2,2],w['L1'][:,m,:],'Gate-conditioned input-to-feature map L1')
  for ax in [axs[0,1],axs[0,2]]:ax.set_yticks(range(len(panel)),[str(r) for r in panel]);ax.set_ylabel('original native neuron index')
  for ax in [axs[1,0],axs[1,1]]:ax.set_xticks(range(len(panel)),[str(r) for r in panel],rotation=45);ax.set_xlabel('original native neuron index');ax.set_yticks(range(len(w['keys'])),[['00','01','10','11'][i%4] for i in range(len(w['keys']))]);ax.set_ylabel('corner (00,01,10,11)')
  axs[0,2].set_xticks([0,1],['b1','D2']);axs[0,2].set_xlabel('affine bias / readout coefficient')
  role='negative XOR interaction contributor' if m==910 else 'historically selected interaction contributor';fig.suptitle(f'Feature {m}: {role}; {info[m]["population"]}, background {int(w["keys"][0])}; corners 00,01,10,11');finish(fig,f'05_feature_{m}')
 fig,axs=plt.subplots(2,2,figsize=(13,8))
 keys=['development_fit_hidden','development_pinv_hidden','development_fit_output','development_pinv_output']
 for ax,key in zip(axs.flat,keys):
  dd=decomp[key];ax.bar(['rT²','rN²','Se²','cross'],list(dd.values()));ax.set_title(key+' error decomposition')
 fig.suptitle('Reused development evaluation n=2000; squared Frobenius energies and signed cross-terms');finish(fig,'04b_decomposition')
 fig,axs=plt.subplots(2,3,figsize=(15,7))
 for ax,m in zip(axs.flat,FEATURES):
  ax.plot(A['D2'][m],label='D2 row');ax.set_xlabel('original native neuron index');ax.set_title(f'Feature {m} D2 row');ax.legend()
 finish(fig,'05b_D2_rows')
 fig,axs=plt.subplots(2,3,figsize=(15,7))
 for ax,m in zip(axs.flat,FEATURES):
  ax.plot(A['S2_fit'][:,m],label='S2 fit column');ax.plot(A['S2_pinv'][:,m],label='S2 pinv column',alpha=.7);ax.set_xlabel('original native neuron index');ax.set_title(f'Feature {m} reverse embedding');ax.legend()
 finish(fig,'05c_S2_columns')
 fig,axs=plt.subplots(1,3,figsize=(15,4))
 for ax,key in zip(axs,['beta2_feature_fit','beta2_feature_pinv','c']):
  vec=A[key] if key!='c' else A['Q'].T@A['c'];ax.bar(range(9),vec);ax.set_title(key+' semantic affine offset');ax.set_xlabel('semantic index')
 finish(fig,'03b_semantic_offsets')
 fig,axs=plt.subplots(1,2,figsize=(12,5));heat(axs[0],A['P']@A['D1'],'P D1 = I12');heat(axs[1],A['D1']@A['P'],'D1 P: rank-12 projector in 20 dimensions');finish(fig,'06_input_projectors')
 pages.close()
 lockinfo=read(run/'embedding_lock.json');sm=summaries['development'];svals=np.linalg.svd(A['D2'],compute_uv=False)
 lines=['# Native feature matrices v0.6',f'Completed CUDA frozen inference and FP64 reverse ridge. Selected alpha **{lockinfo["alpha"]}**, standardized reverse design rank **{fit["rank"]}**, condition **{lockinfo["condition"]:.4g}**. P rank {np.linalg.matrix_rank(A["P"])}, M rank {sum(svals>1e-10*svals[0])}, M condition {svals[0]/svals[-1]:.4g}.', '','## Reused development evaluation (n=2000)','Method | Hidden relative RMSE | Hidden centered error | Output relative RMSE | Passing neurons','---|---:|---:|---:|---:']
 for k in methods:lines.append(f'{k} | {sm[k]["hidden"]["relative_RMSE"]:.6g} | {sm[k]["hidden"]["centered_relative_error"]:.6g} | {sm[k]["output"]["relative_RMSE"]:.6g} | {sm[k]["passing"]}')
 lines += ['',f'Existing TC output relative RMSE: {sm["TC"]["relative_RMSE"]:.6g}. All affine residual identities and gate-conditioned equations validated. All 493 historically well-read features reproduced, including the six fixed cases.','', 'The native first layer computes C1 a+b1 and thresholds each original neuron. D2 combines the resulting 256 gated responses with its intercept. S2 reconstructs a candidate native hidden state from TC coordinates, and C2=W2 S2 collapses its terminal output computation. Negative affine predictions remain unchanged.','', '## Worked native gates']
 for m in FEATURES:lines.append(f'- Feature {m}: {info[m]["population"]}; TC corner activations {np.round(info[m]["TC_response"],5).tolist()}, native-readout activations {np.round(info[m]["native_response"],5).tolist()}, native gate counts {info[m]["active_gates"]}.')
 lines += ['', 'Feature 910’s negative XOR-task selection role differs from its own activation response. The gate tables carry every native contribution and retain exact undisplayed remainders. C1 rows are dense input combinations; D2 rows and S2 columns are distinct forward/reverse objects. All six induced C2 columns and TC decoder comparisons are exported without reordering.', '', '## Limitations', 'The pseudoinverse native-code round trip is algebraic at full column rank and does not establish actual TC compatibility. Coefficient columns depend on the chosen embedding, correlated predictors, ridge penalty, and fit population. Half-fit column cosines and prediction differences quantify that uncertainty; accurate output alone does not establish accurate hidden reconstruction. No unique intrinsic circuit or independently reachable single-feature intervention is established.', 'Primary response support remains one AND78 background, zero XOR1112/AND01 backgrounds and zero nuisance edges. These primary response claims are unassessable (<32 changes). Legacy illustrations retain original overlap membership and are reported separately. The ordinary bank is reused development evaluation, not a held-out final test. D1 x is algebraic projected input coordinates, not the original Boolean a.', '', '## Matrix inventory', 'P/S1 12×20; D1 20×12; C1 256×20; D2 1024×256; S2 256×1024; C2 12×1024. Frozen matrices, affine offsets, all original indices, fit statistics, full per-neuron/per-feature metrics, residual matrices, and half-fit diagnostics accompany this report.', '', '[Matrix report](matrix_report.pdf) · [Metadata](matrix_metadata.json) · [Reload examples](HOW_TO_USE_MATRICES.md)', '', '## Figures']
 lines += [f'- [{name}](figures/{name}.png) ([PDF](figures/{name}.pdf))' for name in figs]
 (run/'SUMMARY.md').write_text('\n\n'.join(lines)+'\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();main(args.run)
