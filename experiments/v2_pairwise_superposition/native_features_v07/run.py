"""SPEC v0.7: one frozen proposal/validation/factorization pass."""
import json, hashlib, shutil, subprocess, sys, csv
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import numpy as np
import torch
from experiments.v2_pairwise_superposition.latent_correspondence_v05.run import models, DATA, MAN, csvwrite
from experiments.v2_pairwise_superposition.latent_correspondence_v05.core import verify
BASE=Path('experiments/v2_pairwise_superposition'); OLD=BASE/'latent_correspondence_runs/v05-20261102T210650Z'
HIST=[173,266,84,910,633,619]
LABELS=['g0: AND(a0,a1)','g1: AND(a0,a2)','g2: AND(a3,a4)','g3: AND(a5,a6)','g4: AND(a7,a8)','g5: AND(a9,a10)','g6: XOR(a11,a12)','g7: a13 AND NOT a14','g8: OR of five ANDs']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(x):
 if torch.is_tensor(x):return clean(x.detach().cpu().numpy())
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,sort_keys=True)+'\n')
def tensor(a):return torch.as_tensor(a,dtype=torch.float64,device='cuda')
def arr(a):return a.detach().cpu().numpy()
def bits(ks):return ((np.asarray(ks,dtype=np.int64)[:,None]>>np.arange(20))&1).astype(float)
def factor(D,d,W,b):
 # No decoder, target observations, or fitting arguments.
 U,s,V=torch.linalg.svd(D,full_matrices=False);cut=max(1e-12,1e-10*float(s[0])) if len(s) else 1e-12;keep=s>cut
 J=(V[keep].T/s[keep])@U[:,keep].T;C=W@J;R=W-C@D;off=b-C@d
 return J,C,R,off,dict(singular_values=s,cutoff=cut,rank=int(keep.sum()),condition=float(s[keep][0]/s[keep][-1]) if keep.any() else None,J_operator_norm=float(1/s[keep][-1]) if keep.any() else 0.)
def calibration(run):
 p=run/'calibration';p.mkdir();a=np.array([[0,0],[0,1],[1,0],[1,1]],float);K=np.array([[1,0],[0,1],[1,1]],float);e=np.array([0,0,-1.]);z=np.maximum(a@K.T+e,0);xor=np.array([0,1,1,0]);saved={};results={};rows=[];dec=[]
 for name,W,b,w,target,subset in [('A',K,e,np.array([[1,1,-2.]]),z,None),('B',np.array([[1,-1],[-1,1.]]),np.zeros(2),np.ones((1,2)),z,None),('B_positive',np.array([[1,-1],[-1,1.]]),np.zeros(2),np.ones((1,2)),np.maximum(a@np.array([[1,-1],[-1,1.]]).T,0),None),('A_incomplete',K,e,np.array([[1,1,-2.]]),z,[0,1])]:
  h=np.maximum(a@W.T+b,0);np.testing.assert_array_equal((h@w.T)[:,0],xor);np.testing.assert_array_equal((z@np.array([1,1,-2.])),xor)
  fit=np.linalg.lstsq(np.column_stack([h,np.ones(4)]),target,rcond=None)[0];D=fit[:-1].T;d=fit[-1];err=h@D.T+d-target;ok=np.max(abs(err),0)<=1e-10;ids=np.flatnonzero(ok) if subset is None else np.array(subset);D2=D[ids];d2=d[ids]
  # CPU-sized exact control; same declared SVD convention.
  u,s,v=np.linalg.svd(D2,full_matrices=False);keep=s>max(1e-12,1e-10*s[0] if len(s) else 0);J=(v[keep].T/s[keep])@u[:,keep].T;C=w@J;R=w-C@D2;off=-C@d2
  np.testing.assert_allclose(h@w.T,(h@D2.T+d2)@C.T+off+h@R.T,atol=1e-12)
  if name=='A':assert ok.all();np.testing.assert_allclose(C,[[1,1,-2]],atol=1e-12)
  if name=='B':assert not ok.any();np.testing.assert_allclose(np.mean(err**2,0),.125);assert np.array_equal(h[0],h[3]);np.testing.assert_array_equal(z@np.array([[1,0,-1],[0,1,-1]]).T,h);assert C.shape==(1,0) and J.shape==(2,0)
  if name=='B_positive':assert ok.all();np.testing.assert_allclose(C,[[1,1]],atol=1e-12)
  if name=='A_incomplete':np.testing.assert_allclose(R,[[0,0,-2]],atol=1e-12)
  vals=dict(D1=np.eye(2),W1=W,C1=W,b1=b,W2=w,b2=np.zeros(1),D_all=D,d_all=d,D2=D2,d2=d2,D2_pinv=J,C2=C,R2=R,b2_feature=off)
  saved.update({name+'_'+k:v for k,v in vals.items()});results[name]=dict(matrices=vals,MSE=np.mean(err**2,0),max_abs=np.max(abs(err),0),accepted_ids=ids)
  for m in range(target.shape[1]):dec.append(dict(network=name,feature=m,status='ACCEPT' if ok[m] else 'REJECT_THIS_READOUT',mse=np.mean(err[:,m]**2),max_abs=max(abs(err[:,m]))))
  for n in range(4):rows.append(dict(network=name,corner=['00','01','10','11'][n],input=a[n].tolist(),h=h[n].tolist(),z=target[n].tolist(),native_read=(h@D.T+d)[n].tolist(),output=float(xor[n]),feature_output=float(((h@D2.T+d2)@C.T+off)[n,0]),residual=float((h@R.T)[n,0])))
 np.savez(p/'matrices.npz',**saved);write(p/'results.json',results);csvwrite(p/'decisions.csv',dec);csvwrite(p/'all_four_corner_values.csv',rows);return results

def metrics(h,z,M,d,stats,edges):
 f=h@M.T+d;err=f-z;v,r=stats;tau=torch.maximum(torch.full_like(r,1e-8),1e-3*r);active=z>tau;n=active.sum(0);ni=len(z)-n
 ss=(err**2).sum(0);asq=(err**2*active).sum(0);az=(z**2*active).sum(0);isq=(err**2*~active).sum(0);en=torch.zeros_like(r);ed=en.clone();ec=torch.zeros_like(n)
 for start in range(0,len(edges),1024):
  idx=torch.as_tensor(edges[start:start+1024],device='cuda');dz=z[idx[:,1]]-z[idx[:,0]];de=err[idx[:,1]]-err[idx[:,0]];en+=(de**2).sum(0);ed+=(dz**2).sum(0);ec+=(abs(dz)>tau).sum(0)
 values=arr(torch.stack([ss/len(z)/v,torch.sqrt(asq/az),torch.sqrt(isq/ni)/r,torch.sqrt(en/ed)]));support=arr(torch.stack([n>=32,ni>=32,(ec>=32)&(ed>1e-24),v>torch.maximum(torch.full_like(v,1e-12),1e-8*r*r)]));out=[]
 for m in range(len(r)):
  avail=[bool(support[3,m]),bool(support[0,m] and az[m]>1e-24),bool(support[1,m] and r[m]>1e-12),bool(support[2,m])];names=['NMSE','active_RRMSE','inactive_leakage','finite_change_RRMSE'];fail=[nm for j,nm in enumerate(names) if avail[j] and values[j,m]>[.05,.2,.05,.2][j]];missing=[nm for j,nm in enumerate(names) if not avail[j]]
  out.append(dict(feature=m,status='REJECT_THIS_READOUT' if fail else 'UNASSESSED' if missing else 'ACCEPT',failed=fail,missing=missing,**{nm:float(values[j,m]) if avail[j] else None for j,nm in enumerate(names)},active=int(n[m]),inactive=int(ni[m]),responsive_edges=int(ec[m]),negative_rms=float(torch.sqrt(torch.mean(torch.minimum(f[:,m],torch.zeros_like(f[:,m]))**2))),negative_min=float(f[:,m].min())))
 return out,f

def banks(run,man):
 splits=read(MAN)['partitions'];protected=list(BASE.rglob('protected_final_keys.npy'));assert protected
 final=set()
 for p in protected:final.update(map(int,np.load(p)))
 assert sha(BASE/'reports/workflow-v04-20261030/protected_final_keys.npy')==read(BASE/'reports/workflow-v04-20261030/reservation_access.json')['index_sha256']
 excluded=set(man['fit_keys']+man['selection_keys'])|final
 historical=set(map(int,np.load(OLD/'exclusion_keys.npz')['historical']))|set(man['ordinary_confirmation_keys']+man['primary_unique_keys']+man['legacy_unique_keys'])
 train=set(splits['train']['vector_keys']);domain=sorted(sum(1<<j for j in c) for n in [3,4,5,6] for c in combinations(range(20),n));assert len(domain)==60249
 mask=sum(1<<j for j in [0,1,7,8,11,12]);out={};manifest={};exposure=[]
 for name,side in [('V',0),('A',1)]:
  ks=[k for k in domain if k not in excluded and hashlib.sha256(f'v0.7|7101|{k&~mask}'.encode()).digest()[0]&1==side];assert len(ks)>=512;lookup={k:i for i,k in enumerate(ks)};edges=[];counts={}
  for j in range(20):
   e=[(lookup[k],lookup[k|(1<<j)]) for k in ks if not k&(1<<j) and k|(1<<j) in lookup];edges+=e;counts[j]=len(e)
  contexts={}
  for i,j in [(7,8),(11,12),(0,1)]:
   cs=[[k,k|(1<<j),k|(1<<i),k|(1<<i)|(1<<j)] for k in ks if not k&((1<<i)|(1<<j)) and all(q in lookup for q in [k|(1<<j),k|(1<<i),k|(1<<i)|(1<<j)])];cs.sort(key=lambda c:tuple(bits([c[0]])[0]));bg={c[0] for c in cs};ne=[(b,t,b|(1<<t)) for b in sorted(bg) for t in range(15,20) if not b&(1<<t) and b|(1<<t) in bg];contexts[f'{i},{j}']=dict(corners=cs,nuisance_context_flips=ne)
  out[name]=(ks,np.array(edges,dtype=np.int64).reshape(-1,2),contexts);np.savez(run/f'{name}_indices.npz',keys=ks,edges=edges)
  manifest[name]=dict(keys=ks,density={n:sum(k.bit_count()==n for k in ks) for n in [3,4,5,6]},edges_by_bit=counts,contexts=contexts,teacher_TC_train_exposure=sum(k in train for k in ks),prior_development_exposure=sum(k in historical for k in ks))
  exposure += [dict(bank=name,key=k,density=k.bit_count(),teacher_TC_train=k in train,prior_development=k in historical) for k in ks]
 assert set(out['V'][0]).isdisjoint(out['A'][0]);assert all(not set(v[0])&excluded for v in out.values())
 write(run/'input_manifests.json',dict(endianness='key=sum(a_j*2**j); a0 least significant',protected_sources={str(p):sha(p) for p in protected},banks=manifest));csvwrite(run/'input_exposure.csv',exposure);return out

@torch.no_grad()
def main(run):
 assert 'SPEC v0.7' in Path('SPEC.md').read_text();assert 'SPEC v0.7' in Path('CODEX_PROMPT.md').read_text();run.mkdir(parents=True,exist_ok=False)
 for name in ['source_snapshot','matrices','figures','worked_cases']:(run/name).mkdir()
 previous={str(p):sha(p) for root in [BASE,Path('results'),Path('logs')] for p in root.rglob('*') if p.is_file() and run not in p.parents and '__pycache__' not in str(p)};write(run/'preservation_before.json',previous)
 for p in [Path(__file__),Path(__file__).with_name('report.py'),Path('SPEC.md'),Path('CODEX_PROMPT.md')]:shutil.copy2(p,run/'source_snapshot'/p.name)
 cal=calibration(run);print('EXACT CALIBRATION PASSED',flush=True)
 assert torch.cuda.is_available(),'CUDA unavailable; exact calibration complete'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 verify(OLD,'readout_lock.json');verify(OLD,'manifest_lock.json');prot=read(OLD/'protocol.json');man=read(OLD/'context_manifest.json');bank=banks(run,man)
 write(run/'protocol_lock.json',dict(version='0.7',seed=7101,cutoff='max(1e-12,1e-10*smax)',thresholds=[.05,.2,.05,.2],support=32,denominator_floor=1e-24,source_hashes={str(p):sha(p) for p in [Path('SPEC.md'),Path('CODEX_PROMPT.md'),OLD/'M.npy',OLD/'d.npy',Path(prot['teacher']),Path(prot['student']),MAN,DATA]},input_manifest_sha256=sha(run/'input_manifests.json'),cuda=torch.version.cuda,torch=torch.__version__,GPU=torch.cuda.get_device_name(),command=sys.argv))
 t,s,p,q=models(OLD);M=tensor(np.load(OLD/'M.npy'));d=tensor(np.load(OLD/'d.npy'));assert M.shape==(1024,256)
 W1=t.W1.weight@p;b1=t.W1.bias;K=s.encoder.weight@p;e=s.encoder.bias;checks={}
 def infer(ks):
  hs=[];zs=[]
  for start in range(0,len(ks),512):
   a=tensor(bits(ks[start:start+512]));hs.append((a@W1.T+b1).relu());zs.append((a@K.T+e).relu())
  return torch.cat(hs),torch.cat(zs)
 hf,zf=infer(man['fit_keys']);v=zf.var(0,unbiased=False);r=(zf.square().mean(0)).sqrt();st=dict(np.load(OLD/'fit_statistics.npz'));torch.testing.assert_close(v,tensor(st['v']),atol=1e-12,rtol=1e-9)
 # Import equivalence only; outputs cannot enter candidate API.
 a=tensor(bits(bank['V'][0][:512]));x=a@p.T;h0=t.hidden(x)[1];z0=s.latents(x);h,z=infer(bank['V'][0][:512]);torch.testing.assert_close(h,h0,atol=1e-10,rtol=1e-9);torch.testing.assert_close(z,z0,atol=1e-10,rtol=1e-9)
 W2=q.T@t.W2.weight;b2=q.T@t.W2.bias;torch.testing.assert_close(h@W2.T+b2,t.W2(h0)@q,atol=1e-10,rtol=1e-9)
 checks['canonical_FP64_max']=float(max((h-h0).abs().max(),(z-z0).abs().max(),(h@W2.T+b2-t.W2(h0)@q).abs().max()))
 t.float();s.float();xf=a.float()@p.float().T;hf32=t.hidden(xf)[1];zf32=s.latents(xf);sf32=t.W2(hf32)@q.float();torch.testing.assert_close(hf32.double(),h,atol=2e-6,rtol=2e-5);torch.testing.assert_close(zf32.double(),z,atol=2e-6,rtol=2e-5);torch.testing.assert_close(sf32.double(),h@W2.T+b2,atol=2e-6,rtol=2e-5);checks['canonical_FP32_max']=float(max((hf32-h).abs().max(),(zf32-z).abs().max(),(sf32-(h@W2.T+b2)).abs().max()))
 hv,zv=infer(bank['V'][0]);rows,fv=metrics(hv,zv,M,d,(v,r),bank['V'][1]);assert rows==metrics(hv,zv,M,d,(v,r),bank['V'][1])[0];csvwrite(run/'candidate_metrics_validation.csv',rows)
 ids=[r['feature'] for r in rows if r['status']=='ACCEPT'];D=M[ids];dd=d[ids]
 for name,status in [('accepted','ACCEPT'),('rejected','REJECT_THIS_READOUT'),('unassessed','UNASSESSED')]:write(run/f'{name}_feature_ids.json',[r['feature'] for r in rows if r['status']==status])
 np.save(run/'matrices/D2.npy',arr(D));np.save(run/'matrices/d2.npy',arr(dd));write(run/'feature_lock.json',{str(p.relative_to(run)):sha(p) for p in [run/'candidate_metrics_validation.csv',run/'accepted_feature_ids.json',run/'rejected_feature_ids.json',run/'unassessed_feature_ids.json',run/'matrices/D2.npy',run/'matrices/d2.npy']});lockhash=sha(run/'feature_lock.json');print('FEATURE LOCK',len(ids),'accepted',flush=True)
 # First output factorization access after lock.
 J,C,R,off,diag=factor(D,dd,W2,b2);write(run/'factorization_lock.json',dict(feature_lock_sha256=lockhash,diagnostics=diag));print('FACTORIZED rank',diag['rank'],flush=True)
 mats=dict(D1=torch.eye(20,device='cuda',dtype=torch.float64),W1=W1,C1=W1,b1=b1,D_all=M,d_all=d,D2=D,d2=dd,D2_pinv=J,W2=W2,b2=b2,C2=C,b2_feature=off,R2=R)
 for name,value in mats.items():np.save(run/'matrices'/f'{name}.npy',arr(value))
 write(run/'matrices/names_and_shapes.json',{name:dict(shape=list(value.shape),row_feature_ids=ids if name=='D2' else None,column_feature_ids=ids if name in ['C2','D2_pinv'] else None,outputs=LABELS if name in ['C2','W2','R2'] else None) for name,value in mats.items()})
 ha,za=infer(bank['A'][0]);audit,fa=metrics(ha,za,M,d,(v,r),bank['A'][1]);csvwrite(run/'candidate_metrics_audit.csv',audit);assert sha(run/'feature_lock.json')==lockhash
 native=ha@W2.T+b2;pred=fa[:,ids]@C.T+off;sub=za[:,ids]@C.T+off;res=ha@R.T;amp=(fa[:,ids]-za[:,ids])@C.T
 torch.testing.assert_close(native,pred+res,atol=1e-8,rtol=1e-8);torch.testing.assert_close(native-sub,res+amp,atol=1e-8,rtol=1e-8)
 output=[]
 for j,label in enumerate(LABELS):
  den=float(native[:,j].norm());output.append(dict(output=label,feature_relative_RMSE=float((native[:,j]-pred[:,j]).norm())/den if den else None,TC_substitution_relative_RMSE=float((native[:,j]-sub[:,j]).norm())/den if den else None,residual_RMS=float(res[:,j].square().mean().sqrt()),residual_max=float(res[:,j].abs().max()),amplified_readout_RMS=float(amp[:,j].square().mean().sqrt()),accounting_max=float((native-pred-res)[:,j].abs().max())))
 csvwrite(run/'output_accounting.csv',output);np.savez(run/'audit_predictions.npz',keys=bank['A'][0],native=arr(native),feature_only=arr(pred),TC_substitution=arr(sub),residual=arr(res),amplified_readout=arr(amp))
 checks.update(exact_accounting_max=float((native-pred-res).abs().max()),substitution_identity_max=float((native-sub-res-amp).abs().max()),R2_relative_Frobenius=float(R.norm()/W2.norm()),aggregate_feature_RRMSE=float((native-pred).norm()/native.norm()),aggregate_TC_RRMSE=float((native-sub).norm()/native.norm()),residual_RMS=float(res.square().mean().sqrt()),residual_max=float(res.abs().max()))
 for name,left,right in [('DJD',D@J@D,D),('JDJ',J@D@J,J),('DJ_symmetric',D@J,(D@J).T),('JD_symmetric',J@D,(J@D).T)]:
  checks[name]=float((left-right).norm()/max(1.,float(right.norm())));assert checks[name]<1e-7
 # Synthetic rank deficient, empty and full-rank controls, independent of decoder.
 for ds in [torch.eye(4,device='cuda',dtype=torch.float64),tensor([[1,0,0,0],[2,0,0,0]]),torch.empty((0,4),device='cuda',dtype=torch.float64)]:
  w=tensor([[1,2,3,4]]);j,c,rr,oo,di=factor(ds,torch.zeros(len(ds),device='cuda',dtype=torch.float64),w,tensor([.3]));torch.testing.assert_close(c@ds+rr,w);assert ds.shape[0]!=4 or float(rr.norm())<1e-12
 np.testing.assert_array_equal(np.load(run/'matrices/D2.npy'),np.load(OLD/'M.npy')[ids]);torch.testing.assert_close(tensor(np.load(run/'matrices/C2.npy'))@D+tensor(np.load(run/'matrices/R2.npy')),W2,atol=1e-9,rtol=1e-9)
 write(run/'numerical_checks.json',checks)
 from .report import report
 report(run,cal,mats,rows,audit,ids,diag,checks,output,bank,hf,ha,za,fa)
 assert all(sha(path)==v for path,v in previous.items()),'Previous artifact changed';write(run/'preservation_audit.json',dict(files_verified=len(previous),all_unchanged=True,no_training=True,protected_final_scored=False))
 write(run/'execution_summary.json',dict(status='complete',exit_code=0,CUDA=True,accepted=len(ids),rejected=sum(r['status']=='REJECT_THIS_READOUT' for r in rows),unassessed=sum(r['status']=='UNASSESSED' for r in rows),audit_confirmed=sum(audit[m]['status']=='ACCEPT' for m in ids),tests='all executable assertions passed',feature_lock_sha256=lockhash));(run/'tests.log').write_text('PASS: exact controls, canonical FP64/FP32, bank exclusions/grouping, deterministic validation, serialization alignment, affine accounting, Penrose/projectors, synthetic empty/rank-deficient/full-rank controls, preservation.\n');print('COMPLETE',run,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
