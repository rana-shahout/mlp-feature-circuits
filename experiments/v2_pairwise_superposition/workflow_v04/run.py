"""One bounded run: prepare -> discover -> native -> confirm -> report."""
import argparse,csv,shutil,subprocess,sys
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from src.models import SparseTranscoder
from src.runtime import load_checkpoint
from ..stage2.core import frozen_teacher
from .core import *
BASE=Path('experiments/v2_pairwise_superposition');DATA=BASE/'data/first-teacher-20261027/dataset.npz'
MAN=BASE/'data/first-teacher-20261027/split_manifest.json'
OUTPUTS=[4,6,0]
def csvwrite(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def embeddings():
 with np.load(DATA) as z:p,q=z['P'],z['Q']
 return torch.tensor(p,device='cuda',dtype=torch.float64),torch.tensor(q,device='cuda',dtype=torch.float64)
def student(run):
 prot=read(run/'protocol.json');cp=load_checkpoint(prot['student']);assert sha(prot['student'])==prot['hashes'][prot['student']]
 m=SparseTranscoder(12,1024,12).cuda();m.load_state_dict(cp['model']);return m.eval().requires_grad_(False).double()
def teacher(run):
 prot=read(run/'protocol.json');assert sha(prot['teacher'])==prot['hashes'][prot['teacher']]
 m,cp=frozen_teacher(Path(prot['teacher']),'cuda');return m.double()
def prepare(run):
 assert torch.cuda.is_available(),'CUDA required';torch.set_num_threads(4)
 run.mkdir(parents=True,exist_ok=False)
 hist=[p for root in [BASE/'checkpoints',BASE/'data',BASE/'reports',BASE/'runs',BASE/'traces',Path('results'),Path('logs')] for p in root.rglob('*') if p.is_file()]
 write(run/'protected_before.json',{str(p):sha(p) for p in hist})
 snap=run/'source_snapshot';snap.mkdir()
 for p in [Path('SPEC.md'),Path('CODEX_PROMPT.md'),*Path('src').glob('*.py'),*BASE.rglob('*.py')]:
  if 'source_snapshot' not in p.parts:shutil.copyfile(p,snap/str(p).replace('/','__'))
 (run/'git_status.txt').write_text(subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout)
 (run/'gpu.txt').write_text(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
 v=read(BASE/'runs/stage2-transcoders-20261027/teacher_verification.json');sel=read(BASE/'reports/stage2-transcoders-20261027/selection.json');assert sel['best_validation_fidelity']=='w1024_l0.0001'
 tp=Path(v['checkpoint']);sp=BASE/'checkpoints/stage2-transcoders-20261027'/sel['best_validation_fidelity']/'best.pt'
 assert sha(tp)==v['checkpoint_sha256'];assert sha(DATA)==v['dataset_sha256'];assert sha(MAN)==v['split_manifest_sha256']
 cp=load_checkpoint(sp);ident=read(BASE/'runs/stage3-and78-20261027/model_verification.json');assert sha(sp)==ident['transcoder_sha256'];assert cp['identity']['teacher_sha256']==sha(tp);assert cp['identity']['dataset_sha256']==sha(DATA);assert cp['width']==1024 and cp['lambda']==.0001
 assert {k:tuple(v.shape) for k,v in cp['model'].items()}=={'encoder.weight':(1024,12),'encoder.bias':(1024,),'decoder.weight':(12,1024),'decoder.bias':(12,)}
 tc=load_checkpoint(tp);assert tc['identity']['dataset_sha256']==sha(DATA);assert tc['identity']['config_sha256']==sha(BASE/'runs/capacity256-20261027/config.yaml')
 p,q=embeddings();assert p.shape==(12,20) and q.shape==(12,9);assert torch.linalg.matrix_rank(p)==12;torch.testing.assert_close(p.norm(dim=0),torch.ones(20,device='cuda',dtype=p.dtype),atol=2e-7,rtol=2e-7);torch.testing.assert_close(q.T@q,torch.eye(9,device='cuda',dtype=p.dtype),atol=2e-7,rtol=2e-7)
 prot={'version':'0.4','created':now(),'teacher':str(tp),'student':str(sp),'hashes':{str(f):sha(f) for f in [tp,sp,DATA,MAN,Path('SPEC.md'),Path('CODEX_PROMPT.md')]},'P_raw_sha256':raw(p.float().cpu().numpy()),'Q_raw_sha256':raw(q.float().cpu().numpy()),'dimensions':{'P':[12,20],'Q':[12,9],'teacher':[12,256,12],'student':[12,1024,12]},'outputs':OUTPUTS,'seeds':[5101,5102],'budgets':BUDGETS,'random_replicates':20,'max_swaps':5,'thresholds':{'corner_mae':.05,'corner_p95':.10,'joint_mae':.10},'partition':'SHA256 of ASCII 5101: followed by zero-padded 20-bit full input key, first byte low bit; 0=D 1=H','sampling':'seed5101 shuffle each density pool, interleave sizes3/4, D cap64 H cap128; all 190 pairs pre-generated','score':'mean absolute full student joint interaction','historical_teacher_gate':'accepted pilot; failed original >=99% all-primitive balanced-accuracy gate','holdout':'development explanation holdout; model training overlap tagged; selected complete corners all same hash partition; not IID','narrative':'conditional hinges with actual biases and background q; no pure detector inferred; gates strict pre>0, nearzero margin<=1e-7','worked_rule':'lowest background key in confirmation, use lexicographically first eligible nuisance pair if any instead; metadata only','source_hashes':{str(f):sha(f) for f in snap.iterdir()},'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'original_seed':3105,'analysis':'FP64 copies; FP32 reference; no fitting or training'}
 write(run/'protocol.json',prot)
 from .reservation import index
 index(BASE/'data/balanced-semantic-20261027/reserved_final_UNSCORED.npz',run)
 reserved=set(map(int,np.load(run/'protected_final_keys.npy')));original=read(MAN)['partitions'];known={int(k):name for name,part in original.items() for k in part['vector_keys']};test={k for k,v in known.items() if v=='test'};old=set();history=[]
 # Recover only input arrays from historical probe files; no stored responses read.
 files=list((BASE/'reports').rglob('*.npz'))+list((BASE/'traces').rglob('selected_inputs.npz'))
 for f in files:
  chosen=[]
  with np.load(f) as z:
   for name in z.files:
    if name=='a' or name.endswith('_corners') or (f.name=='development_contexts.npz'):
     a=z[name]
     if a.ndim>=2 and a.shape[-1]==20 and np.all((a==0)|(a==1)):
      if f.name=='development_contexts.npz':a=corners(keys(a),(7,8))
      old.update(map(int,keys(a).reshape(-1)));chosen.append(name)
  if chosen:history.append({'path':str(f),'sha256':sha(f),'input_arrays':chosen})
 excluded=reserved|test|old;np.savez_compressed(run/'exclusion_keys.npz',reserved=np.array(sorted(reserved)),test=np.array(sorted(test)),historical=np.array(sorted(old)))
 write(run/'history_identity_sources.json',{'sources':history,'limitations':'Recoverable stored probe inputs excluded. No guarantee against unrecorded prior exposure; researchers knew expected pairs. Discovery is procedural, not human-blind.'})
 rng=np.random.default_rng(5101);manifest={};Dkeys=set();Hkeys=set()
 for pair in combinations(range(20),2):
  pools={0:{3:[],4:[]},1:{3:[],4:[]}};counts={'before':3876,'excluded':0,'mixed':0}
  other=[i for i in range(20) if i not in pair]
  for size in (3,4):
   for active in combinations(other,size):
    bg=sum(1<<i for i in active);ks=[bg,bg|(1<<pair[1]),bg|(1<<pair[0]),bg|(1<<pair[0])|(1<<pair[1])]
    if any(k in excluded for k in ks):counts['excluded']+=1;continue
    ss=[split(k) for k in ks]
    if len(set(ss))!=1:counts['mixed']+=1;continue
    pools[ss[0]][size].append(bg)
  record={'pair':pair,'conditioning':counts}
  for s,label,cap,seen in [(0,'D',64,Dkeys),(1,'H',128,Hkeys)]:
   for size in (3,4):rng.shuffle(pools[s][size])
   bg=[]
   for n in range(max(map(len,pools[s].values()))):
    for size in (3,4):
     if n<len(pools[s][size]):bg.append(pools[s][size][n])
   bg=bg[:cap];ks=keys(corners(bg,pair));assert all(int(k) not in excluded and split(k)==s for k in ks.flat);seen.update(map(int,ks.flat))
   record[label]={'background_keys':bg,'corner_keys':ks,'training_overlap':[[known.get(int(k),'unseen') for k in row] for row in ks],'eligible_counts':{str(n):len(pools[s][n]) for n in (3,4)},'n':len(bg),'density':{str(n):sum(k.bit_count()==n for k in bg) for n in (3,4)}}
  manifest[f'{pair[0]},{pair[1]}']=record
 assert Dkeys.isdisjoint(Hkeys)
 write(run/'context_manifest.json',manifest);write(run/'manifest_validation.json',{'globally_disjoint':True,'complete_corner_exclusions':True,'D_unique_inputs':len(Dkeys),'H_unique_inputs':len(Hkeys),'all_pairs':190,'protected_final_scored':False,'final_file_access':'input identity only, documented in reservation_access.json'})
 lock(run,'prepare_lock.json',['protocol.json','context_manifest.json','exclusion_keys.npz','protected_final_keys.npy']);print('PREPARE COMPLETE',flush=True)
def panel_info(d,j,ids):
 d=cpu(d);pol=polynomial(np.moveaxis(d['signed'],1,-1));info=[]
 for u in ids:
  gate=d['pre'][:,:,u]>0;codes=gate@(1<<np.arange(4));vals,cts=np.unique(codes,return_counts=True)
  info.append({'unit':int(u),'complete_input_row':d['C'][u],'complete_output_write':d['R'][:,u],'bias':d['b'][u],'q_min':d['q'][:,u].min(),'q_max':d['q'][:,u].max(),'gate_counts':dict(zip(map(str,vals),map(int,cts))),'modal_gate':int(vals[cts.argmax()]),'margin_min':np.abs(d['pre'][:,:,u]).min(),'nearzero_fraction':np.mean(np.abs(d['pre'][:,:,u])<=1e-7),'mean_signed_alpha':pol[:,u].mean(0),'q_equation':'original bias + sum nonpair C[input]*background[input]','response_rule':'output write times max(q+u*ai+v*ak,0); strictly positive gates; no hidden coupling'})
 return info

@torch.no_grad()
def nominate(outputs,m,p,q,discovery):
 """No teacher, labels, expected pairs, historic IDs, or H metadata accepted."""
 allkeys=sorted({int(k) for r in discovery.values() for row in r['corner_keys'] for k in row});lookup={k:i for i,k in enumerate(allkeys)};scores=[]
 # Deduplicated, batched inference; full per-unit tensors only for primary pairs.
 for start in range(0,len(allkeys),2048):
  a=torch.tensor(inputs(allkeys[start:start+2048]),device='cuda',dtype=p.dtype);scores.append((m(a@p.T)@q).cpu().numpy())
 values=np.concatenate(scores);rows=[];ranked={}
 for j in outputs:
  rr=[]
  for pairkey,r in discovery.items():
   i,k=map(int,pairkey.split(','));idx=np.array([[lookup[int(key)] for key in row] for row in r['corner_keys']]);v=values[idx,j];alpha=polynomial(v);delta=alpha[:,3]
   row={'output':j,'i':i,'k':k,'score':np.abs(delta).mean(),'signed_mean':delta.mean(),'median':np.median(delta),'sd':delta.std(),'positive_fraction':np.mean(delta>0),'negative_fraction':np.mean(delta<0),'zero_fraction':np.mean(delta==0),'n':len(v),'density3':r['density']['3'],'density4':r['density']['4']}
   for t in range(4):row[f'alpha{t}_mean']=alpha[:,t].mean();row[f'alpha{t}_sd']=alpha[:,t].std()
   rr.append(row)
  rr.sort(key=lambda r:(-r['score'],r['i'],r['k']))
  for n,r in enumerate(rr):r['rank']=n+1
  rows+=rr;ranked[str(j)]=rr[:3]
 return rows,ranked,{'unique_discovery_inputs':len(allkeys),'batch':2048}

@torch.no_grad()
def discover(run):
 verify(run,'prepare_lock.json');p,q=embeddings();m=student(run);man=read(run/'context_manifest.json');dis={k:r['D'] for k,r in man.items()}
 rows,top,cache=nominate(OUTPUTS,m,p,q,dis);csvwrite(run/'pair_rankings.csv',rows);hyp={}
 for j in OUTPUTS:
  r=top[str(j)][0];pair=[r['i'],r['k']];bg=dis[f'{pair[0]},{pair[1]}']['background_keys'];d=evaluate(m,p,q,bg,pair,j);arr=cpu(d);al=polynomial(np.moveaxis(arr['signed'],1,-1));means=al.mean(0);ids=set()
  # Top signed joint, sign restricted; then abs for every coefficient.
  for sign in (1,-1):
   ix=[u for u in range(1024) if sign*means[u,3]>0];ids.update(sorted(ix,key=lambda u:(-sign*means[u,3],u))[:3])
  for t in range(4):ids.update(sorted(range(1024),key=lambda u:(-np.abs(al[:,:,t]).mean(0)[u],u))[:2])
  ids=sorted(ids);arr['panel']=np.array(ids);arr['alpha']=al;arr['remainder']=arr['signed'].sum(-1)-arr['signed'][:,:,ids].sum(-1);arr['corner_keys']=np.array(dis[f'{pair[0]},{pair[1]}']['corner_keys']);save_arrays(run/f'discovery_student_{j}.npz',arr)
  info=panel_info(d,j,ids)
  hyp[str(j)]={'output':j,'top3':top[str(j)],'pair':pair,'panel':ids,'components':info,'mean_alpha':polynomial(arr['total']).mean(0),'alpha_sd':polynomial(arr['total']).std(0),'gamma':arr['beta'][j],'remaining_units_mean_alpha':polynomial(arr['remainder']).mean(0),'conditional_hypothesis':f'On density3/4 discovery backgrounds, output {j} has pair ({pair[0]},{pair[1]}) with mean Boolean coefficients '+str(polynomial(arr['total']).mean(0).tolist())+'. It is the sum of the recorded positive and negative affine-ReLU hinges, gamma, and an explicit remaining-unit function. Component gate descriptions are conditional on q; no single-unit semantic detector or invariant gate pattern is presumed. Test full profile and each coefficient on fixed held-out contexts.','domain':'complete permitted D corners, background sizes3/4; no IID or global Boolean assertion','gate_hypothesis':'Modal per-unit discovery pattern is a descriptive candidate; confirmation mismatches are reported as failures of invariant-pattern interpretation.'}
 write(run/'transcoder_hypotheses.json',hyp);write(run/'discovery_cache.json',cache)
 lock(run,'discovery_lock.json',['transcoder_hypotheses.json','pair_rankings.csv','prepare_lock.json',*[f'discovery_student_{j}.npz' for j in OUTPUTS]])
 print('DISCOVERY LOCKED',[(j,hyp[str(j)]['pair']) for j in OUTPUTS],flush=True)

@torch.no_grad()
def native(run):
 verify(run,'prepare_lock.json');verify(run,'discovery_lock.json');p,q=embeddings();m=teacher(run);hyp=read(run/'transcoder_hypotheses.json');man=read(run/'context_manifest.json');result={};rng=np.random.default_rng(5102)
 for j in OUTPUTS:
  hh=hyp[str(j)];pair=hh['pair'];bank=man[f'{pair[0]},{pair[1]}']['D'];d=evaluate(m,p,q,bank['background_keys'],pair,j);tc=np.load(run/f'discovery_student_{j}.npz')['total'];sets={};traces={};primary={};dm=[]
  for method,target in [('guided',torch.tensor(tc,device='cuda')),('native_only',d['total'])]:
   selected,trace=selector(d['signed'],target,d['beta'][j]);traces[method]=trace;sets[method]=selected;chosen=None
   for k,ids in selected.items():
    pred=(d['signed'][:,:,ids].sum(-1)+d['beta'][j]).cpu().numpy();met=metrics(pred,d['total'].cpu().numpy());dm.append({'output':j,'method':method,'budget':int(k),**met})
    if chosen is None and met['profile_pass'] and met['joint_pass']:chosen=int(k)
   primary[method]={'budget':chosen if chosen is not None else 20,'discovery_pass':chosen is not None}
  sets['random']={str(k):[sorted(rng.choice(256,k,replace=False).tolist()) for _ in range(20)] for k in BUDGETS}
  used=sorted({u for method in ['guided','native_only'] for ids in sets[method].values() for u in ids});info=panel_info(d,j,used)
  result[str(j)]={'pair':pair,'sets':sets,'primary':primary,'components':info,'discovery_metrics':dm,'selection_objective':'MSE over all four corners, fixed original beta and readout coefficients; guided target full student, native_only target full teacher','context_independent_sets':True,'worked_selection':worked_metadata(man[f'{pair[0]},{pair[1]}']['H']['background_keys'],pair)}
  write(run/f'selector_traces_{j}.json',traces);save_arrays(run/f'discovery_native_{j}.npz',d)
 write(run/'native_explanations.json',result)
 lock(run,'native_lock.json',['native_explanations.json','discovery_lock.json',*[f'selector_traces_{j}.json' for j in OUTPUTS],*[f'discovery_native_{j}.npz' for j in OUTPUTS]])
 print('NATIVE LOCKED',flush=True)

def nuisance_edges(bg,pair):
 pool=set(bg);return [(b,t,b|(1<<t)) for b in sorted(pool) for t in range(15,20) if t not in pair and not b&(1<<t) and b|(1<<t) in pool]
def worked_metadata(bg,pair):
 edges=nuisance_edges(bg,pair)
 return {'background':edges[0][0] if edges else min(bg),'matched':list(edges[0]) if edges else None,'rule':'first ascending eligible key/nuisance edge, else minimum key; metadata only','matched_edge_count':len(edges)}

@torch.no_grad()
def confirm(run):
 verify(run,'prepare_lock.json');verify(run,'discovery_lock.json');verify(run,'native_lock.json');p,q=embeddings();m=teacher(run);s=student(run);hyp=read(run/'transcoder_hypotheses.json');native=read(run/'native_explanations.json');man=read(run/'context_manifest.json');rows=[];details={}
 for j in OUTPUTS:
  pair=hyp[str(j)]['pair'];bank=man[f'{pair[0]},{pair[1]}']['H'];bg=bank['background_keys'];d=evaluate(m,p,q,bg,pair,j);t=evaluate(s,p,q,bg,pair,j);dn=cpu(d);tn=cpu(t);truth=dn['total'];lookup={b:n for n,b in enumerate(bg)};edges=nuisance_edges(bg,pair);record={'teacher_mean_alpha':polynomial(truth).mean(0),'teacher_sd_alpha':polynomial(truth).std(0),'student_mean_alpha':polynomial(tn['total']).mean(0),'training_overlap_counts':{label:sum(row.count(label) for row in bank['training_overlap']) for label in ['train','validation','test','unseen']},'nuisance_edges':edges,'gate_validation':{},'student_gate_validation':{}}
  for label,array,info in [('native',dn,native[str(j)]['components']),('student',tn,hyp[str(j)]['components'])]:
   out={}
   for comp in info:
    u=comp['unit'];g=array['pre'][:,:,u]>0;codes=g@(1<<np.arange(4));vals,cts=np.unique(codes,return_counts=True)
    ucoef,vcoef=array['C'][u,pair];qs=array['q'][:,u];only11=np.all(g==[False,False,False,True],1)
    if ucoef>0 and vcoef>0:np.testing.assert_array_equal(only11,(qs>-(ucoef+vcoef))&(qs<=-max(ucoef,vcoef)))
    out[str(u)]={'gate_counts':dict(zip(map(str,vals),map(int,cts))),'discovery_modal':comp['modal_gate'],'invariant_modal_failure_fraction':np.mean(codes!=comp['modal_gate']),'only11_fraction':only11.mean(),'missing_i_only_fraction':np.mean(codes==2),'missing_k_only_fraction':np.mean(codes==4),'min_abs_margin':np.abs(array['pre'][:,:,u]).min(),'nearzero_fraction':np.mean(np.abs(array['pre'][:,:,u])<=1e-7),'q_range':[qs.min(),qs.max()]}
   record['gate_validation' if label=='native' else 'student_gate_validation']=out
  preds={'student':tn['total']};predmeta={'student':('student',1024,-1)}
  for method,sets in native[str(j)]['sets'].items():
   for k,v in sets.items():
    for rep,ids in enumerate(v if method=='random' else [v]):
     name=f'{method}_k{k}_r{rep}';preds[name]=dn['beta'][j]+dn['signed'][:,:,ids].sum(-1);predmeta[name]=(method,int(k),rep)
  unseen=np.array(bank['training_overlap'])!='train';context_unseen=unseen.all(1)
  nuisance={};collateral={};remainders={};cancellation={}
  for name,pred in preds.items():
   method,k,rep=predmeta[name];met=metrics(pred,truth);met.update({'output':j,'method':method,'budget':k,'replicate':rep,'n_training_unseen_rows':int(unseen.sum()),'n_all_training_unseen_contexts':int(context_unseen.sum()),'training_unseen_corner_mae':np.abs(pred-truth)[unseen].mean() if unseen.any() else None,'training_unseen_joint_mae':np.abs(polynomial(pred)-polynomial(truth))[context_unseen,3].mean() if context_unseen.any() else None});rows.append(met)
   remainders[name]=truth-pred
   if edges:
    ix0=[lookup[b] for b,t,b1 in edges];ix1=[lookup[b1] for b,t,b1 in edges];change_error=(pred[ix1]-pred[ix0])-(truth[ix1]-truth[ix0]);nuisance[name]={'edges':len(edges),'change_mae':np.abs(change_error).mean(),'change_max':np.abs(change_error).max()}
   if method not in ['random','student']:
    ids=native[str(j)]['sets'][method][str(k)];v=dn['signed'][:,:,ids];cancellation[name]={'positive_corner_mean':np.maximum(v,0).mean((0,2))*len(ids),'negative_corner_mean':np.minimum(v,0).mean((0,2))*len(ids),'mean_absolute_sum':np.abs(v).sum(-1).mean(),'mean_absolute_net':np.abs(v.sum(-1)).mean()}
    collateral[name]=np.einsum('ctr,jr->ctj',dn['h'][:,:,ids],dn['R'][:,ids]);np.testing.assert_allclose(pred+(truth-pred),truth,atol=1e-9,rtol=1e-8)
  # FP32 checkpoint inference reference on exactly these permitted contexts.
  a=torch.tensor(corners(bg,pair),device='cuda',dtype=torch.float32);p32=p.float();q32=q.float();x=a@p32.T
  native32=(m.float()(x.reshape(-1,12))@q32).reshape(len(bg),4,9).cpu().numpy();student32=(s.float()(x)@q32).cpu().numpy();m.double();s.double()
  np.testing.assert_allclose(native32[:,:,j],truth,atol=2e-6,rtol=2e-5);np.testing.assert_allclose(student32[:,:,j],tn['total'],atol=2e-6,rtol=2e-5)
  record['FP32_vs_FP64_max']={'native':np.abs(native32[:,:,j]-truth).max(),'student':np.abs(student32[:,:,j]-tn['total']).max()};record['nuisance_metrics']=nuisance;record['cancellation']=cancellation
  record['nuisance_unit_changes']={}
  for label,array in [('native',dn),('student',tn)]:
   changes=[]
   for b,bit,b1 in edges:
    i0,i1=lookup[b],lookup[b1];delta=array['q'][i1]-array['q'][i0];err=delta-array['C'][:,bit];assert np.max(np.abs(err))<1e-9
    changes.append({'backgrounds':[b,b1],'bit':bit,'q_identity_max':np.abs(err).max(),'gate_switch_count':np.sum((array['pre'][i0]>0)!=(array['pre'][i1]>0)),'output_change':array['total'][i1]-array['total'][i0]})
   record['nuisance_unit_changes'][label]=changes
  dn['corner_keys']=np.array(bank['corner_keys']);tn['corner_keys']=np.array(bank['corner_keys']);dn['FP32_outputs']=native32;tn['FP32_outputs']=student32
  save_arrays(run/f'confirmation_native_{j}.npz',dn);save_arrays(run/f'confirmation_student_{j}.npz',tn);np.savez_compressed(run/f'confirmation_predictions_{j}.npz',**preds);np.savez_compressed(run/f'confirmation_remainders_{j}.npz',**remainders);np.savez_compressed(run/f'collateral_{j}.npz',**collateral);details[str(j)]=record
 csvwrite(run/'confirmation_metrics.csv',rows);write(run/'confirmation_details.json',details);lock(run,'confirmation_lock.json',['confirmation_metrics.csv','confirmation_details.json',*[f'confirmation_native_{j}.npz' for j in OUTPUTS],*[f'confirmation_student_{j}.npz' for j in OUTPUTS]])
 print('CONFIRMATION COMPLETE',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['all','prepare','discover','native','confirm','report']);ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stages=['prepare','discover','native','confirm','report'] if args.stage=='all' else [args.stage]
 for stage in stages:
  print(now(),stage,flush=True)
  if stage=='report':
   from .report import report
   report(args.run)
  else:globals()[stage](args.run)
if __name__=='__main__':main()
