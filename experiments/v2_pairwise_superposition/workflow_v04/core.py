import hashlib,json,datetime
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from ..trace.core import polynomial,from_polynomial

BUDGETS=[1,3,5,10,20]
BITS=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.uint8)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,v):
 def default(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,Path):return str(x)
  raise TypeError(type(x).__name__)
 Path(p).write_text(json.dumps(v,indent=2,default=default,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def keys(a):return np.asarray(a,dtype=np.uint64)@(np.uint64(1)<<np.arange(20,dtype=np.uint64))
def inputs(ks):return ((np.asarray(ks,dtype=np.uint64)[...,None]>>np.arange(20,dtype=np.uint64))&1).astype(np.uint8)
def corners(bg,pair):
 a=np.repeat(inputs(bg)[:,None,:],4,1);a[:,:,list(pair)]=BITS;return a
def split(k):return hashlib.sha256(f'5101:{int(k):020b}'.encode()).digest()[0]&1
def lock(run,name,files):
 write(run/name,{'timestamp':now(),'files':{f:sha(run/f) for f in files}})
def verify(run,name):
 r=read(run/name)
 for f,h in r['files'].items():assert sha(run/f)==h,(name,f)
 return r

def metrics(pred,truth):
 e=pred-truth;ae=np.abs(e);al=polynomial(pred)-polynomial(truth)
 d={'n_contexts':len(pred),'corner_mae':ae.mean(),'corner_rmse':np.sqrt((e*e).mean()),'corner_p95':np.quantile(ae,.95),'corner_max':ae.max(),'context_profile_failure_fraction':np.mean((ae.mean(1)>.05)|(ae.max(1)>.10)),'joint_mae':np.abs(al[:,3]).mean(),'remainder_sd':e.std(),'remainder_min':e.min(),'remainder_max':e.max()}
 for t in range(4):d.update({f'alpha{t}_signed_error':al[:,t].mean(),f'alpha{t}_mae':np.abs(al[:,t]).mean(),f'alpha{t}_max':np.abs(al[:,t]).max()})
 d['profile_pass']=bool(d['corner_mae']<=.05 and d['corner_p95']<=.10);d['joint_pass']=bool(d['joint_mae']<=.10)
 return d

@torch.no_grad()
def selector(v,target,beta,budgets=BUDGETS):
 # CUDA Gram algebra; coefficients are exactly 0/1, original beta always present.
 x=v.reshape(-1,v.shape[-1]);y=(target-beta).reshape(-1);g=x.T@x/len(x);c=x.T@y/len(x);diag=g.diag();constant=y.square().mean();out={};traces={}
 for k in budgets:
  s=[];trace=[]
  def objective(s):return float(constant-2*c[s].sum()+g[s][:,s].sum())
  while len(s)<k:
   scores=diag-2*c
   if s:scores=scores+2*g[:,s].sum(1)
   scores[s]=float('inf');s.append(int(scores.argmin()));trace.append({'operation':'add','set':sorted(s),'mse':objective(s)})
  for _ in range(5):
   best=0.;choice=None;sumg=g[:,s].sum(1)
   for old in sorted(s):
    delta=diag-2*c+2*(sumg-g[:,old])-(diag[old]-2*c[old]+2*(sumg[old]-g[old,old]));delta[s]=float('inf');new=int(delta.argmin());value=float(delta[new])
    if value<best-1e-12:best=value;choice=(old,new)
   if choice is None:break
   s[s.index(choice[0])]=choice[1];trace.append({'operation':'swap','old_new':choice,'set':sorted(s),'mse':objective(s)})
  out[str(k)]=sorted(s);traces[str(k)]=trace
 return out,traces

@torch.no_grad()
def evaluate(model,p,q,bg,pair,j):
 device=p.device;a=torch.tensor(corners(bg,pair),device=device,dtype=p.dtype);x=a@p.T
 if hasattr(model,'encoder'):w=model.encoder.weight;b=model.encoder.bias;out=model.decoder
 else:w=model.W1.weight;b=model.W1.bias;out=model.W2
 C=w@p;R=q.T@out.weight;beta=q.T@out.bias;pre=a@C.T+b;h=pre.relu();signed=h*R[j];total=signed.sum(-1)+beta[j]
 direct=out((x@w.T+b).relu())@q
 torch.testing.assert_close(total,direct[:,:,j],atol=1e-9,rtol=1e-8)
 res={'C':C,'R':R,'b':b,'beta':beta,'pre':pre,'h':h,'signed':signed,'total':total,'all_outputs':direct,'q':pre[:,0]}
 aa=polynomial(np.moveaxis(signed.cpu().numpy(),1,-1));np.testing.assert_allclose(aa.sum(1)+np.array([float(beta[j]),0,0,0]),polynomial(total.cpu()),atol=1e-9,rtol=1e-8)
 return res

def cpu(d):return {k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in d.items()}
def save_arrays(p,d):np.savez_compressed(p,**cpu(d))
