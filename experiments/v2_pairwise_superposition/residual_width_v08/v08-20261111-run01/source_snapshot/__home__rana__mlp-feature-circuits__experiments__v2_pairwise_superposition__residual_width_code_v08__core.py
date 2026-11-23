"""v0.8 arithmetic; no reverse fitting or output-target solver."""
import time
from pathlib import Path
import numpy as np
import torch
from ..native_features_v07.run import arr,tensor,write,read,sha,bits,BASE,OLD,HIST,LABELS,metrics,csvwrite
V07=BASE/'native_matrix_runs/v07-20261110T-run01'

def factor(M,d,W,b,ids):
 D=M[ids];dd=d[ids];U,s,V=torch.linalg.svd(D,full_matrices=False,driver='gesvd') if D.is_cuda else torch.linalg.svd(D,full_matrices=False);cut=max(1e-12,1e-10*float(s[0])) if len(s) else 1e-12;k=s>cut
 J=(V[k].T/s[k])@U[:,k].T;C=W@J;R=W-C@D;off=b-C@dd
 return dict(ids=ids.copy(),D=D,d=dd,J=J,C=C,R=R,bF=off,s=s,cutoff=cut,rank=int(k.sum()),condition=float(s[k][0]/s[k][-1]) if k.any() else 0.,J_norm=float(1/s[k][-1]) if k.any() else 0.,C_norm=float(C.norm()))

def cache(H,Z,M,d,W,b,edges):
 E=H@M.T+d-Z;Y=H@W.T+b;n=len(H);ix=torch.as_tensor(edges,device=H.device);DH=H[ix[:,1]]-H[ix[:,0]];DE=E[ix[:,1]]-E[ix[:,0]];DY=Y[ix[:,1]]-Y[ix[:,0]]
 return dict(H=H,Z=Z,E=E,Y=Y,edges=ix,GH=H.T@H/n,GE=E.T@E/n,GHE=H.T@E/n,meanH=H.mean(0),den=Y.square().mean(0),edgeGH=DH.T@DH/len(DH),edgeGE=DE.T@DE/len(DE),edgeGHE=DH.T@DE/len(DE),edgeDen=DY.square().mean(0),DH=DH,DE=DE)

def quad(A,G):
 x=(A@G*A).sum(1);tol=1e-10*max(1.,float(A.square().sum()*G.norm()));assert float(x.min())>=-tol,('negative_energy',x,tol)
 return x.clamp_min(0)
def score(F,c,direct=False):
 ids=F['ids'];C=F['C'];R=F['R'];N=quad(R,c['GH']);L=quad(C,c['GE'][ids][:,ids]);cross=2*(R@c['GHE'][:,ids]*C).sum(1);T=N+L+cross;assert float(T.min())>=-1e-9;T=T.clamp_min(0);den=c['den'];total=den.sum();mean=(R@c['meanH']).square();vary=N-mean;assert float(vary.min())>=-1e-9
 result=dict(rho_W=float(R.norm()/W_norm(c)),rho_N=float((N.sum()/total).sqrt()),rho_L=float((L.sum()/total).sqrt()),rho_T=float((T.sum()/total).sqrt()),objective=float((N.sum()+L.sum())/total),cross=float(cross.sum()/total),mean_energy=float(mean.sum()),varying_energy=float(vary.sum()),rank=F['rank'],condition=F['condition'],J_norm=F['J_norm'],C_norm=F['C_norm'],per_output_N=arr((N/den).sqrt()),per_output_L=arr((L/den).sqrt()),per_output_T=arr((T/den).sqrt()),score_norms=arr((den*len(c['H'])).sqrt()))
 EN=quad(R,c['edgeGH']);EL=quad(C,c['edgeGE'][ids][:,ids]);EC=2*(R@c['edgeGHE'][:,ids]*C).sum(1);ET=(EN+EL+EC).clamp_min(0);ed=c['edgeDen'];result.update(edge_rho_N=float((EN.sum()/ed.sum()).sqrt()),edge_rho_L=float((EL.sum()/ed.sum()).sqrt()),edge_rho_T=float((ET.sum()/ed.sum()).sqrt()),edge_per_output_N=arr((EN/ed).sqrt()),edge_per_output_L=arr((EL/ed).sqrt()),edge_per_output_T=arr((ET/ed).sqrt()))
 if direct:
  en=c['H']@R.T;el=c['E'][:,ids]@C.T;et=en+el
  for target,actual in [(N,en.square().mean(0)),(L,el.square().mean(0)),(T,et.square().mean(0))]:torch.testing.assert_close(target,actual,atol=1e-9,rtol=1e-8)
  torch.testing.assert_close(c['Y'],(c['H']@F['D'].T+F['d'])@C.T+F['bF']+en,atol=1e-8,rtol=1e-8)
  result.update(max_abs_N=float(en.abs().max()),max_abs_L=float(el.abs().max()),max_abs_T=float(et.abs().max()),per_output_max_N=arr(en.abs().amax(0)),per_output_max_L=arr(el.abs().amax(0)),per_output_max_T=arr(et.abs().amax(0)))
 return result

def W_norm(c):return c['native_W_norm']

def geometry(F,v):
 q=v-v@F['J']@F['D'];nu=q.norm();u=q/nu if nu>0 else q;gain=(F['R']@u).square();return q,nu,gain

def innovation(F,m,M,d,c):
 q,nu,gain=geometry(F,M[m]);gamma=M[m]@F['J'];delta=d[m]-gamma@F['d']
 if nu<=max(1e-12,1e-10*float(M[m].norm())):return dict(value=None,edge=None,support=0,nu=float(nu),relative_novelty=float(nu/M[m].norm()),gain_W=float(gain.sum()),gain_by_output=arr(gain),assessable=False)
 en=c['H']@q/nu;diff=(c['E'][:,m]-c['E'][:,F['ids']]@gamma)/nu;de=c['DE'][:,m]-c['DE'][:,F['ids']]@gamma;de=de/nu;dn=c['DH']@q/nu;tau=max(1e-8,1e-3*float(dn.square().mean().sqrt()));support=int((abs(dn)>tau).sum());den=(en-en.mean()).square().sum();eden=dn.square().sum();ok=float(en.std(unbiased=False))>1e-8 and support>=32 and float(den)>1e-24 and float(eden)>1e-24
 # Explicit sign and intercept identity on every V input.
 et=(c['Z'][:,m]-c['Z'][:,F['ids']]@gamma-delta)/nu;torch.testing.assert_close(en-et,diff,atol=1e-8,rtol=1e-8)
 return dict(value=float(diff.norm()/den.sqrt()) if ok else None,edge=float(de.norm()/eden.sqrt()) if ok else None,support=support,nu=float(nu),relative_novelty=float(nu/M[m].norm()),gain_W=float(gain.sum()),gain_by_output=arr(gain),assessable=ok)

def savefactor(path,F,native,statuses):
 path.mkdir(parents=True,exist_ok=False)
 for k,name in [('D','D2'),('d','d2'),('J','J'),('C','C2'),('R','R2'),('bF','bF'),('s','singular_values')]:np.save(path/(name+'.npy'),arr(F[k]))
 write(path/'metadata.json',dict(ids=F['ids'],strict_statuses=[statuses[m] for m in F['ids']],inclusion_tags=['STRICT_ACCEPT' if statuses[m]=='ACCEPT' else 'PROVISIONALLY_INCLUDED' for m in F['ids']],rank=F['rank'],condition=F['condition'],cutoff=F['cutoff'],J_norm=F['J_norm'],native_matrices=str(native.resolve()),native_sha256=sha(native),full_rank_is_algebraic=F['rank']==256))

@torch.no_grad()
def infer_canonical(ks,W1,b1,K,e):
 H=[];Z=[]
 for start in range(0,len(ks),512):
  a=tensor(bits(ks[start:start+512]));H.append((a@W1.T+b1).relu());Z.append((a@K.T+e).relu())
 return torch.cat(H),torch.cat(Z)

def rebuild(ks):
 lookup={k:i for i,k in enumerate(ks)};edges=[(lookup[k],lookup[k|1<<j]) for k in ks for j in range(20) if not k&(1<<j) and k|1<<j in lookup];contexts={}
 for i,j in [(7,8),(11,12),(0,1)]:contexts[f'{i},{j}']=[[k,k|1<<j,k|1<<i,k|1<<i|1<<j] for k in ks if not k&((1<<i)|(1<<j)) and all(t in lookup for t in [k|1<<j,k|1<<i,k|1<<i|1<<j])]
 return np.array(edges,dtype=np.int64).reshape(-1,2),contexts
