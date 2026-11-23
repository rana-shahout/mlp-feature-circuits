import inspect
import numpy as np
import torch
from .core import factor,geometry,cache,score,innovation

def t(x):return torch.tensor(x,dtype=torch.float64)
def test_rank_gain_and_empty():
 W=t([[1,2,3]]);b=t([0]);M=t([[1,0,0],[2,0,0],[0,1,0],[0,0,1]]);d=t([0,0,0,0])
 for ids in [[],[0],[0,1]]:
  F=factor(M,d,W,b,ids);q,nu,gain=geometry(F,M[2]);G=factor(M,d,W,b,ids+[2]);torch.testing.assert_close(G['R'],F['R']-(F['R']@(q/nu))[:,None]*(q/nu)[None,:]);torch.testing.assert_close(F['R'].square().sum()-G['R'].square().sum(),gain.sum())
 assert geometry(factor(M,d,W,b,[0]),M[1])[1]<1e-12
 assert factor(M,d,W,b,[0,2,3])['R'].norm()<1e-12 # algebraic, not discovery

def test_nonnegative_hidden_lost_cancellation():
 W=t([[1,-1]]);D=t([[1,0]]);d=t([0]);F=factor(D,d,W,t([0]),[]);G=factor(D,d,W,t([0]),[0]);h=t([[1,1],[2,2]])
 assert G['R'].norm()<F['R'].norm()
 assert (h@G['R'].T).norm()>(h@F['R'].T).norm()

def test_statistics_and_innovation():
 torch.manual_seed(8);H=torch.rand(100,4,dtype=torch.float64);M=torch.randn(3,4,dtype=torch.float64);d=t([.2,-.3,.1]);Z=H@M.T+d+.01*torch.randn(100,3,dtype=torch.float64);W=torch.randn(2,4,dtype=torch.float64);b=t([.2,.5]);edges=np.array([[i,i+1] for i in range(99)]);c=cache(H,Z,M,d,W,b,edges);c['native_W_norm']=W.norm();F=factor(M,d,W,b,[0]);r=score(F,c,True);assert abs(r['rho_T']**2-r['objective']-r['cross'])<1e-10
 inn=innovation(F,1,M,d,c);assert inn['nu']>0 and inn['assessable'];assert list(inspect.signature(factor).parameters)==['M','d','W','b','ids']
 # Tiny new signal can be overwhelmed by raw latent noise.
 M=t([[1,0],[1,1e-5]]);d=t([.1,.2]);H=torch.rand(100,2,dtype=torch.float64);Z=H@M.T+d;Z[:,1]+=.01;W=t([[1,1]]);c=cache(H,Z,M,d,W,t([0]),edges);c['native_W_norm']=W.norm();i=innovation(factor(M,d,W,t([0]),[0]),1,M,d,c);assert i['value']>.25

def test_status_and_order_policy():
 strict=['ACCEPT','REJECT_THIS_READOUT'];copy=strict.copy();ids=[0,1];tags=['STRICT_ACCEPT' if strict[i]=='ACCEPT' else 'PROVISIONALLY_INCLUDED' for i in ids];assert strict==copy and tags[1]=='PROVISIONALLY_INCLUDED';assert sum(s=='ACCEPT' for s in strict)==1

def test_cancellation_and_mean_centered_split():
 H=t([[2,1],[3,1],[4,2],[5,3]]);M=t([[1,0]]);d=t([.3]);W=t([[1,-1]]);b=t([0]);Z=H@M.T+d-H[:,1,None];edges=np.array([[0,1],[1,2],[2,3]]);c=cache(H,Z,M,d,W,b,edges);c['native_W_norm']=W.norm();F=factor(M,d,W,b,[0]);r=score(F,c,True)
 assert r['rho_T']<1e-7 and r['objective']>0 and r['cross']<0
 EN=H@F['R'].T;mean=(EN.mean(0)**2).sum();vary=((EN-EN.mean(0))**2).mean(0).sum();torch.testing.assert_close(EN.square().mean(0).sum(),mean+vary)
 assert abs(r['mean_energy']-float(mean))<1e-12 and abs(r['varying_energy']-float(vary))<1e-12

def test_raw_ids_units_offsets_and_seed_orders():
 M=t([[3,0],[0,7]]);d=t([.2,-.4]);W=t([[2,5]]);F=factor(M,d,W,t([.1]),[1,0]);torch.testing.assert_close(F['D'],M[[1,0]]);torch.testing.assert_close(F['d'],d[[1,0]]);torch.testing.assert_close(F['C']@F['D']+F['R'],W)
 a=np.random.Generator(np.random.PCG64(8200)).permutation(15);b=np.random.Generator(np.random.PCG64(8200)).permutation(15);np.testing.assert_array_equal(a,b)
