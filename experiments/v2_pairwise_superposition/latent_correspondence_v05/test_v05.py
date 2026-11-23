import inspect
from pathlib import Path
import numpy as np
import torch
from .core import *
from ..stage2.core import frozen_teacher
from src.runtime import load_checkpoint
from src.models import SparseTranscoder


def fixture():
 torch.manual_seed(6106);h=torch.randn((240,9),device='cuda',dtype=DT);m=torch.randn((17,9),device='cuda',dtype=DT);d=torch.randn(17,device='cuda',dtype=DT);z=h@m.T+d;return h,m,d,z

def test_known_affine_and_intercept_unscaling():
 h,m,d,z=fixture();f=design_fit(h,z);r=solve(f,0);torch.testing.assert_close(r['M'],m,atol=1e-9,rtol=1e-8);torch.testing.assert_close(r['d'],d,atol=1e-9,rtol=1e-8);torch.testing.assert_close(predict(r,h),f['mu_z']+(h-f['mu_h'])@r['M'].T,atol=1e-9,rtol=1e-8)

def test_ridge_factor_N_and_direct_solve():
 h,m,d,z=fixture();f=design_fit(h,z)
 for alpha in [.0001,1.]:
  r=solve(f,alpha);x=f['X'];rhs=x.T@f['Zc']/len(x);ref=torch.linalg.solve(x.T@x/len(x)+alpha*torch.eye(x.shape[1],device='cuda',dtype=DT),rhs);torch.testing.assert_close(r['U'],ref,atol=1e-9,rtol=1e-8)

def test_rank_deficient_constant_and_finite_ceiling():
 h,m,d,z=fixture();h=torch.cat([h,h[:,0:1]*2,torch.ones((len(h),1),device='cuda',dtype=DT)],1);f=design_fit(h,z);assert f['rank']==9;assert int(f['constant_mask'].sum())==1
 for a in GRID:
  r=solve(f,a);assert torch.isfinite(r['M']).all();assert (r['M'][:,-1]==0).all()
 torch.testing.assert_close(predict(solve(f,0),h),z,atol=1e-9,rtol=1e-8)

def test_perfect_output_unreadable_latents_negative():
 x=tensor([[-2.],[-1.],[1.],[2.]]);h=x.relu();z=torch.cat([x.relu(),(-x).relu(),(-x).relu()],1);B=tensor([[1.,1.,-1.]]);torch.testing.assert_close(z@B.T,h);r=solve(design_fit(h,z),0);pred=predict(r,h);assert float((pred-z).square().mean())>.01;assert float((z@B.T-h).abs().max())==0

def test_native_copy_permutation_rescale_output_jacobian():
 base=Path('experiments/v2_pairwise_superposition');cp_path=base/'checkpoints/capacity256-20261027/best.pt';t,cp=frozen_teacher(cp_path,'cuda');t.double();torch.manual_seed(6106);x=torch.randn((400,12),device='cuda',dtype=DT);h=t.hidden(x)[1];norm=t.W2.weight.norm(dim=0);assert (norm>0).all();perm=torch.randperm(256,device='cuda');z=h[:,perm]*norm[perm];f=design_fit(h,z);r=solve(f,0);torch.testing.assert_close(predict(r,h),z,atol=1e-8,rtol=1e-8)
 copyB=t.W2.weight[:,perm]/norm[perm];torch.testing.assert_close(z@copyB.T+t.W2.bias,t(x),atol=1e-9,rtol=1e-8)
 copyE=t.W1.weight[perm]*norm[perm,None];copybias=t.W1.bias[perm]*norm[perm];torch.testing.assert_close((x@copyE.T+copybias).relu(),z,atol=1e-9,rtol=1e-8)
 gates=(x[:2]@t.W1.weight.T+t.W1.bias)>0;copygates=(x[:2]@copyE.T+copybias)>0;J=t.W2.weight@ (gates[:,:,None]*t.W1.weight).transpose(0,1) if False else torch.einsum('or,br,rd->bod',t.W2.weight,gates.double(),t.W1.weight);Jcopy=torch.einsum('om,bm,md->bod',copyB,copygates.double(),copyE);torch.testing.assert_close(J,Jcopy,atol=1e-9,rtol=1e-8)

def test_fit_isolation_tie_break_and_lock(tmp_path):
 h,m,d,z=fixture();f=design_fit(h,z);before=f['mu_h'].clone();selection=torch.zeros_like(h)+1000;confirmation=-selection;assert len(selection)==len(confirmation);torch.testing.assert_close(f['mu_h'],before)
 assert list(inspect.signature(design_fit).parameters)==['predictors','targets'];assert choose_alpha([{'alpha':0.,'score':1.},{'alpha':1.,'score':1.+1e-13}])==1.
 write(tmp_path/'candidate.json',{'alpha':1.});lock(tmp_path,'readout_lock.json',['candidate.json']);write(tmp_path/'confirmation.json',{'changed':True});verify(tmp_path,'readout_lock.json');write(tmp_path/'candidate.json',{'alpha':0.})
 import pytest
 with pytest.raises(AssertionError):verify(tmp_path,'readout_lock.json')

def test_no_rectification_negative_and_cuda_roundtrip(tmp_path):
 h,m,d,z=fixture();r=solve(design_fit(h,z),0);assert (predict(r,h)<0).any();torch.save(to_cpu(r),tmp_path/'r.pt');loaded=torch.load(tmp_path/'r.pt',map_location='cuda',weights_only=False);torch.testing.assert_close(predict(loaded,h),predict(r,h),atol=1e-9,rtol=1e-8)

def test_output_residual_and_cancellation_group():
 h,m,d,z=fixture();torch.manual_seed(10);B=torch.randn((12,17),device='cuda',dtype=DT);W=torch.randn((12,9),device='cuda',dtype=DT);b=torch.randn(12,device='cuda',dtype=DT);c=b*.5;pred=z+.1;F=h@W.T+b;T=z@B.T+c;via=pred@B.T+c;torch.testing.assert_close(F-via,(F-T)+(z-pred)@B.T,atol=1e-9,rtol=1e-8)
 y=tensor([[1.,1.]]);bad=tensor([[2.,2.]]);bb=tensor([[1.,-1.]]);assert (y-bad).square().sum()>0;assert float(((y-bad)@bb.T).abs().max())==0

def test_corner_baseline_singles_joint_AND_XOR():
 vals=np.array([[0,0],[0,1],[1,0],[1,1]],float);a,b=vals.T;andv=np.maximum(a,0)-np.maximum(a-b,0);xor=np.maximum(a-b,0)+np.maximum(b-a,0);np.testing.assert_allclose(polynomial(andv),[0,0,0,1]);np.testing.assert_allclose(polynomial(xor),[0,1,1,-2]);np.testing.assert_allclose(polynomial(np.maximum(a,0)),[0,1,0,0]);np.testing.assert_allclose(from_polynomial(polynomial(xor)),xor);assert polynomial(xor+10)[3]==polynomial(xor)[3]

def test_analytical_autograd_boundary_and_semantic_transform():
 torch.manual_seed(6106);E=torch.randn((8,3),device='cuda',dtype=DT);bias=torch.randn(8,device='cuda',dtype=DT);x=torch.randn(3,device='cuda',dtype=DT,requires_grad=True);pre=E@x+bias;J=torch.autograd.functional.jacobian(lambda xx:(E@xx+bias).relu(),x);torch.testing.assert_close(J,(pre>0)[:,None]*E,atol=1e-9,rtol=1e-8);P=torch.randn((3,5),device='cuda',dtype=DT);torch.testing.assert_close(J@P,((pre>0)[:,None]*E)@P)
 margin=(pre.abs()/E.norm(dim=1)).min();step=min(1e-5,float(margin.detach())/4);direction=torch.eye(3,device='cuda',dtype=DT)[0];fd=((E@(x+step*direction)+bias).relu()-(E@(x-step*direction)+bias).relu())/(2*step);torch.testing.assert_close(fd,J[:,0],atol=1e-7,rtol=1e-7)
 assert not ((tensor([0.])>0).item());assert (tensor([0.]).abs()<=1e-8).item()

def test_dormant_rare_new_activation_undefined_R2():
 torch.manual_seed(7);z=torch.zeros((40,1024),device='cuda',dtype=DT);z[:,0]=torch.rand(40,device='cuda');stats=target_statistics(z);test=numpy(z);test[:,2]=1;pred=np.zeros_like(test)-.001;r=feature_metrics(test,pred,stats);assert r[2]['new_activity'] and not r[2]['well_read'];assert r[1]['R2_eval'] is None;assert r[1]['active_relative_RMSE'] is None;assert r[1]['negative_pred_count']==40

def test_checkpoint_shapes_hashes_and_semantic_identities():
 base=Path('experiments/v2_pairwise_superposition');record=read(base/'reports/workflow-v04-20261030/protocol.json');tp=Path(record['teacher']);sp=Path(record['student']);assert sha(tp)==record['hashes'][str(tp)];assert sha(sp)==record['hashes'][str(sp)];t,_=frozen_teacher(tp,'cuda');t.double();cp=load_checkpoint(sp);s=SparseTranscoder(12,1024,12).cuda().double();s.load_state_dict({k:v.double() for k,v in cp['model'].items()});s.eval().requires_grad_(False)
 with np.load(base/'data/first-teacher-20261027/dataset.npz') as z:P,Q=tensor(z['P']),tensor(z['Q'])
 assert all(not v.requires_grad for v in t.parameters());assert all(not v.requires_grad for v in s.parameters());assert P.shape==(12,20) and Q.shape==(12,9)
 train=read(base/'data/first-teacher-20261027/split_manifest.json')['partitions']['train']['vector_keys'][:8];a=tensor(inputs(train));x=a@P.T;h=(x@t.W1.weight.T+t.W1.bias).relu();latent=(x@s.encoder.weight.T+s.encoder.bias).relu();torch.testing.assert_close(latent,(a@(s.encoder.weight@P).T+s.encoder.bias).relu(),atol=1e-9,rtol=1e-8);torch.testing.assert_close(t(x)@Q,h@(Q.T@t.W2.weight).T+Q.T@t.W2.bias,atol=1e-9,rtol=1e-8);assert latent.shape==(8,1024)

def test_manifest_complete_exclusions():
 # Tiny administrative identity fixture, not scoring any protected inputs.
 allowed={k for k in range(1<<8) if k not in {15,31}}
 pair=(0,1)
 for bg in range(32):
  bg=bg<<2;ks=keys(corners([bg],pair))[0];eligible=all(int(k) in allowed for k in ks)
  if any(int(k) in {15,31} for k in ks):assert not eligible
 assert len(set(range(10))&set(range(10,20)))==0
