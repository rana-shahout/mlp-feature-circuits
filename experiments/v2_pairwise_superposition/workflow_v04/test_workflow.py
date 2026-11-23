import inspect
import numpy as np
import pytest
import torch
from .core import *
from .run import nominate,BASE,DATA,MAN
from .reservation import index

def test_corner_and_polynomial():
 a=corners([1<<3],(1,2));np.testing.assert_array_equal(a[0,:,1:3],BITS);np.testing.assert_array_equal(polynomial([0,1,1,0]),[0,1,1,-2]);x=np.random.default_rng(1).normal(size=(13,4));np.testing.assert_allclose(from_polynomial(polynomial(x)),x,atol=1e-14)
def test_additive_AND_XOR_affine_and_negative():
 a,b=BITS.astype(float).T;h1=np.maximum(a,0);h2=np.maximum(a-b,0);np.testing.assert_array_equal(h1-h2,a*b);assert polynomial(h1)[3]==0
 xor=np.maximum(a-b,0)+np.maximum(b-a,0);np.testing.assert_array_equal(polynomial(xor),[0,1,1,-2]);np.testing.assert_array_equal(polynomial(2+3*a-4*b),[2,3,-4,0])
 # Incomplete additive construction has correct joint but fails baseline/singles.
 wrong=-h2;assert polynomial(wrong)[3]==1;assert not np.array_equal(wrong,a*b);assert not metrics(wrong[None],(a*b)[None])['profile_pass']
 assert not metrics(np.zeros((2,4)),np.tile([0,0,0,1],(2,1)))['joint_pass']
def test_gates_signed_and_nuisance():
 u,v=2.,3.
 for q in [-5.,-4.999,-3.,-2.999,0.]:
  pre=q+u*BITS[:,0]+v*BITS[:,1];assert np.array_equal(pre>0,[0,0,0,1])==(-u-v<q<=-max(u,v))
 # Suppression with negative write can give positive joint.
 assert polynomial(-np.maximum(BITS[:,0].astype(float)-BITS[:,1].astype(float),0))[3]==1
 C=np.arange(120).reshape(6,20)*.003;b=np.arange(6)*.1;a=np.zeros(20);a[3]=1;z=a.copy();z[17]=1;np.testing.assert_allclose((C@z+b)-(C@a+b),C[:,17],atol=1e-14)
 assert C.shape==(6,20) # independent affine gates, no neuron-neuron matrix

def test_split_exclusions_and_lock(tmp_path):
 reserved={11,15};D=set();H=set()
 for pair in [(0,1),(2,3)]:
  for b in range(256):
   if b&((1<<pair[0])|(1<<pair[1])):continue
   ks=keys(corners([b],pair))[0]
   if any(int(k) in reserved for k in ks):continue
   ss=[split(k) for k in ks]
   if len(set(ss))==1:(D if ss[0]==0 else H).update(map(int,ks))
 assert D.isdisjoint(H);assert not (D|H)&reserved
 write(tmp_path/'a.json',{'D':sorted(D)});lock(tmp_path,'lock.json',['a.json']);verify(tmp_path,'lock.json');write(tmp_path/'H.json',{'H':[]});verify(tmp_path,'lock.json');write(tmp_path/'a.json',{'D':[]})
 with pytest.raises(AssertionError):verify(tmp_path,'lock.json')
 with pytest.raises(FileNotFoundError):verify(tmp_path,'missing.json')

def test_discovery_API_and_reservation_isolation(tmp_path):
 assert list(inspect.signature(nominate).parameters)==['outputs','m','p','q','discovery']
 source=inspect.getsource(nominate);assert 'teacher(' not in source and 'targets(' not in source
 # g is an object array, loading it with allow_pickle=False would fail.
 f=tmp_path/'reservation.npz';np.savez(f,a=np.eye(20,dtype=np.uint8),g=np.array([object()],dtype=object));index(f,tmp_path);assert len(np.load(tmp_path/'protected_final_keys.npy'))==20

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required by orchestrator')
def test_cuda_selector_fixed_coefficients():
 torch.manual_seed(9);v=torch.randn(17,4,8,device='cuda',dtype=torch.float64);beta=torch.tensor(.2,device='cuda');truth=v.sum(-1)+beta;sets,tr=selector(v,truth,beta,[1,3,5]);sets2,_=selector(v,truth,beta,[1,3,5]);assert sets==sets2
 before=v.clone()
 for k,ids in sets.items():
  assert len(ids)==int(k);pred=v[:,:,ids].sum(-1)+beta;rest=v[:,:,[i for i in range(8) if i not in ids]].sum(-1);torch.testing.assert_close(pred+rest,truth,atol=1e-12,rtol=1e-12)
  assert sum(r['operation']=='swap' for r in tr[k])<=5
 torch.testing.assert_close(v,before);torch.testing.assert_close(v.sum(-1)+beta,truth)
 # New confirmation data never enters selection and cannot change its result.
 other=torch.randn_like(v);assert other.shape==v.shape;assert selector(v,truth,beta,[1,3,5])[0]==sets

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_actual_checkpoints_cuda_coordinates():
 from ..stage2.core import frozen_teacher
 from src.models import SparseTranscoder
 from src.runtime import load_checkpoint
 v=read(BASE/'runs/stage2-transcoders-20261027/teacher_verification.json');teacher,cp=frozen_teacher(Path(v['checkpoint']),'cuda');assert sha(v['checkpoint'])==v['checkpoint_sha256']
 path=BASE/'checkpoints/stage2-transcoders-20261027/w1024_l0.0001/best.pt';cp=load_checkpoint(path);student=SparseTranscoder(12,1024,12).cuda();student.load_state_dict(cp['model'])
 assert cp['identity']['teacher_sha256']==sha(v['checkpoint']);assert cp['identity']['dataset_sha256']==sha(DATA)
 with np.load(DATA) as data:p0,q0=data['P'],data['Q']
 ref=read(MAN)['partitions']['train']['vector_keys'][:8];a0=inputs(ref)
 p=torch.tensor(p0,device='cuda');q=torch.tensor(q0,device='cuda');a=torch.tensor(a0,device='cuda',dtype=torch.float32);x=a@p.T
 assert p.shape==(12,20) and q.shape==(12,9)
 with torch.no_grad():
  for m,enc,dec,n in [(teacher,teacher.W1,teacher.W2,256),(student,student.encoder,student.decoder,1024)]:
   out32=(dec((x@enc.weight.T+enc.bias).relu())@q).cpu().numpy();m.double();pd,qd,ad=p.double(),q.double(),a.double();C=enc.weight@pd;R=qd.T@dec.weight;beta=qd.T@dec.bias;h=(ad@C.T+enc.bias).relu();identity=h@R.T+beta;direct=dec((ad@pd.T@enc.weight.T+enc.bias).relu())@qd
   torch.testing.assert_close(identity,direct,atol=1e-9,rtol=1e-8);assert C.shape==(n,20) and R.shape==(9,n);np.testing.assert_allclose(out32,direct.cpu(),atol=2e-6,rtol=2e-5)
   # A legacy reference input is used for algebra only; no new bank is ranked.
   bg=int(ref[0])&~((1<<1)|(1<<2));d=evaluate(m,pd,qd,[bg],(1,2),4);d=cpu(d);alpha=polynomial(np.moveaxis(d['signed'],1,-1));np.testing.assert_allclose(alpha.sum(1)+[d['beta'][4],0,0,0],polynomial(d['total']),atol=1e-9,rtol=1e-8)
