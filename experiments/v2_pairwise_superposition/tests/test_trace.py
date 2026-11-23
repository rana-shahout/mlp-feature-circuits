import numpy as np
import torch
from src.models import OriginalMLP,SparseTranscoder
from src.runtime import select_device
from src.coordinates import original_coordinates,transcoder_coordinates
from experiments.v2_pairwise_superposition.trace.core import polynomial,from_polynomial,select_pair,corner_inputs,trace


def test_complete_boolean_and_example_retains_additive_unit():
    a=np.array([0,0,1,1]);b=np.array([0,1,0,1]);h1=np.maximum(a,0);h2=np.maximum(a-b,0)
    np.testing.assert_array_equal(h1-h2,[0,0,0,1]);np.testing.assert_array_equal(polynomial(h1),[0,1,0,0]);np.testing.assert_array_equal(polynomial(-h2),[0,-1,0,1]);np.testing.assert_array_equal(polynomial(h1-h2),[0,0,0,1]);assert polynomial(-h2)[1]!=0


def test_polynomial_roundtrip_and_linearity():
    rng=np.random.default_rng(18);v=rng.normal(size=(2,20,4))
    np.testing.assert_allclose(from_polynomial(polynomial(v)),v,atol=1e-14);np.testing.assert_allclose(polynomial(v.sum(1)),polynomial(v).sum(1),atol=1e-14)


def test_metadata_only_choice_sorted_and_certified():
    base=(1<<3)|(1<<6)|(1<<12);other=base|(1<<15);corners=[k|v for k in (base,other) for v in (0,1<<8,1<<7,(1<<7)|(1<<8))]
    manifest={'discovery':{'context_keys':[other,base]},'heldout':{'context_keys':[]}}
    result=select_pair(manifest,{k:'train' for k in corners},set());assert result['background_keys']==[base,other];assert result['flipped_input']==15
    try:select_pair(manifest,{},set())
    except RuntimeError:pass
    else:raise AssertionError('Unknown reserved membership must not silently pass')


def test_11_only_interval_boundary():
    u,v=1.,2.
    for q in (-3.,-2.999,-2.,-1.999,0.):
        gate=(np.array([q,q+v,q+u,q+u+v])>0).tolist();assert (gate==[False,False,False,True])==(-(u+v)<q<=-max(u,v))


def test_cuda_full_trace_and_nuisance_identity():
    device=select_device('cuda');torch.manual_seed(5);p=torch.randn(12,20,device=device,dtype=torch.float64);q=torch.linalg.qr(torch.randn(12,9,device=device,dtype=torch.float64)).Q
    bg=np.zeros((2,20),dtype=np.uint8);bg[:,[3,6,12]]=1;bg[1,15]=1;a=torch.tensor(corner_inputs(bg),device=device,dtype=torch.float64);bt=torch.tensor(bg,device=device,dtype=torch.float64);x=a@p.T
    native=OriginalMLP(12,256,12).to(device).double().eval().requires_grad_(False);pre,h=native.hidden(x.reshape(-1,12));result=trace(original_coordinates(native,p,q),bt,pre.reshape(2,4,256),(native.W2(h)@q).reshape(2,4,9),[224,192,30])
    np.testing.assert_allclose(result['q'][1]-result['q'][0],result['input_coefficients'][:,15],atol=1e-12);assert result['semantic_contributions'].shape==(2,4,9,256)
    tc=SparseTranscoder(12,1024,12).to(device).double().eval().requires_grad_(False);result=trace(transcoder_coordinates(tc,p,q),bt,tc.encoder(x),tc(x)@q,[173,266]);assert result['semantic_contributions'].shape==(2,4,9,1024)
