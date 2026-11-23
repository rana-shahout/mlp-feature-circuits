import numpy as np
from .core import *
def test_all_motifs_and_roundtrip():
 for mask in range(16):
  p=np.array([1 if mask&(1<<t) else -1 for t in range(4)],float)[None,:,None];assert gate_codes(p).item()==mask
 r=np.random.default_rng(3).normal(size=(20,4));np.testing.assert_allclose(from_polynomial(polynomial(r)),r,atol=1e-14);polynomial_error_parts(r)
def test_complete_signed_partition_and_energy():
 rng=np.random.default_rng(4);v=rng.normal(size=(7,4,9));omit,r=partition(v,[1,4]);alignment=error_alignment(v[:,:,omit],r);assert abs(alignment.sum()-np.mean(r*r))<1e-12
 a=alpha(v[:,:,omit]);s=signed_stats(a);np.testing.assert_allclose(s['positive_mean']+s['negative_mean'],s['net_mean'],atol=1e-14)
 target=rng.normal(size=(7,4));np.testing.assert_allclose(loss_delta(target,v[:,:,0]),np.mean((target-v[:,:,0])**2)-np.mean(target**2))
def test_additive_vs_interaction_negative_case():
 # Equal interaction coefficients do not mean equal complete functions.
 a=np.array([[0,0,0,1.]]);b=np.array([[0,-1,0,0.]])
 assert polynomial(a)[0,3]==polynomial(b)[0,3];assert np.max(np.abs(a-b))==1
 # Pure positive-write missing-partner hinge gives positive single, negative interaction.
 p=np.array([[0,0,1,0.]])[:,:,None];np.testing.assert_array_equal(alpha(p),[[[0,1,0,-1]]])
 # Negative 11-only write gives negative joint without XOR singles.
 np.testing.assert_array_equal(alpha(-a[:,:,None]),[[[0,0,0,-1]]])
def test_discovery_modal_group_fixed_on_new_contexts():
 d=np.array([[[-1],[-1],[-1],[1]], [[-2],[-2],[-2],[2]]]);modal,sign=modal_groups(d,np.array([-1.]));assert modal.item()==8 and sign.item()==-1
 h=np.ones_like(d);assert (gate_codes(h)!=modal).all();assert modal.item()==8
