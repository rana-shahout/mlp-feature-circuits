import numpy as np
from ..workflow_v04.core import polynomial,from_polynomial
LABELS={0:'inactive',8:'11-only',1:'00-only',4:'i-only / missing-k',2:'k-only / missing-i',12:'i-supported',10:'k-supported',3:'i-off',5:'k-off',15:'always-on affine',14:'any-input-on hinge',7:'11-off'}
def gate_codes(pre):return (np.asarray(pre)>0).astype(int).transpose(0,2,1)@(1<<np.arange(4))
def alpha(signed):return polynomial(np.moveaxis(signed,1,-1))
def modal_groups(pre,readout):
 codes=gate_codes(pre);modes=np.array([np.bincount(codes[:,u],minlength=16).argmax() for u in range(codes.shape[1])]);return modes,np.sign(readout).astype(int)
def polynomial_error_parts(r):
 al=polynomial(r);add=from_polynomial(al*np.array([1,1,1,0]));joint=from_polynomial(al*np.array([0,0,0,1]));np.testing.assert_allclose(add+joint,r,atol=1e-10,rtol=1e-8)
 # Nonorthogonal Boolean basis: additive/joint energy cannot be added without cross term.
 energies={'additive_MSE':np.mean(add**2),'joint_MSE':np.mean(joint**2),'twice_cross':2*np.mean(add*joint),'total_MSE':np.mean(r**2)};assert abs(energies['additive_MSE']+energies['joint_MSE']+energies['twice_cross']-energies['total_MSE'])<1e-10
 return energies
def partition(v,ids):
 omitted=np.array([u for u in range(v.shape[-1]) if u not in ids]);rest=v[:,:,omitted].sum(-1);selected=v[:,:,ids].sum(-1);np.testing.assert_allclose(selected+rest,v.sum(-1),atol=1e-10,rtol=1e-8);return omitted,rest
def signed_stats(al):
 pos=np.maximum(al,0).sum(1);neg=np.minimum(al,0).sum(1);net=al.sum(1);mass=pos-neg;cancel=np.divide(mass-np.abs(net),mass,out=np.zeros_like(mass),where=mass>1e-12)
 return {'positive_mean':pos.mean(0),'negative_mean':neg.mean(0),'net_mean':net.mean(0),'cancellation_fraction_mean':cancel.mean(0),'absolute_mass_mean':mass.mean(0)}
def error_alignment(v,r):
 # Signed accounting of error energy, not a necessity/causality score.
 return np.mean(v*r[:,:,None],axis=(0,1))
def loss_delta(residual,v):return float(np.mean(v*v-2*residual*v))
