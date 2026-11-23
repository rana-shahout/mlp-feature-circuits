import numpy as np
import torch
from ..latent_correspondence_v05.core import tensor,design_fit,solve,predict

def test_reverse_standardization_intercept_and_reload(tmp_path):
 rng=np.random.default_rng(60);z=tensor(rng.normal(size=(90,11))*np.arange(1,12)+4);S=tensor(rng.normal(size=(7,11)));q=tensor(rng.normal(size=7));h=z@S.T+q;f=design_fit(z,h);r=solve(f,0)
 torch.testing.assert_close(r['M'],S,atol=1e-10,rtol=1e-9);torch.testing.assert_close(predict(r,z),f['X']@r['U']+f['mu_z'],atol=1e-10,rtol=1e-9)
 W=tensor(rng.normal(size=(3,7)));b=tensor(rng.normal(size=3));C=W@r['M'];offset=b+W@r['d'];torch.testing.assert_close(predict(r,z)@W.T+b,z@C.T+offset,atol=1e-10,rtol=1e-9)
 np.savez(tmp_path/'m.npz',S=r['M'].cpu(),q=r['d'].cpu(),C=C.cpu(),b=offset.cpu());a=np.load(tmp_path/'m.npz');np.testing.assert_allclose(z.cpu().numpy()@a['C'].T+a['b'],(h@W.T+b).cpu(),atol=1e-10,rtol=1e-9)

def test_rectangular_pseudoinverse_and_rank_deficiency():
 rng=np.random.default_rng(61);M=rng.normal(size=(13,5));d=rng.normal(size=13);h=rng.normal(size=(20,5));S=np.linalg.pinv(M,rcond=1e-10);q=-S@d;np.testing.assert_allclose((h@M.T+d)@S.T+q,h,atol=1e-10,rtol=1e-9);assert np.linalg.matrix_rank(M@S)==5
 M[:,-1]=M[:,0];S=np.linalg.pinv(M,rcond=1e-10);assert np.linalg.matrix_rank(S@M)==4;np.testing.assert_allclose(M@S@M,M,atol=1e-10,rtol=1e-9);assert not np.allclose(S@M,np.eye(5))

def test_all_residual_identities_batch_and_columns():
 rng=np.random.default_rng(62);h=rng.normal(size=(17,5));M=rng.normal(size=(13,5));d=rng.normal(size=13);S=rng.normal(size=(5,13));q=rng.normal(size=5);W=rng.normal(size=(3,5));b=rng.normal(size=3);z=rng.normal(size=(17,13));C=W@S;off=b+W@q;zn=h@M.T+d;ez=z-zn;rN=zn@S.T+q-h;rT=z@S.T+q-h;y=h@W.T+b;yn=zn@C.T+off;yt=z@C.T+off
 pairs=[(rN,h@(S@M-np.eye(5)).T+S@d+q),(rT,rN+ez@S.T),(yn-y,rN@W.T),(yt-y,rT@W.T),(yt-yn,ez@C.T),(yn-y,h@(C@M-W).T+C@d+off-b),((z@S.T+q)@M.T+d-z,rT@M.T-ez)]
 for a,c in pairs:np.testing.assert_allclose(a,c,atol=1e-10,rtol=1e-9)
 for n in range(len(h)):
  hn,zt=h[n],z[n];nn=M@hn+d;rn=S@nn+q-hn;rt=S@zt+q-hn
  np.testing.assert_allclose(rt,rn+S@(zt-nn),atol=1e-10,rtol=1e-9);np.testing.assert_allclose(C@zt+off-(W@hn+b),W@rt,atol=1e-10,rtol=1e-9)

def test_gate_conditioned_native_expression():
 rng=np.random.default_rng(63);a=rng.integers(0,2,size=(4,20));C=rng.normal(size=(8,20));b=rng.normal(size=8);M=rng.normal(size=(11,8));d=rng.normal(size=11)
 for aa in a:
  u=C@aa+b;g=u>0;L=(M*g)@C;o=(M*g)@b+d;np.testing.assert_allclose(L@aa+o,M@np.maximum(u,0)+d,atol=1e-10,rtol=1e-9)
