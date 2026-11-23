import copy
import numpy as np
import torch
import yaml
import pytest
from experiments.v2_pairwise_superposition.capacity import BASE,validate_control,nearest_labels
from experiments.v2_pairwise_superposition.capacity_report import concentration
from src.models import OriginalMLP
from src.runtime import select_device,seed_everything,save_checkpoint,load_checkpoint
from src.coordinates import original_coordinates
from src.interactions import finite_interaction


def test_only_width_override():
    baseline=yaml.safe_load((BASE/'runs/first-teacher-20261027/config.yaml').read_text());c=yaml.safe_load((BASE/'configs/capacity128.yaml').read_text());validate_control(c,baseline)
    bad=copy.deepcopy(c);bad['training']['learning_rate']=.002
    with pytest.raises(ValueError):validate_control(bad,baseline)


def test_exact_nearest_label_distances_cuda():
    device=select_device('cuda');x=torch.tensor([[0.,0.],[1.,0.],[3.,0.],[5.,0.]],device=device,dtype=torch.float64)
    g=torch.tensor([[0]*9,[0]*9,[1]*9,[1]*9],device=device)
    r=nearest_labels(x,g,x,g,torch.arange(4,device=device),batch=2)
    np.testing.assert_allclose(r['opposite_distance'][:,0],[3,2,2,4]);np.testing.assert_allclose(r['same_distance'][:,0],[1,1,2,2])
    direct=((x[:,None]-x[None,:])**2).sum(-1).sqrt();torch.testing.assert_close(torch.cdist(x,x),direct)
    y=torch.tensor([[2.,0.]],device=device,dtype=torch.float64);t=torch.ones((1,9),device=device)
    v=nearest_labels(y,t,x,g,batch=1);assert v['opposite_distance'][0,0]==1


def test_width128_cuda_roundtrip_and_interactions(tmp_path):
    device=select_device('cuda');seed_everything(3105,True);m=OriginalMLP(12,128,12).to(device)
    assert m.W1.weight.shape==(128,12) and m.W2.weight.shape==(12,128)
    x=torch.randn(7,12,device=device);y=m(x);loss=y.square().mean();loss.backward();assert all(torch.isfinite(v.grad).all() for v in m.parameters())
    path=tmp_path/'roundtrip.pt';save_checkpoint(path,{'model':m.state_dict()});clone=OriginalMLP(12,128,12).to(device);clone.load_state_dict(load_checkpoint(path)['model']);torch.testing.assert_close(y,clone(x))
    m=m.double();p=torch.randn(12,20,device=device,dtype=torch.float64);q=torch.linalg.qr(torch.randn(12,9,device=device,dtype=torch.float64)).Q
    ctx=torch.zeros(2,20,device=device,dtype=torch.float64)
    for indices in ((0,1),(0,1,2)):
        r=finite_interaction(ctx,indices,original_coordinates(m,p,q));a=r['corners'];signs=r['signs']
        direct=(m((a@p.T).reshape(-1,12)).reshape(2,len(signs),12)@q*signs[None,:,None]).sum(1)
        assert r['contributions'].shape==(2,9,128);torch.testing.assert_close(r['value'],direct,atol=1e-9,rtol=1e-8)


def test_concentration_preserves_cancellation():
    r=concentration(np.array([[2.,-1.,0.],[0.,0.,0.]]));np.testing.assert_allclose(r['net'],[1,0]);np.testing.assert_allclose(r['effective_units'],[1.8,0]);np.testing.assert_allclose(r['cancellation'],[2/3,0]);np.testing.assert_allclose(r['top1_signed_residual'],[1,0])
