import inspect,copy
import numpy as np
import torch
import yaml
import pytest
from experiments.v2_pairwise_superposition import capacity,capacity256
from experiments.v2_pairwise_superposition.capacity256 import BASE,validate_control,signed_statistics
from src.models import OriginalMLP
from src.runtime import select_device,seed_everything,save_checkpoint,load_checkpoint
from src.coordinates import original_coordinates
from src.interactions import finite_interaction


def test_only_architecture_changes_in_config_and_training():
    prior=yaml.safe_load((BASE/'runs/capacity128-20261027/config.yaml').read_text());c=yaml.safe_load((BASE/'configs/capacity256.yaml').read_text());validate_control(c,prior)
    bad=copy.deepcopy(c);bad['training']['max_epochs']=2100
    with pytest.raises(ValueError):validate_control(bad,prior)
    assert inspect.getsource(capacity.train).replace('OriginalMLP(12,128,12)','OriginalMLP(12,256,12)').strip()==inspect.getsource(capacity256.train).strip()


def test_normalized_concentration_and_duplicate_units():
    v=np.array([[2.,-1.,0.,0.],[0.,0.,0.,0.]])
    r=signed_statistics(v);padded=signed_statistics(np.pad(v,((0,0),(0,4))));split=signed_statistics(np.repeat(v/2,2,axis=1))
    np.testing.assert_allclose(r['net'],[1,0]);np.testing.assert_allclose(r['effective_units'],[1.8,0]);np.testing.assert_allclose(r['effective_fraction_of_width'],[.45,0]);np.testing.assert_allclose(r['cancellation'],[2/3,0])
    np.testing.assert_allclose(padded['effective_units'],r['effective_units']);np.testing.assert_allclose(padded['effective_fraction_of_width'],r['effective_fraction_of_width']/2)
    np.testing.assert_allclose(split['effective_units'],r['effective_units']*2);np.testing.assert_allclose(split['effective_fraction_of_width'],r['effective_fraction_of_width']);np.testing.assert_allclose(split['cancellation'],r['cancellation'])


def test_cuda_width256_gradients_roundtrip_and_signed_sums(tmp_path):
    device=select_device('cuda');seed_everything(3105,True);m=OriginalMLP(12,256,12).to(device)
    assert m.W1.weight.shape==(256,12) and m.W2.weight.shape==(12,256)
    x=torch.randn(8,12,device=device);y=m(x);torch.nn.functional.mse_loss(y,torch.zeros_like(y)).backward();assert all(torch.isfinite(v.grad).all() for v in m.parameters())
    save_checkpoint(tmp_path/'cp.pt',{'model':m.state_dict()});clone=OriginalMLP(12,256,12).to(device);clone.load_state_dict(load_checkpoint(tmp_path/'cp.pt')['model']);torch.testing.assert_close(y,clone(x))
    m=m.double();p=torch.randn(12,20,device=device,dtype=torch.float64);q=torch.linalg.qr(torch.randn(12,9,device=device,dtype=torch.float64)).Q
    for indices in ((0,1),(0,1,2)):
        r=finite_interaction(torch.zeros(2,20,device=device,dtype=torch.float64),indices,original_coordinates(m,p,q));a=r['corners'];signs=r['signs'];assert r['contributions'].shape==(2,9,256)
        direct=(m((a@p.T).reshape(-1,12)).reshape(2,len(signs),12)@q*signs[None,:,None]).sum(1);torch.testing.assert_close(r['value'],direct,atol=1e-9,rtol=1e-8)
