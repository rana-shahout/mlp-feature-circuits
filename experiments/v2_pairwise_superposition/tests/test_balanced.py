import numpy as np
import pytest
import torch
from experiments.v2_pairwise_superposition.balanced import objective,rankings


def test_full_population_conditional_equivalence():
    torch.manual_seed(1)
    q=torch.linalg.qr(torch.randn(12,9,dtype=torch.float64)).Q
    g=(torch.rand(100,9)>.8).double();y=torch.randn(100,12,dtype=torch.float64,requires_grad=True);p=g.mean(0)
    s=y@q
    expected=torch.stack([.5*(s[g[:,j]==1,j]-1).square().mean()+.5*s[g[:,j]==0,j].square().mean() for j in range(9)]).mean()
    expected=expected+(y-s@q.T).square().sum(1).mean()/12
    torch.testing.assert_close(objective(y,g,q,p),expected)
    objective(y,g,q,p).backward();assert torch.isfinite(y.grad).all()
    assert torch.isfinite(objective(y[:1],torch.zeros_like(g[:1]),q,p))
    assert torch.isfinite(objective(y[:1],torch.ones_like(g[:1]),q,p))


def test_orthogonal_norm_reduction():
    q=torch.eye(12,dtype=torch.float64)[:,:9];g=torch.zeros(2,9,dtype=torch.float64);y=torch.zeros(2,12,dtype=torch.float64);y[:,9:]=2
    torch.testing.assert_close(objective(y,g,q,torch.full((9,),.2,dtype=torch.float64)),torch.tensor(1.,dtype=torch.float64))


def test_rankings_ties():
    r=rankings(np.ones(4),np.array([1,0,1,0]));assert r['roc_auc']==.5;assert r['average_precision']==.5
    r=rankings(np.array([.9,.8,.2,.1]),np.array([1,1,0,0]));assert r['roc_auc']==1;assert r['average_precision']==1


def test_cuda_smoke():
    from src.runtime import select_device
    from src.models import OriginalMLP
    device=select_device('cuda');m=OriginalMLP(12,64,12).to(device);q=torch.linalg.qr(torch.randn(12,9,device=device)).Q
    loss=objective(m(torch.randn(8,12,device=device)),torch.zeros(8,9,device=device),q,torch.full((9,),.2,device=device));loss.backward()
    assert torch.isfinite(loss)


def test_cuda_batched_corner_forward_identity():
    from src.runtime import select_device
    from src.models import OriginalMLP
    from src.coordinates import original_coordinates
    from src.interactions import finite_interaction
    device=select_device('cuda');m=OriginalMLP(12,64,12).to(device).double()
    p=torch.randn(12,20,device=device,dtype=torch.float64);q=torch.linalg.qr(torch.randn(12,9,device=device,dtype=torch.float64)).Q
    contexts=torch.zeros(3,20,device=device,dtype=torch.float64)
    for indices in ((0,1),(0,1,2)):
        r=finite_interaction(contexts,indices,original_coordinates(m,p,q));corners=r['corners'];signs=r['signs']
        with torch.no_grad():direct=(m((corners@p.T).reshape(-1,12)).reshape(len(contexts),len(signs),12)@q*signs[None,:,None]).sum(1)
        torch.testing.assert_close(direct,r['value'],atol=1e-9,rtol=1e-8)
