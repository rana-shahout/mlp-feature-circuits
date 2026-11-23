import inspect
import numpy as np
import torch
from src.models import OriginalMLP
from src.runtime import select_device
from src.coordinates import original_coordinates
from experiments.v2_pairwise_superposition.stage3.methods import subset_search,gate_parameters,predicted_hidden
from experiments.v2_pairwise_superposition.stage3.run import development,C,corners_np,main


def test_signed_subset_retains_negative_witness():
    x=torch.tensor([[1.5,-.5,0.],[0.,0.,2.]],dtype=torch.float64);y=torch.tensor([1.,0.],dtype=torch.float64)
    result=subset_search(x,y,[1,2]);assert result[1]==[0];assert result[2]==[0,1]


def test_gate_proxy_exact_when_background_has_no_effect():
    c=np.zeros((3,20));c[:,7]=[1,2,-1];c[:,8]=[1,-1,1];bias=np.array([-1.,.2,1.]);bases=np.zeros((5,20));params=gate_parameters(c,bias,bases)
    a=corners_np(bases);actual=np.maximum(a@c.T+bias,0)
    np.testing.assert_allclose(predicted_hidden(a,params,'context2'),actual);np.testing.assert_allclose(predicted_hidden(a,params,'invariant'),actual)


def test_development_disjoint_complete_and_reproducible():
    settings=dict(C['contexts']);settings.update(proposals_unique=300,discovery=10,heldout=10,stress_heldout=4)
    d={'a':np.zeros((0,20),dtype=np.uint8),'train':np.array([],int),'validation':np.array([],int),'test':np.array([],int)}
    banks,meta,manifest=development(d,settings);again,_,_=development(d,settings)
    for key in banks:np.testing.assert_array_equal(banks[key],again[key])
    for key in ('discovery','heldout'):
        density=corners_np(banks[key]).sum(-1);assert ((density>=3)&(density<=6)).all()
    a=set(manifest['discovery']['context_keys']);b=set(manifest['heldout']['context_keys']);assert not a&b
    source=inspect.getsource(main);assert source.index("'discovery_frozen.json'")<source.index("for split in ('heldout','stress')")


def test_actual_cuda_zero_clamp_and_no_parameter_edit():
    device=select_device('cuda');m=OriginalMLP(12,256,12).to(device).double().eval().requires_grad_(False);p=torch.randn(12,20,device=device,dtype=torch.float64);q=torch.linalg.qr(torch.randn(12,9,device=device,dtype=torch.float64)).Q
    a=torch.tensor(corners_np(np.zeros((3,20))),device=device,dtype=torch.float64);_,h=m.hidden((a@p.T).reshape(-1,12));h=h.reshape(3,4,256);base=m.W2(h)@q;d=original_coordinates(m,p,q)['D2_eff'];selected=[2,11,19];before={k:v.clone() for k,v in m.state_dict().items()};signs=torch.tensor([1.,-1.,-1.,1.],device=device)
    deltas=[]
    for replacement in (torch.zeros(256,device=device,dtype=torch.float64),h[:,0].mean(0)):
        altered=h.clone();altered[:,:,selected]=replacement[selected];effect=m.W2(altered)@q-base;expected=(replacement[selected]-h[:,:,selected])@d[:,selected].T
        torch.testing.assert_close(effect,expected,atol=1e-9,rtol=1e-8);deltas.append((effect*signs[None,:,None]).sum(1))
    torch.testing.assert_close(*deltas,atol=1e-9,rtol=1e-8);assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
