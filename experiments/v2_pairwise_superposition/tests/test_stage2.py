import numpy as np
import torch
from src.models import SparseTranscoder
from src.runtime import seed_everything,select_device
from src.coordinates import transcoder_coordinates
from src.interactions import finite_interaction
from experiments.v2_pairwise_superposition.stage2.core import loss_terms,fidelity,pareto
from experiments.v2_pairwise_superposition.stage2.run import validate,C


def test_loss_reduction_and_teacher_detach():
    p=torch.ones(4,12,requires_grad=True);teacher=torch.zeros(4,12,requires_grad=True);z=torch.ones(4,512,requires_grad=True)
    loss,rec,l1=loss_terms(p,teacher,z,.001)
    torch.testing.assert_close(rec,torch.tensor(12.));torch.testing.assert_close(l1,torch.tensor(512.));torch.testing.assert_close(loss,torch.tensor(12.512))
    loss.backward();assert teacher.grad is None
    assert torch.isfinite(p.grad).all() and torch.isfinite(z.grad).all()


def test_fidelity_distinguishes_labels_and_variance():
    q=np.eye(12)[:,:9];teacher=np.zeros((4,12));teacher[2:,:9]=1;g=np.ones((4,9))
    result=fidelity(teacher,teacher,g,q);assert result['teacher_fidelity']['exact_vector_agreement']==1
    assert result['student_task']['exact_vector_accuracy']==.5
    assert result['teacher_fidelity']['teacher_incorrect']['n']==2
    shifted=fidelity(teacher+1,teacher,g,q)['teacher_fidelity'];assert shifted['full_output_mse']==1
    assert shifted['variance_normalized_mse']==1/((teacher-teacher.mean(0))**2).mean()


def test_predefined_sweep_and_pareto():
    validate(C);assert len(C['widths'])*len(C['lambdas'])==6
    rows=[{'name':'a','mse':1.,'l0':2},{'name':'b','mse':2.,'l0':1},{'name':'c','mse':3.,'l0':3}]
    assert set(pareto(rows))=={'a','b'}


def test_cuda_unit_norm_and_paired_initialization():
    device=select_device('cuda')
    for width in (512,1024):
        seed_everything(4105,True);m=SparseTranscoder(12,width,12).to(device)
        seed_everything(4105,True);same=SparseTranscoder(12,width,12).to(device)
        assert all(torch.equal(v,same.state_dict()[k]) for k,v in m.state_dict().items())
        opt=torch.optim.Adam(m.parameters(),lr=.001);x=torch.randn(8,12,device=device);z=m.latents(x)
        loss,_,_=loss_terms(m.decoder(z),torch.randn(8,12,device=device),z,.001);loss.backward();opt.step();m.normalize_decoder()
        torch.testing.assert_close(m.decoder.weight.norm(dim=0),torch.ones(width,device=device),atol=3e-7,rtol=3e-7)


def test_cuda_exact_transcoder_coordinates_pairs_and_triples():
    device=select_device('cuda');seed_everything(4105,True)
    for width in (512,1024):
        m=SparseTranscoder(12,width,12).to(device).double();p=torch.randn(12,20,device=device,dtype=torch.float64);q=torch.linalg.qr(torch.randn(12,9,device=device,dtype=torch.float64)).Q
        coord=transcoder_coordinates(m,p,q)
        assert coord['C1'].shape==(width,20) and coord['D2_eff'].shape==(9,width)
        ctx=torch.zeros(3,20,device=device,dtype=torch.float64)
        for indices in ((7,8),(11,12),(0,1),(0,1,2)):
            r=finite_interaction(ctx,indices,coord);a=r['corners'];signs=r['signs']
            direct=(m((a@p.T).reshape(-1,12)).reshape(3,len(signs),12)@q*signs[None,:,None]).sum(1)
            torch.testing.assert_close(r['value'],direct,atol=1e-9,rtol=1e-8)
            torch.testing.assert_close(r['positive_sum']+r['negative_sum'],direct,atol=1e-9,rtol=1e-8)
