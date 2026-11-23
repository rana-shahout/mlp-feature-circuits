import numpy as np
import pytest
import torch

from src.coordinates import transcoder_coordinates
from src.data import embeddings
from src.interactions import finite_interaction
from src.models import OriginalMLP, SparseTranscoder
from src.train_transcoder import imitation_loss, validate_transcoder_config
from src.transcoder_evaluation import contribution_statistics, fidelity_metrics
from src.transcoder_interactions import tensor_statistics


def test_full_teacher_output_and_l1_reductions_are_unambiguous():
    student=torch.zeros(2,12)
    teacher=torch.zeros_like(student);teacher[:,11]=1
    z=torch.ones(2,3)
    loss,mse,l1=imitation_loss(student,teacher,z,.1)
    assert mse.item()==pytest.approx(1/12)
    assert l1.item()==3
    assert loss.item()==pytest.approx(1/12+.3)
    assert imitation_loss(student,teacher,torch.ones(2,6),.1)[2].item()==6


def test_teacher_errors_are_preserved_in_fidelity_not_hidden_by_task_metrics():
    _,q=embeddings(16,12,1,2)
    g=np.zeros((4,8),dtype=np.float32)
    erroneous=np.ones((4,8),dtype=np.float32)@q.T
    result=fidelity_metrics(erroneous,erroneous,g,q,.5)
    assert result['fidelity']['exact_vector_agreement']==1
    assert result['fidelity']['full_output_mse']==0
    assert result['student_task']['exact_vector_accuracy']==0
    assert result['fidelity']['per_output'][4]['teacher_errors_preserved_fraction']==1


def test_effective_units_use_absolute_mass_and_are_scale_invariant():
    c=np.array([[2.,-1.5]])
    stats=contribution_statistics(c)
    assert stats['value'][0]==.5
    assert stats['positive_sum'][0]==2 and stats['negative_sum'][0]==-1.5
    assert stats['effective_units'][0]==pytest.approx(3.5**2/(4+2.25))
    np.testing.assert_allclose(stats['effective_units'],contribution_statistics(7*c)['effective_units'])
    assert contribution_statistics(np.zeros((1,4)))['effective_units'][0]==0


def test_tensor_statistics_preserve_full_signed_sums_and_match_reference():
    c=np.array([[[[2.,-1.5,0.,.1]]],[[[0.,0.,0.,0.]]]])
    expected=contribution_statistics(c)
    actual=tensor_statistics(torch.tensor(c))
    for key,value in expected.items():
        np.testing.assert_allclose(actual[key].numpy(),value,atol=1e-12)


def test_config_cannot_silently_change_the_scientific_objective_or_unseal_holdout():
    from copy import deepcopy
    import yaml
    from pathlib import Path
    c=yaml.safe_load(Path('configs/transcoder_v1.yaml').read_text())
    validate_transcoder_config(c)
    bad=deepcopy(c);bad['lambdas']=[0,-.1]
    with pytest.raises(ValueError,match='nonnegative'):validate_transcoder_config(bad)
    bad=deepcopy(c);bad['training']['l1_reduction']='mean_over_units'
    with pytest.raises(ValueError,match='protocol'):validate_transcoder_config(bad)
    bad=deepcopy(c);bad['final_population']['score_in_initial_run']=True
    with pytest.raises(ValueError,match='reserve'):validate_transcoder_config(bad)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.cuda)])
def test_student_coordinates_pair_and_triple_exact_sums(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    model=SparseTranscoder(16,13,12).to(device=device,dtype=torch.float64)
    p,q=[torch.tensor(v,dtype=torch.float64,device=device) for v in embeddings(16,12,1,2)]
    coordinates=transcoder_coordinates(model,p,q)
    contexts=torch.zeros(3,32,dtype=torch.float64,device=device);contexts[:,18]=1
    for ids in [(6,7),(8,9,10),(4,16,17)]:
        result=finite_interaction(contexts,ids,coordinates)
        corners=result['corners']
        direct=(model((corners@p.T).reshape(-1,16))@q).reshape(3,2**len(ids),8)
        difference=(direct*result['signs'][None,:,None]).sum(1)
        torch.testing.assert_close(result['contributions'].sum(-1),difference,atol=1e-10,rtol=1e-8)


def test_only_student_changes_after_imitation_optimization():
    teacher=OriginalMLP(3,5,2).eval().requires_grad_(False)
    before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    student=SparseTranscoder(3,7,2)
    x=torch.randn(8,3)
    with torch.no_grad():y=teacher(x)
    optimizer=torch.optim.Adam(student.parameters(),lr=.001)
    z=student.latents(x)
    imitation_loss(student.decoder(z),y,z,.01)[0].backward()
    optimizer.step();student.normalize_decoder()
    for key,value in teacher.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    assert all(p.grad is None for p in teacher.parameters())
    torch.testing.assert_close(student.decoder.weight.norm(dim=0),torch.ones(7))
