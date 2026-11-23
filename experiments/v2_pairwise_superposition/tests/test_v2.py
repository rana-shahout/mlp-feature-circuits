from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.models import OriginalMLP
from src.coordinates import original_coordinates,semantic_forward
from src.interactions import finite_interaction
from src.runtime import save_checkpoint,load_checkpoint,select_device
from experiments.v2_pairwise_superposition.data import targets,RULES,make_embeddings,generate,stratified_split,keys,clause_counts
from experiments.v2_pairwise_superposition.analysis import metrics,probe_bank,native_diagnostics
from experiments.v2_pairwise_superposition.run import validate,BASE


@pytest.fixture
def config():return yaml.safe_load((BASE/'configs/teacher_v2.yaml').read_text())


def reference(row):
    b=list(map(bool,row))
    first=[b[0] and b[1],b[0] and b[2],b[3] and b[4],b[5] and b[6],b[7] and b[8],b[9] and b[10]]
    return first+[b[11]!=b[12],b[13] and not b[14],any(first[i] for i in (0,2,3,4,5))]


def test_all_nine_truth_tables_exhaustive_semantic_domain_and_nuisance_invariance():
    # 2^15 relevant states, not 2^20 enumeration for a training prerequisite.
    a=np.zeros((2**15,20),dtype=np.uint8)
    a[:,:15]=np.asarray(list(product((0,1),repeat=15)),dtype=np.uint8)
    expected=np.asarray([reference(row) for row in a],dtype=np.float32)
    np.testing.assert_array_equal(targets(a),expected)
    a[:,15:]=1
    np.testing.assert_array_equal(targets(a),expected)
    isolated=np.zeros(20,dtype=np.uint8);isolated[[0,2]]=1
    assert targets(isolated)[1]==1 and targets(isolated)[8]==0
    with pytest.raises(ValueError):targets(np.zeros((1,32)))


def test_v2_embeddings_are_not_v1_defaults():
    p,q=make_embeddings(3101,3102)
    assert p.shape==(12,20) and q.shape==(12,9) and np.linalg.matrix_rank(p)==12
    np.testing.assert_allclose(np.linalg.norm(p,axis=0),1,atol=1e-6)
    np.testing.assert_allclose(q.T@q,np.eye(9),atol=1e-6)
    assert not np.allclose(p.T@p,np.eye(20))
    other=make_embeddings(3101,3102)
    for a,b in zip((p,q),other):np.testing.assert_array_equal(a,b)


def test_generation_uniqueness_support_and_requested_labels(config):
    d=deepcopy(config['data']);d.update(n_samples=2000,min_samples=1000,max_attempts=50000)
    a,mode,requested,report=generate(d,3103)
    assert len(a)==2000==len(np.unique(keys(a))) and np.all((a.sum(1)>=3)&(a.sum(1)<=6))
    g=targets(a)
    assert np.all(g.sum(0)>0) and np.all(g.sum(0)<len(a))
    for i in np.flatnonzero(mode):assert g[i,requested[i]]==(mode[i]==1)
    assert (clause_counts(a)>=2).any()
    for other in zip((a,mode,requested),generate(d,3103)[:3]):np.testing.assert_array_equal(*other)


def test_stratified_splits_are_disjoint_reproducible_and_have_exact_global_quotas(config):
    d=deepcopy(config['data']);d.update(n_samples=2000,min_samples=1000,max_attempts=50000)
    a,*_=generate(d,12)
    split,manifest=stratified_split(a,[.7,.15,.15],42,20)
    assert [len(split[name]) for name in ('train','validation','test')]==[1400,300,300]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))),np.arange(2000))
    again,_=stratified_split(a,[.7,.15,.15],42,20)
    for name in split:np.testing.assert_array_equal(split[name],again[name])
    populations=[set(keys(a[idx])) for idx in split.values()]
    assert not populations[0]&populations[1] and not populations[0]&populations[2] and not populations[1]&populations[2]


def test_probe_bank_labels_density_and_overlapping_rule_controls(config):
    a,meta,bases,names=probe_bank(config);g=targets(a)
    assert len(a)==len(meta) and len(names)==len(bases)
    for i,m in enumerate(meta):
        assert m['outside_training_density']==(not 3<=a[i].sum()<=6)
        if m['kind']=='overlap_012' and m['context']=='isolated:d0' and m['pattern']=='101':assert g[i,1]==1 and g[i,8]==0
    for bit in range(15,20):
        idx=[i for i,m in enumerate(meta) if m['kind']=='distractor_flip' and m['context']==names[0] and m['designed_rule']==str(bit)]
        np.testing.assert_array_equal(g[idx[0]],g[idx[1]])


def test_analytical_relu_boolean_identities_and_signed_cancellation():
    corners=torch.tensor(list(product((0.,1.),repeat=2)),dtype=torch.float64)
    and_value=torch.relu(corners.sum(1)-1)
    or_value=corners.sum(1)-and_value
    xor_value=torch.relu(corners[:,0]-corners[:,1])+torch.relu(corners[:,1]-corners[:,0])
    inhibit=torch.relu(corners[:,0]-corners[:,1])
    values=torch.stack([and_value,or_value,xor_value,inhibit],dim=-1)
    expected=torch.tensor([[x and y,x or y,bool(x)!=bool(y),x and not y] for x,y in corners.tolist()],dtype=torch.float64)
    torch.testing.assert_close(values,expected,atol=0,rtol=0)
    signs=torch.tensor([1.,-1.,-1.,1.],dtype=torch.float64)
    torch.testing.assert_close((values*signs[:,None]).sum(0),torch.tensor([1.,-1.,-2.,-1.],dtype=torch.float64),atol=0,rtol=0)
    coords={'C1':torch.tensor([[1.,1.],[1.,1.]],dtype=torch.float64),'b1':torch.tensor([-1.,-1.],dtype=torch.float64),
            'D2_eff':torch.tensor([[2.,-1.5]],dtype=torch.float64),'semantic_bias':torch.tensor([17.],dtype=torch.float64)}
    interaction=finite_interaction(torch.zeros(1,2,dtype=torch.float64),(0,1),coords)
    assert interaction['contributions'].tolist()==[[[2.,-1.5]]] and interaction['value'].item()==.5


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.cuda)])
def test_actual_v2_shapes_coordinates_checkpoint_and_neuron_sums(config,tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    model=OriginalMLP(12,64,12).to(device=device,dtype=torch.float64)
    p,q=[torch.tensor(v,dtype=torch.float64,device=device) for v in make_embeddings(1,2)]
    a=torch.tensor(np.random.default_rng(4).integers(0,2,(23,20)),dtype=torch.float64,device=device)
    c=original_coordinates(model,p,q)
    assert c['C1'].shape==(64,20) and c['D2_eff'].shape==(9,64)
    torch.testing.assert_close(model.W1.weight@(p@a[0]),(model.W1.weight@p)@a[0])
    torch.testing.assert_close(model(a@p.T)@q,semantic_forward(a,c))
    for pair in ((0,1),(11,12),(13,14)):
        r=finite_interaction(a,pair,c);corners=r['corners']
        actual=(model((corners@p.T).reshape(-1,12))@q).reshape(23,4,9)
        delta=(actual*r['signs'][None,:,None]).sum(1)
        torch.testing.assert_close(r['contributions'].sum(-1),delta,atol=1e-10,rtol=1e-8)
    checkpoint=tmp_path/'teacher.pt';save_checkpoint(checkpoint,{'model':model.state_dict()})
    restored=OriginalMLP(12,64,12).to(device=device,dtype=torch.float64);restored.load_state_dict(load_checkpoint(checkpoint)['model'])
    torch.testing.assert_close(restored(a@p.T),model(a@p.T),atol=0,rtol=0)
    with pytest.raises(ValueError):model(a) # Semantic a must never be passed as x.


def test_nine_output_metric_definitions_and_undefined_classes():
    _,q=make_embeddings(1,2);g=np.array([[0]*9,[0]*9,[1]*9,[1]*9],dtype=float)
    semantic=np.repeat(np.array([[0.],[1.],[0.],[1.]]),9,axis=1)
    result=metrics(semantic@q.T,g,q,.5);r=result['per_rule'][0]
    assert len(result['per_rule'])==9 and result['exact_vector_accuracy']==.5
    assert r['precision']==r['recall']==r['false_positive_rate']==r['false_negative_rate']==.5
    assert metrics(np.zeros((4,12)),np.zeros((4,9)),q,.5)['per_rule'][0]['recall'] is None


def test_frozen_configuration_and_no_cpu_fallback(config,monkeypatch):
    validate(config)
    bad=deepcopy(config);bad['dimensions']['hidden']=1024
    with pytest.raises(ValueError,match='frozen'):validate(bad)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='No CPU fallback'):select_device('cuda')
