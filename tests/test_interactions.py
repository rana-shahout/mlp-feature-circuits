"""Analytical diagnostics tests, written but not executed by Codex."""

from itertools import product

import numpy as np
import pytest
import torch

from scripts.diagnose_mlp import finite_corners, validation_rank
from src.coordinates import original_coordinates, semantic_forward
from src.data import embeddings, targets
from src.diagnostics import diagnostic_metrics, grouped_rows, make_contexts
from src.interactions import finite_interaction, decompose_corners
from src.models import OriginalMLP


def scalar_formula(rule, a):
    x, y, z = [int(v) for v in a]
    return {"AND": x*y, "OR": int(x or y), "XOR": int(x != y),
            "inhibition": x*(1-y), "AND3": x*y*z}[rule]


def fixture_coordinates():
    # h = a, b, relu(a+b-1), relu(a-b), relu(b-a), relu(a+b+c-2).
    return {"C1": torch.tensor([[1.,0,0],[0,1,0],[1,1,0],[1,-1,0],[-1,1,0],[1,1,1]], dtype=torch.float64),
            "b1": torch.tensor([0.,0,-1,0,0,-2], dtype=torch.float64),
            "D2_eff": torch.tensor([[0.,0,1,0,0,0],[1,1,-1,0,0,0],[0,0,0,1,1,0],
                                     [0,0,0,1,0,0],[0,0,0,0,0,1]], dtype=torch.float64),
            "semantic_bias": torch.tensor([2.,-3,4,-5,6], dtype=torch.float64)}


def test_all_analytical_pair_and_triple_differences_against_scalar_references():
    coord = fixture_coordinates()
    contexts = torch.tensor(list(product((0.,1.), repeat=3)), dtype=torch.float64)
    for ids in [(0,1),(0,2),(1,2),(0,1,2)]:
        result = finite_interaction(contexts, ids, coord)
        expected = []
        for context in contexts.tolist():
            row = []
            for rule in ("AND","OR","XOR","inhibition","AND3"):
                total = 0
                for bits in product((0,1), repeat=len(ids)):
                    a = context.copy()
                    for i, bit in zip(ids, bits):
                        a[i] = bit
                    total += (-1)**(len(ids)-sum(bits))*scalar_formula(rule,a)
                row.append(total)
            expected.append(row)
        torch.testing.assert_close(result["value"], torch.tensor(expected, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(result["contributions"].sum(-1), result["direct"], atol=0, rtol=0)
    pair = finite_interaction(contexts, (0,1), coord)
    assert pair["value"][0, :4].tolist() == [1., -1., -2., -1.]


def test_signed_cancellation_includes_negative_terms_and_output_bias_cancels():
    coord = {"C1": torch.tensor([[1.,1.],[1.,1.]], dtype=torch.float64),
             "b1": torch.tensor([-1.,-1.], dtype=torch.float64),
             "D2_eff": torch.tensor([[2.,-1.5]], dtype=torch.float64),
             "semantic_bias": torch.tensor([19.], dtype=torch.float64)}
    result = finite_interaction(torch.zeros(1,2,dtype=torch.float64),(0,1),coord)
    assert result["contributions"].tolist() == [[[2.,-1.5]]]
    assert result["value"].item() == 0.5
    assert result["positive_sum"].item() == 2
    assert result["negative_sum"].item() == -1.5
    assert result["cancellation_fraction"].item() == pytest.approx(1-0.5/3.5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_random_original_forward_matches_complete_neuron_differences(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    model = OriginalMLP().to(device=device, dtype=torch.float64)
    p, q = [torch.tensor(v, dtype=torch.float64, device=device) for v in embeddings(16,12,11,12)]
    coords = original_coordinates(model,p,q)
    base = torch.tensor(np.random.default_rng(3).integers(0,2,(4,32)), dtype=torch.float64, device=device)
    torch.testing.assert_close(model(base@p.T)@q, semantic_forward(base,coords))
    for ids in [(8,9),(8,9,10),(18,19,20)]:
        result = finite_interaction(base, ids, coords)
        corners = result["corners"]
        actual = (model((corners@p.T).reshape(-1,16))@q).reshape(4,2**len(ids),8)
        actual_delta = (actual*result["signs"][None,:,None]).sum(1)
        torch.testing.assert_close(result["value"], actual_delta, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(result["value"], result["positive_sum"]+result["negative_sum"])


def test_batched_corners_preserve_fixed_context_and_corner_signs():
    contexts=np.zeros((2,32),dtype=np.uint8)
    contexts[:,[8,9,10,20]]=1
    for terms in [[(8,9),(0,1)],[(8,9,10),(18,19,20)]]:
        a,signs=finite_corners(contexts,terms)
        assert a.shape==(2,2,2**len(terms[0]),32)
        for t,ids in enumerate(terms):
            other=np.setdiff1d(np.arange(32),ids)
            assert np.all(np.take(a[t],other,axis=-1)==np.take(contexts,other,axis=-1)[:,None,:])
            for corner,bits in enumerate(product((0,1),repeat=len(ids))):
                for col,idx in enumerate(ids):
                    assert np.all(a[t,:,corner,idx]==bits[col])
                assert signs[corner]==(-1)**(len(ids)-sum(bits))
        values=targets(a)
        if len(terms[0])==3:
            np.testing.assert_array_equal((values[0]*signs[None,:,None]).sum(1)[:,4],[1,1])


def test_diagnostic_metric_definitions_and_zero_denominators():
    _,q=embeddings(16,12,1,2)
    g=np.array([[0]*8,[0]*8,[1]*8,[1]*8],dtype=float)
    scores=np.repeat(np.array([[0.],[1.],[0.],[1.]]),8,axis=1)
    metrics=diagnostic_metrics(scores@q.T,g,q,0.5)
    r=metrics["per_rule"][0]
    assert metrics["exact_vector_accuracy"]==0.5
    assert r["precision"]==r["recall"]==r["false_positive_rate"]==r["false_negative_rate"]==0.5
    empty_class=diagnostic_metrics(np.zeros((4,12)),np.zeros((4,8)),q,0.5)["per_rule"][0]
    assert empty_class["false_negative_rate"] is None
    assert empty_class["false_positive_rate"]==0


def test_grouped_errors_have_correct_support_and_do_not_merge_distractor_sets():
    a=np.zeros((4,32),dtype=np.uint8);a[:2,18]=1;a[2:,19]=1
    g=np.array([[0]*8,[1]*8,[1]*8,[0]*8],dtype=float)
    scores=np.repeat(np.array([[0.],[0.],[1.],[1.]]),8,axis=1)
    rows=list(grouped_rows(a,scores,g,.5,np.array([1,1,2,2]),"identities"))
    row=rows[0]
    assert row["n"]==2 and row["n_positive"]==1
    assert row["false_negative_rate"]==1 and row["false_positive_rate"]==0
    assert len(rows)==16


def test_contexts_reproducible_and_validation_seed_ranking_does_not_use_test():
    settings=[{"name":"a","semantic_active":[10],"distractor_count":4}]
    a,names=make_contexts(settings,4);b,_=make_contexts(settings,4)
    np.testing.assert_array_equal(a,b)
    assert a[0,10]==1 and a[0,18:].sum()==4 and names==["a"]
    def metrics(score,test):
        return {"validation":{"per_rule":[{"balanced_accuracy":score}]*8,"output_mse":.1},"test":{"exact_vector_accuracy":test}}
    assert validation_rank(metrics(.95,.1))>validation_rank(metrics(.90,1.))


def test_affine_and_dead_neurons_have_no_pair_or_triple_interactions():
    coord={"C1":torch.tensor([[1.,1.,1.],[1.,-1.,2.]],dtype=torch.float64),
           "b1":torch.tensor([3.,-100.],dtype=torch.float64),
           "D2_eff":torch.tensor([[2.,7.]],dtype=torch.float64),
           "semantic_bias":torch.tensor([12.],dtype=torch.float64)}
    base=torch.zeros((1,3),dtype=torch.float64)
    for ids in [(0,1),(0,1,2)]:
        result=finite_interaction(base,ids,coord)
        assert torch.count_nonzero(result["contributions"])==0
        assert result["value"].item()==0


def test_nested_xor_third_difference_is_negative_two():
    a,signs=finite_corners(np.zeros((1,32),dtype=np.uint8),[(4,16,17)])
    value=(targets(a).astype(float)*signs[None,None,:,None]).sum(-2)
    assert value[0,0,7]==-2


def test_batched_decomposition_matches_each_analytical_context_and_term():
    contexts=np.zeros((8,32),dtype=np.uint8)
    contexts[:,:3]=np.asarray(list(product((0,1),repeat=3)))
    terms=[(0,1),(0,2),(1,2)]
    a,signs=finite_corners(contexts,terms)
    coord=fixture_coordinates()
    batch=decompose_corners(torch.tensor(a[...,:3],dtype=torch.float64),torch.tensor(signs),coord)
    for t,term in enumerate(terms):
        scalar=finite_interaction(torch.tensor(contexts[:,:3],dtype=torch.float64),term,coord)
        torch.testing.assert_close(batch["value"][t],scalar["value"],atol=0,rtol=0)
        torch.testing.assert_close(batch["contributions"][t],scalar["contributions"],atol=0,rtol=0)


def test_semantic_context_changes_hold_distractor_identities_fixed():
    settings=[{"name":"off","semantic_active":[],"distractor_count":4},
              {"name":"on","semantic_active":[10],"distractor_count":4}]
    a,_=make_contexts(settings,901)
    np.testing.assert_array_equal(a[0,18:],a[1,18:])
    assert a[0,10]==0 and a[1,10]==1
