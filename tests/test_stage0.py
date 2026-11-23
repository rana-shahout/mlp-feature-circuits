"""Stage 0 analytical fixtures; these do not train or analyze a learned circuit."""

from itertools import product

import numpy as np
import pytest
import torch

from src.data import embeddings
from src.models import OriginalMLP


def analytical_fixture():
    # Inputs (a,b,c). Outputs AND(a,b), OR(a,b), XOR(a,b), a AND NOT b,
    # AND(a,b,c). Explicit ReLU identities, not claims about trained mechanisms.
    model = OriginalMLP(3, 7, 5).double()
    with torch.no_grad():
        model.W1.weight.copy_(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                          [1, 1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 1]]))
        model.W1.bias.copy_(torch.tensor([0, 0, 0, -1, 0, 0, -2]))
        model.W2.weight.zero_()
        model.W2.bias.zero_()
        model.W2.weight[0, 3] = 1
        model.W2.weight[1, [0, 1, 3]] = torch.tensor([1., 1., -1.], dtype=torch.float64)
        model.W2.weight[2, [4, 5]] = 1
        model.W2.weight[3, 4] = 1
        model.W2.weight[4, 6] = 1
    return model


def test_hand_constructed_network_full_truth_table():
    a = torch.tensor(list(product((0., 1.), repeat=3)), dtype=torch.float64)
    expected = torch.tensor([[x and y, x or y, bool(x) != bool(y), x and not y, x and y and z]
                             for x, y, z in a.tolist()], dtype=torch.float64)
    torch.testing.assert_close(analytical_fixture()(a), expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_forward_shapes_and_coordinate_identities(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA identity check requires CUDA")
    p, q = (torch.tensor(v, dtype=torch.float64, device=device) for v in embeddings(16, 12, 1, 2))
    rng = np.random.default_rng(3)
    a = torch.tensor(rng.integers(0, 2, size=(23, 32)), dtype=torch.float64, device=device)
    model = OriginalMLP().to(device=device, dtype=torch.float64)
    x = a @ p.T
    pre, hidden = model.hidden(x)
    assert pre.shape == hidden.shape == (23, 48)
    assert model(x).shape == (23, 12)
    assert next(model.parameters()).device.type == device
    # Column-vector and batched equations, including all biases/nonlinearities.
    torch.testing.assert_close(model.W1.weight @ (p @ a[0]), (model.W1.weight @ p) @ a[0])
    c1 = model.W1.weight @ p
    d2_eff = q.T @ model.W2.weight
    reference = (d2_eff @ torch.relu(c1 @ a.T + model.W1.bias[:, None]) + (q.T @ model.W2.bias)[:, None]).T
    torch.testing.assert_close(model(x) @ q, reference)
    column_forward = model.W2.weight @ torch.relu(model.W1.weight @ x[0] + model.W1.bias) + model.W2.bias
    torch.testing.assert_close(model(x)[0], column_forward)
    with pytest.raises(ValueError, match="Expected x"):
        model(a)
