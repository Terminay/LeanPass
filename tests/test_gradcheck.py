import numpy as np

from leanpass import Tensor


def finite_difference_grad(f, x, eps=1e-6):
    orig = x.data.copy()
    grad = np.zeros_like(x.data)
    for idx in np.ndindex(x.data.shape):
        x.data[idx] = orig[idx] + eps
        plus = f().data.copy()
        x.data[idx] = orig[idx] - eps
        minus = f().data.copy()
        x.data[idx] = orig[idx]
        grad[idx] = (plus - minus).sum() / (2 * eps)
    return grad


def test_grad_check_add():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)
    z = (x + y).sum()
    z.backward()

    assert np.allclose(x.grad, [1.0, 1.0])
    assert np.allclose(y.grad, [1.0, 1.0])


def test_grad_check_power():
    x = Tensor([2.0, 3.0], requires_grad=True)
    z = (x ** 3).sum()
    z.backward()

    assert np.allclose(x.grad, [12.0, 27.0])


def test_grad_check_sigmoid():
    x = Tensor([0.5, -0.5], requires_grad=True)
    z = x.sigmoid().sum()
    z.backward()

    numeric = finite_difference_grad(lambda: x.sigmoid(), x)
    assert np.allclose(x.grad, numeric, atol=1e-4)


def test_grad_check_exp_log():
    x = Tensor([0.5, 1.0], requires_grad=True)
    z = x.exp().log().sum()
    z.backward()

    numeric = finite_difference_grad(lambda: x.exp().log(), x)
    assert np.allclose(x.grad, numeric, atol=1e-4)


def test_grad_check_softmax():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    z = x.softmax().sum()
    z.backward()

    numeric = finite_difference_grad(lambda: x.softmax(), x)
    assert np.allclose(x.grad, numeric, atol=1e-4)


def test_grad_check_graph_visualizer():
    x = Tensor([1.0, 2.0], requires_grad=True, name="input")
    y = (x * x + x.sigmoid()).sum()
    graph = y.visualize()

    assert "softmax" not in graph
    assert "relu" not in graph
    assert "input" in graph


def test_visualize_dot():
    x = Tensor([1.0, 2.0], requires_grad=True, name="input")
    y = (x * x + x.sigmoid()).sum()
    dot = y.visualize_dot()

    assert dot.startswith("digraph")
    assert "->" in dot
    assert "input" in dot


def test_eval_forward_handles_reductions():
    # sum() and mean() stored their _prev as a set instead of a tuple, so
    # _eval_forward's `node._prev[0]` indexing raised
    # "TypeError: 'set' object is not subscriptable" for any graph that reduces
    # with sum or mean, which is almost every loss.
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    assert np.isclose((x * x).sum()._eval_forward(), 30.0)

    y = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert np.isclose((y * y).mean()._eval_forward(), 14.0 / 3.0)
