---
sidebar_position: 7
---

# Gradient Checking

Verifying that your autodiff gradients are correct using finite difference approximation.

```python
from leanpass import Tensor

# Build a computation graph
x = Tensor([2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

# Compute analytic gradients
z.backward()
print(f"Analytic gradient: {x.grad}")  # [4.0, 6.0]

# Verify with finite differences
h = 1e-6
x.data[0] += h
z_plus = (x ** 2).sum()
x.data[0] -= 2 * h
z_minus = (x ** 2).sum()
x.data[0] += h

numerical = (z_plus.data - z_minus.data) / (2 * h)
print(f"Numerical gradient: {numerical}")  # ≈ 4.0
```

## Using the gradcheck utility

LeanPass includes a built-in gradient checker for convenience:

```python
from leanpass.tensor import Tensor
from tests.test_gradcheck import check_gradients

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
errors = y._gradcheck()

if errors:
    print("Gradient errors found:")
    for node, diff, analytic, numeric in errors[:3]:
        print(f"  {node.name or 'unnamed'}: diff={diff:.2e}")
else:
    print("All gradients match!")
```

## How it works

Gradient checking computes the numerical gradient using the central difference formula:

```
∂f/∂x ≈ (f(x + h) - f(x - h)) / (2h)
```

For each parameter, it perturbs the value by a small amount `h` (default `1e-6`),
re-evaluates the forward pass, and compares the numerical gradient to the
analytic gradient from backpropagation.

## When to use gradient checking

- **After implementing a new operation** — verify your `_backward` is correct
- **Debugging training issues** — if loss isn't decreasing, check if gradients are wrong
- **Before trusting results** — sanity-check your autodiff implementation

## Limitations

- Gradient checking is **slow** — each parameter requires two forward passes
- Use it for debugging, not for production training loops
- Large networks should be tested on a small subset of parameters
