---
sidebar_position: 1
---

# Tensor Basics

The `Tensor` class is the core data structure in LeanPass. It wraps a NumPy array and tracks operations for automatic differentiation.

## Creating Tensors

```python
from leanpass import Tensor
import numpy as np

# From a list
t = Tensor([1.0, 2.0, 3.0])

# From a NumPy array
t = Tensor(np.array([[1, 2], [3, 4]]))

# With gradient tracking
t = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# With a name (useful for debugging)
t = Tensor([1.0, 2.0], requires_grad=True, name="input")
```

## Key Attributes

| Attribute | Description |
|-----------|-------------|
| `.data` | The underlying NumPy array |
| `.grad` | Gradient array (same shape as `.data`), `None` if `requires_grad=False` |
| `.requires_grad` | Whether this tensor tracks gradients |
| `.name` | Optional label for debugging |

```python
t = Tensor([1.0, 2.0, 3.0], requires_grad=True)

print(t.data)       # [1. 2. 3.]
print(t.grad)       # [0. 0. 0.]  (initialized to zeros)
print(t.shape)      # (3,)
```

## Gradient Tracking

Only tensors with `requires_grad=True` accumulate gradients. When you perform operations on them, the resulting tensor also tracks gradients if any of its inputs do.

```python
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=False)  # no gradient tracking

c = a * b
print(c.requires_grad)  # True (a requires grad)

d = a + 5.0
print(d.requires_grad)  # True
```

## Resetting Gradients

Use `zero_grad_all()` to reset gradients in the computation graph:

```python
y = (x * w).sum() + b
y.backward()

# ... after an optimization step ...
y.zero_grad_all()
```

Or use the optimizer's `zero_grad()` method (recommended):

```python
optimizer.zero_grad()
```

## Shape and Broadcasting

LeanPass follows NumPy's broadcasting rules. When gradients flow through broadcasted operations, they're automatically summed back to the original shape.

```python
a = Tensor(np.ones((3, 1)), requires_grad=True)  # shape (3, 1)
b = Tensor(np.ones((1, 4)), requires_grad=True)  # shape (1, 4)
c = a + b  # broadcasts to (3, 4)

c.backward(np.ones((3, 4)))
print(a.grad.shape)  # (3, 1) — summed back
print(b.grad.shape)  # (1, 4) — summed back
```

## Representation

```python
t = Tensor([1.0, 2.0, 3.0], requires_grad=True, name="x")
print(t)
# Tensor(shape=(3,), requires_grad=True, name=x)
