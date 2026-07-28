---
sidebar_position: 1
---

# Getting Started

## Installation

```bash
pip install leanpass
```

That's it. No CUDA, no C++ extensions, no 500MB download. Just NumPy.

## Your First Forward Pass

```python
import numpy as np
from leanpass import Tensor

# Create some inputs
x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
w = Tensor(np.array([0.5, -0.2, 0.1]), requires_grad=True)
b = Tensor(np.array([0.0]), requires_grad=True)

# Forward pass
y = (x * w).sum() + b

print(y)  # Tensor(shape=(), requires_grad=True)
```

## Your First Backward Pass

```python
# Backward pass — trace gradients through the graph
y.backward()

print(w.grad)  # [1. 2. 3.]
print(x.grad)  # [ 0.5 -0.2  0.1]
print(b.grad)  # [1.]
```

LeanPass builds a computation graph as you write expressions, then walks it backward when you call `.backward()` — applying the chain rule at each node.

## Your First Model

```python
from leanpass.nn import MLP, mse_loss
from leanpass.optim import SGD

# 2-layer network: 4 → 16 → 2
model = MLP([4, 16, 2])
optimizer = SGD(model.parameters(), lr=0.01)

# Dummy data
x = Tensor(np.random.randn(32, 4))
target = Tensor(np.random.randn(32, 2))

for step in range(100):
    pred = model(x)
    loss = mse_loss(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.data:.6f}")
```

That's the core loop — you'll see this pattern throughout the demos.
