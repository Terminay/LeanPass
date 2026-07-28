---
sidebar_position: 3
---

# optim API Reference

Optimization algorithms for training neural networks.

## SGD

```python
class leanpass.optim.SGD(parameters, lr=0.01)
```

Stochastic Gradient Descent — the classic optimizer. Simple, effective, no frills.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parameters` | list[Tensor] | — | List of trainable tensors (from `model.parameters()`) |
| `lr` | float | `0.01` | Learning rate |

### `step()`
Performs a single optimization step. Updates each parameter:
```
param.data -= lr * param.grad
```

```python
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    loss = model(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### `zero_grad()`
Resets gradients of all tracked parameters to zero. Shorthand for `model.zero_grad()` at the optimizer level.

```python
optimizer.zero_grad()
```

## Adam

```python
class leanpass.optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

Adam optimizer — adaptive moment estimation with bias correction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parameters` | list[Tensor] | — | List of trainable tensors |
| `lr` | float | `0.001` | Learning rate |
| `betas` | tuple[float, float] | `(0.9, 0.999)` | Coefficients for running averages of gradient and squared gradient |
| `eps` | float | `1e-8` | Small constant for numerical stability |

### `step()`
Performs a single optimization step with bias-corrected Adam updates.

### `zero_grad()`
Resets gradients of all tracked parameters to zero.

## Choosing an Optimizer

| Optimizer | When to Use | Typical LR |
|-----------|-------------|------------|
| **SGD** | Small datasets, simple problems, when you want to see gradients clearly | 0.01 — 0.1 |
| **Adam** | Larger datasets, faster convergence, works well out of the box | 0.001 — 0.01 |

Both optimizers support the same interface: `step()` and `zero_grad()`.
