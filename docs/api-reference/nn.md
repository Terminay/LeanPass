---
sidebar_position: 2
---

# nn API Reference

Neural network building blocks: layers, loss functions, and the base `Module` class.

## Module

```python
class leanpass.nn.Module
```

Base class for all neural network modules. Provides parameter management.

### `parameters()`
Returns a list of all trainable `Tensor` parameters in the module and its sub-modules.

```python
model = MLP([4, 16, 2])
params = model.parameters()  # list of Tensor objects
```

### `zero_grad()`
Resets gradients of all parameters to zero.

```python
model.zero_grad()
```

## Linear

```python
class leanpass.nn.Linear(in_features, out_features)
```

A fully connected layer: `y = x @ W + b`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_features` | int | Number of input features |
| `out_features` | int | Number of output features |

**Parameters:**
- `weight` — Tensor of shape `(in_features, out_features)`, initialized with uniform Xavier (Glorot) init: `~U(-1/√n, 1/√n)`
- `bias` — Tensor of shape `(out_features,)`, initialized to zeros

### `forward(x)`
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | Tensor | Input tensor of shape `(batch_size, in_features)` |

Returns a Tensor of shape `(batch_size, out_features)`.

## MLP

```python
class leanpass.nn.MLP(layer_sizes)
```

A simple feedforward network with ReLU activations between layers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer_sizes` | list[int] | List specifying the size of each layer, including input and output |

```python
# 3-layer: 4 → 16 → 16 → 2
model = MLP([4, 16, 16, 2])
```

The last layer has no activation — apply softmax or sigmoid separately if needed.

### `forward(x)`
| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | Tensor | Input tensor of shape `(batch_size, input_dim)` |

Returns a Tensor of shape `(batch_size, output_dim)`.

## Loss Functions

### `mse_loss(prediction, target)`
Mean squared error loss for regression.

```python
from leanpass.nn import mse_loss

loss = mse_loss(prediction, target)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prediction` | Tensor | Model output |
| `target` | Tensor | Ground truth |

Returns a scalar Tensor: `mean((prediction - target)²)`.

### `cross_entropy_loss(prediction, target, axis=-1)`
Categorical cross-entropy loss for multi-class classification.

```python
from leanpass.nn import cross_entropy_loss

loss = cross_entropy_loss(logits, one_hot_labels)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction` | Tensor | — | Raw logits (not softmaxed) |
| `target` | Tensor | — | One-hot target distribution, same shape as prediction |
| `axis` | int | `-1` | Class dimension |

Internally applies softmax to predictions, then computes `-∑ target * log(prob)`.

### `binary_cross_entropy_loss(prediction, target)`
Binary cross-entropy loss for binary classification.

```python
from leanpass.nn import binary_cross_entropy_loss

loss = binary_cross_entropy_loss(logits, binary_labels)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prediction` | Tensor | Raw logits (not sigmoided) |
| `target` | Tensor | Binary labels (0 or 1, float) |

Internally applies sigmoid, then computes `-mean(target * log(p) + (1-target) * log(1-p))`.
