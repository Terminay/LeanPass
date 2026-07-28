---
sidebar_position: 8
---

# Custom Layers

Extending LeanPass with your own layer types by subclassing `Module`.

## The Module interface

Every layer inherits from `nn.Module` and needs to:

1. Store parameters as `Tensor` objects with `requires_grad=True`
2. Implement `forward()` with the layer's computation
3. Define `__call__` (usually just delegates to `forward`)
4. `parameters()` is inherited — it recursively finds all `Tensor` attributes with `requires_grad=True`

## Example: Dropout layer

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import Module

class Dropout(Module):
    """Randomly zeroes neurons during training to prevent co-adaptation."""

    def __init__(self, rate=0.5):
        self.rate = rate
        self.training = True

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            return x
        mask = np.random.binomial(1, 1 - self.rate, size=x.data.shape) / (1 - self.rate)
        return Tensor(mask * x.data)

    def __call__(self, x):
        return self.forward(x)
```

## Example: Tanh activation

```python
class Tanh(Module):
    """Hyperbolic tangent activation."""

    def forward(self, x):
        pos = x.exp()
        neg = (-x).exp()
        return (pos - neg) / (pos + neg)

    def __call__(self, x):
        return self.forward(x)
```

## Using custom layers in a model

```python
from leanpass.nn import Linear, MLP

class CustomMLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.2):
        self.fc1 = Linear(in_dim, hidden_dim)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def __call__(self, x):
        return self.forward(x)

model = CustomMLP(784, 128, 10)
params = model.parameters()  # automatically finds fc1.weight, fc1.bias, fc2.weight, fc2.bias
print(f"Total parameters: {len(params)}")
```

## Guidelines for custom layers

- **Parameters go in `__init__`** — don't create them in `forward()`
- **Use `requires_grad=True`** only for trainable parameters
- **Return a Tensor** from `forward()` — the graph needs to track operations
- **`parameters()` is recursive** — it finds Tensor attributes on sub-modules too
- **Test gradients** — run `y.backward()` and check that gradients flow through your layer
