# LeanPass

LeanPass is a lightweight NumPy-based autodiff library for building small neural network models and understanding automatic differentiation in a simple, readable way.

```bash
pip install leanpass
```

## Public API

### Tensor

The core data structure is `Tensor`, which wraps a NumPy array and supports automatic differentiation.

```python
from leanpass import Tensor

x = Tensor([[1.0, 2.0]], requires_grad=False)
y = Tensor([[3.0, 4.0]])
z = x + y
```

Supported operations:

- `Tensor + Tensor`
- `Tensor - Tensor`
- `Tensor * Tensor`
- `Tensor / Tensor`
- `Tensor ** Tensor`
- `Tensor @ Tensor`
- `Tensor.sum()`
- `Tensor.mean()`
- `Tensor.relu()`
- `Tensor.sigmoid()`
- `Tensor.tanh()`
- `Tensor.leaky_relu()`
- `Tensor.gelu()`
- `Tensor.softmax()`
- `Tensor.backward()`

### Neural network layers

```python
from leanpass import nn

layer = nn.Linear(4, 8)
mlp = nn.MLP([4, 16, 8])
```

Available components:

- `nn.Linear(in_features, out_features)` creates a linear layer with weights and bias.
- `nn.MLP(layer_sizes)` creates a multilayer perceptron with ReLU activations between layers.
- `nn.mse_loss(predictions, targets)` computes mean squared error.
- `nn.cross_entropy_loss(predictions, targets)` computes categorical cross-entropy for multi-class targets.
- `nn.binary_cross_entropy_loss(predictions, targets)` computes binary cross-entropy for binary classification.

### Optimizers

```python
from leanpass import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
# or
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Available optimizers:

- `optim.SGD(parameters, lr=...)` performs simple gradient descent.
- `optim.Adam(parameters, lr=...)` performs Adam optimization with bias correction.

### Example

```python
from leanpass import Tensor, nn, optim

x = Tensor([[1.0, 2.0]], requires_grad=False)
model = nn.MLP([2, 16, 3])
output = model(x)
print(output)
```

## Notes

This package is intended for clarity and educational use, with a compact implementation that makes the autodiff process easier to inspect.

Full documentation: [https://leanpass.kilobyte136.workers.dev/](https://leanpass.kilobyte136.workers.dev/)