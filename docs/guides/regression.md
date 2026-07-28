---
sidebar_position: 5
---

# Regression with Neural Networks

Using a multi-layer network for non-linear regression — fitting a curve, not just a line.

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import MLP, mse_loss
from leanpass.optim import SGD

# Generate synthetic non-linear data: y = sin(x) + noise
np.random.seed(42)
x_data = np.linspace(-3, 3, 200).reshape(-1, 1)
y_data = np.sin(x_data) + 0.1 * np.random.randn(200, 1)

x = Tensor(x_data)
y = Tensor(y_data)

model = MLP([1, 32, 1])
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    pred = model(x)
    loss = mse_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d} | loss {loss.data:.6f}")

print(f"\nfinal loss: {loss.data:.6f}")
```

## What's happening

1. We generate points along a sine wave with added noise
2. An `MLP([1, 32, 1])` learns the mapping — one input, one hidden layer of 32, one output
3. The hidden layer with ReLU activation lets the network learn non-linear patterns
4. `mse_loss` measures how well the predicted curve fits the data

## Key takeaways

- Regression with neural networks is just curve fitting
- A single hidden layer can approximate any continuous function (universal approximation theorem)
- No activation on the output layer — we want unbounded predictions
- More data and more epochs help the network generalize
