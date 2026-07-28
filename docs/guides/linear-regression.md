---
sidebar_position: 1
---

# Linear Regression

The simplest thing that works: fitting a line to data.

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import Linear, mse_loss
from leanpass.optim import SGD

# Generate synthetic data: y = 2x + 1 + noise
np.random.seed(42)
x_data = np.random.randn(100, 1)
y_data = 2 * x_data + 1 + 0.1 * np.random.randn(100, 1)

x = Tensor(x_data)
y = Tensor(y_data)

model = Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    pred = model(x)
    loss = mse_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d} | loss {loss.data:.6f}")

print(f"\nweight: {model.weight.data.flatten()[0]:.4f}  (expected ≈ 2.0)")
print(f"bias:   {model.bias.data.flatten()[0]:.4f}   (expected ≈ 1.0)")
```

## What's happening

1. We create a `Linear(1, 1)` layer — one input, one output, with a learned weight and bias
2. Each epoch: forward pass → compute loss → zero gradients → backward pass → update parameters
3. After 200 steps, the weight and bias should converge close to the true values

## Key takeaways

- A single `Linear` layer is equivalent to `y = xW + b`
- `mse_loss` measures how far off we are
- SGD nudges the parameters in the direction that reduces the loss
- No activation needed — this is pure linear regression
