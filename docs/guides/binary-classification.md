---
sidebar_position: 3
---

# Binary Classification

Classifying data into one of two categories using a neural network with sigmoid output.

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import MLP, binary_cross_entropy_loss
from leanpass.optim import SGD

# Generate synthetic binary classification data
np.random.seed(42)
n_samples = 200
x_data = np.random.randn(n_samples, 2)
y_data = ((x_data[:, 0]**2 + x_data[:, 1]**2) < 2.0).astype(float).reshape(-1, 1)

x = Tensor(x_data)
y = Tensor(y_data)

model = MLP([2, 16, 1])
optimizer = SGD(model.parameters(), lr=0.5)

for epoch in range(500):
    logits = model(x)
    loss = binary_cross_entropy_loss(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        probs = logits.sigmoid()
        acc = ((probs.data > 0.5) == y_data).mean()
        print(f"epoch {epoch:3d} | loss {loss.data:.4f} | acc {acc:.3f}")

# Final accuracy
probs = model(x).sigmoid()
acc = ((probs.data > 0.5) == y_data).mean()
print(f"\nfinal accuracy: {acc:.3f}")
```

## What's happening

1. We generate 2D points and label them based on whether they fall inside a circle
2. An `MLP([2, 16, 1])` learns the decision boundary — 2 inputs, one hidden layer of 16, one output
3. `binary_cross_entropy_loss` applies sigmoid internally, then computes `-mean(y * log(p) + (1-y) * log(1-p))`
4. Accuracy measures how often the predicted class matches the true class

## Key takeaways

- Binary classification needs a single output neuron with sigmoid
- `binary_cross_entropy_loss` expects raw logits, not probabilities
- A hidden layer lets the network learn non-linear decision boundaries
