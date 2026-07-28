---
sidebar_position: 4
---

# Multi-Class Classification

Classifying data into more than two categories using softmax output and cross-entropy loss.

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import MLP, cross_entropy_loss
from leanpass.optim import SGD

# Generate synthetic 3-class data
np.random.seed(42)
n_samples = 300
x_data = np.random.randn(n_samples, 2)

# Three clusters
y_data = np.zeros((n_samples, 3))
for i in range(n_samples):
    cluster = i % 3
    x_data[i] += np.array([cluster * 3, 0])
    y_data[i, cluster] = 1.0

x = Tensor(x_data)
y = Tensor(y_data)

model = MLP([2, 32, 3])
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    logits = model(x)
    loss = cross_entropy_loss(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 60 == 0:
        probs = logits.softmax()
        preds = np.argmax(probs.data, axis=1)
        labels = np.argmax(y_data, axis=1)
        acc = (preds == labels).mean()
        print(f"epoch {epoch:3d} | loss {loss.data:.4f} | acc {acc:.3f}")

# Final accuracy
probs = model(x).softmax()
preds = np.argmax(probs.data, axis=1)
labels = np.argmax(y_data, axis=1)
print(f"\nfinal accuracy: {(preds == labels).mean():.3f}")
```

## What's happening

1. Three clusters of 2D points, each cluster is a separate class
2. One-hot encoded labels: `[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]`
3. An `MLP([2, 32, 3])` maps 2 inputs → 32 hidden → 3 output logits
4. `cross_entropy_loss` applies softmax internally, compares with one-hot targets
5. Accuracy is measured by which class gets the highest probability

## Key takeaways

- Multi-class needs `C` output neurons (one per class) with softmax
- Labels should be one-hot encoded
- `cross_entropy_loss` expects raw logits, not softmaxed probabilities
