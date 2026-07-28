---
sidebar_position: 2
---

# XOR Classification

The "hello world" of neural networks — solving the XOR problem, which a single linear layer cannot do.

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import MLP, binary_cross_entropy_loss
from leanpass.optim import SGD

# XOR truth table: inputs and labels
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

x = Tensor(x_data)
y = Tensor(y_data.reshape(-1, 1))

model = MLP([2, 8, 1])
optimizer = SGD(model.parameters(), lr=0.5)

for epoch in range(1000):
    logits = model(x)
    probs = logits.sigmoid()
    loss = binary_cross_entropy_loss(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        preds = (probs.data > 0.5).astype(int)
        acc = (preds.flatten() == y_data).mean()
        print(f"epoch {epoch:3d} | loss {loss.data:.4f} | acc {acc:.1f}")

# Final predictions
probs = model(x).sigmoid()
print("\nPredictions:")
for i, (inp, prob) in enumerate(zip(x_data, probs.data)):
    print(f"  {inp} → {prob[0]:.4f} (expected {y_data[i]})")
```

## What's happening

1. XOR is a non-linearly separable problem — no single line can separate the [0,0]/[1,1] from [0,1]/[1,0]
2. A single `Linear` layer can only draw straight lines, so it fails
3. Adding a hidden layer with non-linear activation lets the network learn the XOR function
4. `MLP([2, 8, 1])` has 2 inputs → 8 hidden neurons with ReLU → 1 output with sigmoid
5. `binary_cross_entropy_loss` measures how well the predictions match the binary labels

## Key takeaways

- XOR requires at least one hidden layer to solve
- Non-linear activation functions (like ReLU) are essential for non-linear problems
- A small network (2-8-1) is sufficient for XOR
- The network should learn to predict probabilities close to 0 or 1 for each class