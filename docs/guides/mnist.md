---
sidebar_position: 6
---

# MNIST Digit Classification

Training a neural network to recognize handwritten digits from the MNIST dataset.

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import MLP, cross_entropy_loss
from leanpass.optim import SGD

# Load MNIST (using a lightweight loader — assumes CSV format)
# Format: first column is label, remaining 784 columns are pixel values
data = np.loadtxt("mnist_train.csv", delimiter=",", max_rows=1000)
x_data = data[:, 1:] / 255.0  # normalize pixels to [0, 1]
y_data = data[:, 0].astype(int)

# One-hot encode labels
y_onehot = np.zeros((len(y_data), 10))
y_onehot[np.arange(len(y_data)), y_data] = 1.0

x = Tensor(x_data)
y = Tensor(y_onehot)

model = MLP([784, 128, 64, 10])
optimizer = SGD(model.parameters(), lr=0.1)

batch_size = 32
n_batches = len(x_data) // batch_size

for epoch in range(20):
    epoch_loss = 0.0
    for i in range(n_batches):
        batch_x = Tensor(x_data[i * batch_size:(i + 1) * batch_size])
        batch_y = Tensor(y_onehot[i * batch_size:(i + 1) * batch_size])

        logits = model(batch_x)
        loss = cross_entropy_loss(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data

    # Full evaluation
    logits = model(x)
    probs = logits.softmax()
    preds = np.argmax(probs.data, axis=1)
    acc = (preds == y_data).mean()
    print(f"epoch {epoch:2d} | loss {epoch_loss / n_batches:.4f} | acc {acc:.3f}")
```

## What's happening

1. MNIST images are 28×28 pixels, flattened to 784 inputs
2. We normalize pixel values to [0, 1] for stable training
3. An `MLP([784, 128, 64, 10])` has two hidden layers — enough for decent accuracy
4. Mini-batch training processes 32 samples at a time for efficiency
5. `cross_entropy_loss` with softmax handles the 10-class classification

## Key takeaways

- Normalize inputs to [0, 1] for faster convergence
- Mini-batches make training more efficient than full-batch
- Deeper networks (more layers) can learn more complex patterns
- MNIST is a good benchmark — if your model can't hit 90%+, something's wrong
