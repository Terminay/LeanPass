---
sidebar_position: 4
---

# Activations

Non-linear activation functions are what give neural networks their expressive power. LeanPass includes the essential ones.

## ReLU

```python
x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
y = x.relu()
print(y.data)  # [0. 0. 0. 1. 2.]
```

Rectified Linear Unit — sets negative values to zero. Used in hidden layers.

**Gradient:** `1.0` for positive inputs, `0.0` for negative.

## Sigmoid

```python
x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
y = x.sigmoid()
print(y.data)  # [0.119 0.269 0.5 0.731 0.881]
```

Squashes values to the range `(0, 1)`. Used for binary classification output.

**Gradient:** `sigmoid(x) * (1 - sigmoid(x))`

## Softmax

```python
logits = Tensor(np.array([[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]]), requires_grad=True)
probs = logits.softmax()
print(probs.data)
# [[0.231 0.628 0.141]
#  [0.049 0.870 0.081]]
```

Converts logits to a probability distribution over classes. Used in multi-class classification.

**Gradient:** `softmax(x) * (δᵢⱼ - softmax(xⱼ))` (the Jacobian, computed internally).

## Exponential

```python
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.exp()
print(y.data)  # [ 2.718  7.389 20.086]
```

The exponential function `eˣ`. Used internally by softmax and sigmoid.

## Logarithm

```python
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.log()
print(y.data)  # [0.      0.693  1.099]
```

The natural logarithm `ln(x)`. Used in cross-entropy loss.

## Which One to Use?

| Activation | When to Use |
|-----------|-------------|
| **ReLU** | Hidden layers (default choice) |
| **Sigmoid** | Binary classification output |
| **Softmax** | Multi-class classification output |
| **Exp / Log** | Loss functions, probability calculations |
