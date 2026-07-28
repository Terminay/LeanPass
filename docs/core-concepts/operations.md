---
sidebar_position: 3
---

# Operations

LeanPass supports a focused set of tensor operations — enough to build and train neural networks, without the kitchen sink.

## Arithmetic Operations

All standard arithmetic operators work between tensors, or between a tensor and a Python number.

```python
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

c = a + b      # addition
d = a - b      # subtraction
e = a * b      # element-wise multiplication
f = a / b      # division
```

```python
# With Python scalars
g = a + 10.0
h = a * 2.0
i = 3.0 - a
```

## Matrix Multiplication

Use the `@` operator for matrix multiplication (recommended) or `.matmul()`:

```python
x = Tensor(np.random.randn(32, 64), requires_grad=True)
w = Tensor(np.random.randn(64, 16), requires_grad=True)

y = x @ w   # shape (32, 16)
```

## Power

```python
x = Tensor([2.0, 3.0], requires_grad=True)
y = x ** 2   # element-wise square: [4.0, 9.0]
```

The exponent must be an integer or float — currently only `** 2` is supported.

## Negation

```python
x = Tensor([1.0, -2.0], requires_grad=True)
y = -x       # [-1.0, 2.0]
```

## Reduction Operations

```python
x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)

s = x.sum()    # scalar: 10.0
m = x.mean()   # scalar: 2.5
```

These reduce an entire tensor to a scalar — useful for computing loss values.

## Non-linear Operations

These are covered in detail in [Activations](/LeanPass/docs/core-concepts/activations), but for reference:

```python
x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

r = x.relu()      # [0.0, 0.0, 0.0, 1.0, 2.0]
e = x.exp()       # [0.135, 0.368, 1.0, 2.718, 7.389]
l = x.log()       # nan, nan, -inf, 0.0, 0.693  (log of negative is nan)
s = x.sigmoid()   # [0.119, 0.269, 0.5, 0.731, 0.881]
```

## Operation Graph

Every operation records itself. Inspect the graph with the `_op` attribute:

```python
x = Tensor([2.0], requires_grad=True)
y = x ** 2
z = y + 1.0

print(z._op)    # "+"
print(y._op)    # "**"
print(x._op)    # "leaf" (no operation — a root node)
```

## Gradient Flow

Each operation implements its own `_backward` function. Here's what flows through each:

| Operation | `_backward` behavior |
|-----------|---------------------|
| `+` | Passes gradient unchanged to both inputs (summed if broadcast) |
| `-` | Passes gradient unchanged to first, negated to second |
| `*` | Multiplies gradient by the other input |
| `/` | Multiplies gradient by `1 / other` for first, `-self / other²` for second |
| `**` | Multiplies gradient by `2 * self` |
| `@` | Multiplies gradient by the other matrix transposed appropriately |
| `sum` | Broadcasts gradient back to original shape |
| `mean` | Broadcasts gradient divided by element count |
