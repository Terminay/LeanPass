---
sidebar_position: 2
---

# How Autodiff Works

LeanPass uses **reverse-mode automatic differentiation** (backpropagation) to compute gradients. Here's what that actually means.

## The Computation Graph

Every operation you perform on a `Tensor` records itself in a directed acyclic graph (DAG). The tensors are nodes, and the operations are edges.

```python
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a * b
d = c + 1.0
```

This builds a graph like:

```
a ──┐
    ├── [*] ── c ──┐
b ──┘              ├── [+] ── d ── 1.0
```

Each `Tensor` stores:
- `data` — the computed value
- `grad` — the gradient accumulated during backward
- `_prev` — the parent tensors
- `_op` — the operation that produced this tensor
- `_backward` — a function that computes the local gradient

## Forward Pass

During the forward pass, each operation computes its output value and stores a `_backward` function that knows how to propagate gradients through that specific operation.

```python
# Internally, multiplication looks like:
def __mul__(self, other):
    out = Tensor(self.data * other.data, ...)
    
    def _backward():
        if self.requires_grad:
            self.grad += other.data * out.grad
        if other.requires_grad:
            other.grad += self.data * out.grad
    
    out._backward = _backward
    return out
```

## Backward Pass

When you call `d.backward()`, LeanPass:

1. **Topologically sorts** the graph — ordering nodes so parents come before children
2. **Seeds** the gradient of the output tensor to `1.0`
3. **Walks in reverse** — calling each node's `_backward` function to propagate gradients to its parents

```python
# Simplified backward algorithm
def backward(self):
    topo = topological_sort(self)
    self.grad = np.ones_like(self.data)
    
    for node in reversed(topo):
        node._backward()
```

## The Chain Rule

At each node, the local gradient is computed using the **chain rule**. For a composition:

```
z = f(g(x))
dz/dx = dz/dg * dg/dx
```

Or for a multivariable function:

```
z = x * y
dz/dx = y      (holding y constant)
dz/dy = x      (holding x constant)
```

## Example Trace

```python
x = Tensor([2.0], requires_grad=True)
y = x ** 2
z = y + 3.0
z.backward()
```

**Forward:**
- `x = 2.0`
- `y = 4.0`
- `z = 7.0`

**Backward:**
- `z.grad = 1.0` (seed)
- `y.grad += 1.0 * 1.0 = 1.0` (d_z/d_y = 1.0 from `+`)
- `x.grad += 1.0 * (2 * 2.0) = 4.0` (d_y/d_x = 2*x from `** 2`)

Result: `dz/dx = 4.0` — which matches the analytical derivative `2 * 2.0 = 4.0`.
