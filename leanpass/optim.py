import numpy as np


class SGD:
    """Simple stochastic gradient descent optimizer."""

    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = lr

    def step(self):
        """Apply a plain gradient descent update to each parameter."""
        for param in self.params:
            if param.grad is None:
                continue
            param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        """Zero out gradients so the next backward pass starts clean."""
        for param in self.params:
            param.grad = np.zeros_like(param.data)


class Adam:
    """Adam optimizer with bias-corrected moment estimates."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        """Update each parameter using Adam's adaptive moment estimates."""
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero out gradients for all tracked parameters."""
        for param in self.params:
            param.grad = np.zeros_like(param.data)
