import numpy as np

from .tensor import Tensor


class Module:
    """A base class for layers and model containers."""

    def parameters(self):
        """Collect all trainable Tensor parameters recursively."""
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def zero_grad(self):
        """Reset gradients before each optimization step."""
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)


class Linear(Module):
    """A single fully connected layer: y = x @ W + b."""

    def __init__(self, in_features, out_features):
        scale = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(
            np.random.uniform(-scale, scale, size=(in_features, out_features)),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class MLP(Module):
    """A simple feedforward network with ReLU activations."""

    def __init__(self, layer_sizes):
        self.layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(Linear(in_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Propagate the input through each layer and apply ReLU except last."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = x.relu()
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        """Return parameters from every Linear layer in the network."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss used for regression training."""
    return ((prediction - target) ** 2).mean()
