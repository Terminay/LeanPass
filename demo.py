import numpy as np

from leanpass import nn, optim, Tensor


def cross_entropy(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute the average cross-entropy loss for one-hot labels."""
    probs = predictions.softmax(axis=1)
    log_probs = probs.log()
    return -(targets * log_probs).sum() / targets.data.shape[0]


def main():
    np.random.seed(1)

    # Create synthetic 3-class classification data with simple separable clusters.
    num_examples = 300
    num_classes = 3
    x_data = np.vstack([
        np.random.randn(num_examples // num_classes, 2) + np.array([2.0, 0.0]),
        np.random.randn(num_examples // num_classes, 2) + np.array([-2.0, 0.0]),
        np.random.randn(num_examples // num_classes, 2) + np.array([0.0, 2.0]),
    ])
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    y_data = np.eye(num_classes)[labels]

    model = nn.MLP([2, 16, 16, num_classes])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for step in range(200):
        x = Tensor(x_data, requires_grad=False)
        y = Tensor(y_data, requires_grad=False)

        logits = model(x)
        loss = cross_entropy(logits, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step:03d} - Loss: {loss.data:.6f}")

    # Demonstrate prediction and graph visualization on a sample point.
    sample = Tensor(np.array([[1.0, 1.0]]), requires_grad=False)
    sample_logits = model(sample)
    sample_probs = sample_logits.softmax(axis=1)

    print("\nSample input [1.0, 1.0] softmax probabilities:")
    print(sample_probs.data)
    print("Predicted class:", np.argmax(sample_probs.data, axis=1)[0])

    # Visualize the final sample loss graph to inspect computation structure.
    target = Tensor(np.array([[1.0, 0.0, 0.0]]), requires_grad=False)
    sample_loss = cross_entropy(sample_logits, target)
    print("\nComputation graph for sample loss:")
    print(sample_loss.visualize())


if __name__ == "__main__":
    main()
