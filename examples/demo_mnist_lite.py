"""MNIST-lite demo — a small, CPU-friendly digit classification example."""
import numpy as np
from leanpass import Tensor, nn, optim


def load_digits():
    """Load sklearn's digits dataset (8x8 images, 10 classes)."""
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        digits = load_digits()
        X = digits.data / 16.0  # scale to [0, 1]
        y = np.eye(10)[digits.target]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except ImportError:
        print("sklearn not available; using synthetic digit-like data.")
        return _synthetic_data()


def _synthetic_data():
    """Generate synthetic 64-dim data with 10 classes as fallback."""
    np.random.seed(42)
    N = 500
    X = np.random.randn(N, 64)
    y = np.eye(10)[np.random.randint(0, 10, N)]
    split = int(N * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def accuracy(logits, targets):
    preds = np.argmax(logits, axis=1)
    true = np.argmax(targets, axis=1)
    return np.mean(preds == true)


def main():
    X_train, X_test, y_train, y_test = load_digits()
    model = nn.MLP([64, 32, 10])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[idx]
        y_train_shuffled = y_train[idx]

        x = Tensor(X_train_shuffled, requires_grad=False)
        y = Tensor(y_train_shuffled, requires_grad=False)

        logits = model(x)
        loss = nn.cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            train_acc = accuracy(logits.data, y_train_shuffled)
            print(f"Epoch {epoch:03d} — Loss: {loss.data:.6f}  Train Acc: {train_acc:.4f}")

    # Final evaluation on test set
    x_test = Tensor(X_test, requires_grad=False)
    y_test_t = Tensor(y_test, requires_grad=False)
    test_logits = model(x_test)
    test_loss = nn.cross_entropy_loss(test_logits, y_test_t)
    test_acc = accuracy(test_logits.data, y_test)
    print(f"\nTest set — Loss: {test_loss.data:.6f}  Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()