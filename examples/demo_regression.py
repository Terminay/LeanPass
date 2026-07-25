"""Regression demo — fit a sine wave with a small MLP."""
import numpy as np
from leanpass import Tensor, nn, optim


def main():
    np.random.seed(42)

    # Generate noisy sine wave data
    x_data = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_data = np.sin(x_data) + np.random.randn(*x_data.shape) * 0.1

    model = nn.MLP([1, 16, 16, 1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):
        x = Tensor(x_data, requires_grad=False)
        y = Tensor(y_data, requires_grad=False)

        pred = model(x)
        loss = nn.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} — MSE Loss: {loss.data:.6f}")

    # Show sample predictions vs ground truth
    x_test = np.linspace(-3, 3, 10).reshape(-1, 1)
    y_true = np.sin(x_test)
    preds = model(Tensor(x_test, requires_grad=False))
    print("\nSample predictions (x, true, predicted):")
    for i in range(len(x_test)):
        print(f"  x={x_test[i,0]:.2f}  true={y_true[i,0]:.4f}  pred={preds.data[i,0]:.4f}")


if __name__ == "__main__":
    main()