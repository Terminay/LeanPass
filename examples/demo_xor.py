"""XOR demo — classic minimal example of a non-linearly separable problem."""
import numpy as np
from leanpass import Tensor, nn, optim


def main():
    np.random.seed(42)

    # The four XOR cases
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float64)

    # Small 2-layer network: 2 -> 4 -> 1 with sigmoid activations
    model = nn.MLP([2, 4, 1])
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(1000):
        x = Tensor(x_data, requires_grad=False)
        y = Tensor(y_data, requires_grad=False)

        logits = model(x)
        loss = nn.binary_cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} — Loss: {loss.data:.6f}")

    # Final predictions (rounded to 0 or 1)
    preds = model(Tensor(x_data, requires_grad=False)).sigmoid()
    print("\nFinal predictions:")
    for i, (inp, out) in enumerate(zip(x_data, preds.data)):
        print(f"  {inp} -> {out[0]:.4f}  (rounded: {int(round(out[0]))})")


if __name__ == "__main__":
    main()