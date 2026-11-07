
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(42)

X = 2 * np.random.rand(100, 1)
y = 3 * X + 5 + np.random.randn(100, 1)

n = len(X)
epochs = 1000

# 2. Define learning rates
learning_rates = [0.1, 0.5, 1.0, 2.0, 5.0]

# Use identical initial parameters for fair comparison
initial_w = np.array([0.5])
initial_b = np.array([0.5])

# Store experimental results
results = {}

# 3. Train models with different learning rates
for lr in learning_rates:

    w = initial_w.copy()
    b = initial_b.copy()

    loss_history = []

    for epoch in range(epochs):

        # Forward propagation
        y_pred = X * w + b

        # Calculate loss
        loss = np.sum((y_pred - y) ** 2) / (2 * n)
        loss_history.append(loss)

        # Calculate gradients
        error = y_pred - y
        dw = np.sum(X * error) / n
        db = np.sum(error) / n

        # Update parameters
        w = w - lr * dw
        b = b - lr * db

    results[lr] = {
        "w": w.copy(),
        "b": b.copy(),
        "loss_history": loss_history
    }

    print(f"\nLearning rate: {lr}")
    print(f"Final w: {w}")
    print(f"Final b: {b}")
    print(f"Final loss: {loss_history[-1]:.6f}")

# 4. Plot loss curves
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    plt.plot(
        results[lr]["loss_history"],
        label=f"Learning rate = {lr}"
    )

plt.xlabel("Epoch")
plt.ylabel("MSE Loss (half)")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.show()
