
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Generate Synthetic Dataset
# ==========================================

np.random.seed(42)

# Class 0, using a Gaussian distribution centered at (2, 2) and broadcasted to 100 samples
X0 = np.random.randn(100, 2) + np.array([2, 2])

# Class 1
X1 = np.random.randn(100, 2) + np.array([6, 6])

# Combine features
X = np.vstack((X0, X1))

# Generate labels
y = np.vstack((
    np.zeros((100, 1)),
    np.ones((100, 1))
))

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# 2. Visualize Dataset
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X0[:, 0], X0[:, 1],
    color="blue",
    label="Class 0"
)

plt.scatter(
    X1[:, 0], X1[:, 1],
    color="red",
    label="Class 1"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Binary Classification Dataset")
plt.legend()
plt.show()


# ==========================================
# 3. Sigmoid Function
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ==========================================
# 4. Initialize Parameters
# ==========================================

np.random.seed(42)

W = np.random.randn(2, 1)
b = np.zeros((1, 1))


# ==========================================
# 5. Forward Propagation
# ==========================================

def forward_propagation(X, W, b):

    z = np.dot(X, W) + b

    y_pred = sigmoid(z)

    return y_pred


# ==========================================
# 6. Binary Cross-Entropy Loss
# ==========================================

def binary_cross_entropy(y, y_pred):

    epsilon = 1e-15

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y * np.log(y_pred)
        + (1 - y) * np.log(1 - y_pred)
    )

    return loss


# ==========================================
# 7. Compute Gradients
# ==========================================

def compute_gradients(X, y, y_pred):

    N = X.shape[0]

    dw = (1 / N) * np.dot(X.T, (y_pred - y))

    db = (1 / N) * np.sum(y_pred - y)

    return dw, db


# ==========================================
# 8. Train Model Using Gradient Descent
# ==========================================

def train(X, y, W, b, learning_rate, epochs):

    losses = []

    for epoch in range(epochs):

        # Forward propagation
        y_pred = forward_propagation(X, W, b)

        # Calculate gradients
        dw, db = compute_gradients(X, y, y_pred)

        # Update parameters
        W = W - learning_rate * dw
        b = b - learning_rate * db

        # Calculate loss after update
        y_pred = forward_propagation(X, W, b)
        loss = binary_cross_entropy(y, y_pred)

        losses.append(loss)

    return W, b, losses


# ==========================================
# 9. Train the Model
# ==========================================

W, b, losses = train(
    X, y, W, b,
    learning_rate=0.01,
    epochs=1000
)


final_pred = forward_propagation(X, W, b)

final_loss = binary_cross_entropy(y, final_pred)

print("Final Loss:", final_loss)

# ==========================================
# 10. Plot Training Loss
# ==========================================
# Generate grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# Predict probabilities for grid points
grid = np.c_[xx.ravel(), yy.ravel()]

probs = forward_propagation(grid, W, b)

probs = probs.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))

plt.contourf(
    xx, yy, probs,
    levels=[0, 0.5, 1],
    alpha=0.2,
    colors=["blue", "red"]
)

plt.contour(
    xx, yy, probs,
    levels=[0.5],
    colors="black"
)

plt.scatter(X0[:, 0], X0[:, 1], color="blue", label="Class 0")
plt.scatter(X1[:, 0], X1[:, 1], color="red", label="Class 1")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()
# plt.figure(figsize=(8, 5))

# plt.plot(losses)

# plt.xlabel("Epoch")
# plt.ylabel("Binary Cross-Entropy Loss")
# plt.title("Training Loss Curve")

# plt.show()

learning_rates = [0.001, 0.01, 0.1, 1.0]

plt.figure(figsize=(10, 6))

for lr in learning_rates:

    # Same initialization for every experiment
    W_init = np.random.RandomState(42).randn(2, 1)
    b_init = np.zeros((1, 1))

    W_trained, b_trained, losses = train(
        X, y,
        W_init.copy(),
        b_init.copy(),
        learning_rate=lr,
        epochs=1000
    )

    plt.plot(losses, label=f"LR = {lr}")

    print(f"LR = {lr}, Final Loss = {losses[-1]:.6f}")

plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid()
plt.show()

def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-15)
    recall = TP / (TP + FN + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)

    return accuracy, precision, recall, f1

# Convert probabilities into binary predictions
y_pred = (final_pred >= 0.8).astype(int)

accuracy, precision, recall, f1 = calculate_metrics(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# ==========================================
# 12. Plot Confusion Matrix
# ==========================================

TP = np.sum((y == 1) & (y_pred == 1))
FP = np.sum((y == 0) & (y_pred == 1))
TN = np.sum((y == 0) & (y_pred == 0))
FN = np.sum((y == 1) & (y_pred == 0))

cm = np.array([
    [TN, FP],
    [FN, TP]
])

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Class 0", "Class 1"])
plt.yticks([0, 1], ["Class 0", "Class 1"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )

plt.colorbar()
plt.show()


# ==========================================
# 13. Plot Classification Metrics
# ==========================================

metric_names = ["Accuracy", "Precision", "Recall", "F1"]
metric_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))

plt.bar(metric_names, metric_values)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Classification Performance")

for i, value in enumerate(metric_values):
    plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

plt.show()



# ==========================================
# 14. Threshold Comparison
# ==========================================

thresholds = [0.2, 0.5, 0.8]

for threshold in thresholds:

    y_pred_threshold = (final_pred >= threshold).astype(int)

    acc, prec, rec, f1 = calculate_metrics(
        y, y_pred_threshold
    )

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")