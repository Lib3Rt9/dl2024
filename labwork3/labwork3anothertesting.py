import math
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def loss(y, y_hat):
    loss = 0
    for y_true, y_pred in zip(y, y_hat):
        loss += -y_true * math.log(y_pred) - (1 - y_true) * math.log(1 - y_pred)
    return loss / len(y)

def logistic_regression(X, y, num_iterations, learning_rate):
    N, D = len(X), len(X[0])
    w = [0] * (D + 1)  # Initialize weights with zeros
    losses = []  # Initialize losses list

    for iteration in range(num_iterations):
        y_hat = [sigmoid(sum(w[i] * x[i - 1] for i in range(1, D + 1)) + w[0]) for x in X]
        dw = [0] * (D + 1)

        # Compute gradients
        for i in range(D):
            dw[i + 1] = sum((y_hat[j] - y[j]) * X[j][i] for j in range(N))
        dw[0] = sum(y_hat[j] - y[j] for j in range(N))
        dw = [dw[i] / N for i in range(D + 1)]

        # Update weights
        w = [w[i] - learning_rate * dw[i] for i in range(D + 1)]

        # Compute and store loss
        loss = loss(y, y_hat)
        losses.append(loss)

    return w, losses

# Example usage
X = [[1, 2], [3, 4], [5, 6]]  # Feature matrix
y = [0, 1, 0]  # Labels

num_iterations = 1000
learning_rate = 0.01
weights, losses = logistic_regression(X, y, num_iterations, learning_rate)
print(f"Learned weights: {weights}")

# Plot loss curve
plt.plot(range(num_iterations), losses)
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss Curve")
plt.show()