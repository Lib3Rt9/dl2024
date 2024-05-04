import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(features, weights):
    z = sum(weight * feature for weight, feature in zip(weights, features))
    return sigmoid(z)

def compute_gradient(features, labels, predictions):
    differences = [prediction - label for prediction, label in zip(predictions, labels)]
    gradient = []
    for j in range(len(features[0])):
        gradient.append(sum(difference * features[i][j] for i, difference in enumerate(differences)))
    return gradient

def update_weights(features, labels, weights, lr):
    predictions = [predict(feature, weights) for feature in features]
    gradient = compute_gradient(features, labels, predictions)
    for i in range(len(weights)):
        weights[i] -= lr * gradient[i] / len(features)
    return weights

def train(features, labels, lr, iters):
    weights = [0.0 for _ in range(len(features[0]))]
    for _ in range(iters):
        weights = update_weights(features, labels, weights, lr)
    return weights

# data
X = [[0.2, 0.7], [0.3, 0.3], [0.4, 0.5], [0.5, 0.1]]
y = [1, 0, 1, 0]

learning_rate = 0.0001
iterations = 10000
trained_weights = train(X, y, learning_rate, iterations)
print("Trained weights:", trained_weights)

# Plot
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = predict([X1[i, j], X2[i, j]], trained_weights)

plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Z, levels=[0.0, 0.5, 1.0], cmap='rainbow')
plt.scatter([x[0] for x in X if y[X.index(x)] == 0], [x[1] for x in X if y[X.index(x)] == 0], c='b', label='Class 0')
plt.scatter([x[0] for x in X if y[X.index(x)] == 1], [x[1] for x in X if y[X.index(x)] == 1], c='r', label='Class 1')
plt.legend()
plt.show()