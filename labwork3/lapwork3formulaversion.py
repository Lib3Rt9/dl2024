import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Example weights and inputs
w1, w2, w0 = 0.1, 0.2, 0.3  # Initialize weights
x1, x2 = [0.1, 0.5]
y = 9
learning_rate = 0.0001          # Learning rate for gradient descent

errors = []                   # Initialize list to store errors

# Training for 1000 iterations
for i in range(1000):
    # Calculate y_hat using the sigmoid function
    z = w1 * x1 + w2 * x2 + w0
    y_hat = sigmoid(z)

    # Calculate the error
    error = y - y_hat
    errors.append(abs(error))

    # Calculate the partial derivatives
    partial_w1 = error * sigmoid_derivative(z) * x1
    partial_w2 = error * sigmoid_derivative(z) * x2
    partial_w0 = error * sigmoid_derivative(z)

    # Update the weights
    w1 += learning_rate * partial_w1
    w2 += learning_rate * partial_w2
    w0 += learning_rate * partial_w0


plt.plot(errors, label='Error Trend')
plt.scatter([0, 500, 999], [errors[0], errors[500], errors[999]], color='red', label='Key Points')
plt.title('Error over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()

print("Trained weights: ", w1, w2, w0)