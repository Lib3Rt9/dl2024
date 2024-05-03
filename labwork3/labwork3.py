import math

w0, w1, w2 = 0, 0, 0
learning_rate = 0.01

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def loss(y, y_hat):
    return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

def cost_function(y, y_hat, N):
    return -1/N * sum([loss(y[i], y_hat[i]) for i in range(N)])


X1 = [0.1, 0.3, 0.5, 0.7, 0.9]  # feature 1
X2 = [0.2, 0.4, 0.6, 0.8, 1.0]  # feature 2
Y = [0, 1, 0, 1, 0]   # target variable
N = len(Y)  # number of training examples

# Check if N is zero
if N == 0:
    print("Error: No training examples found.")

else:
    for epoch in range(1000):
        y_hat = [sigmoid(w1 * X1[i] + w2 * X2[i] + w0) for i in range(N)]
        cost = cost_function(Y, y_hat, N)
        
        # Update weights using gradient descent
        w0 -= learning_rate * (1/N) * sum([(y_hat[i] - Y[i]) for i in range(N)])
        w1 -= learning_rate * (1/N) * sum([(y_hat[i] - Y[i]) * X1[i] for i in range(N)])
        w2 -= learning_rate * (1/N) * sum([(y_hat[i] - Y[i]) * X2[i] for i in range(N)])

        # Print weights and cost
        print(f"Epoch {epoch+1}")
        print(f"w0: {w0}, w1: {w1}, w2: {w2}")
        print(f"Cost: {cost}")