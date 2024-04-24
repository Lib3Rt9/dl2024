from math import exp, log

# y_i == y
# y_hat_i == y_hat
def sigmoid(z):
    return 1/(1 + exp(-z))

def y_hat(x1, x2, w0, w1, w2):
    return sigmoid(w1 * x1 + w2 * x2 + w0)

def loss_func(y, y_hat):
    L_i = -(y * log(y_hat)) + (1 - y)*(log(1 - y))
    return L_i

def loss_func_all_datap(x, y, y_hat):
    J = -(1/N) * sum((y * log(y_hat)) + ((1 - y) * log(1 - y_hat)))
    return J

# partial derivative of loss function
def loss_func_pder(x1, x2, y, y_hat):
    dw1 = x1 * (y_hat - y)
    dw2 = x2 * (y_hat - y)
    dw0 = y_hat - y
    return dw1, dw2, dw0

def loss_func_pder_all_datap(N, i, y, y_hat):
    
