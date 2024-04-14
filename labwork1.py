# The original function: x^2
def f(x):
    return x**2

# The derivative of the function: 2x
def df(x):
    return 2*x

# Gradient descent
def gradient_descent(init_x, learning_rate, num_iters):

    print(f"{'Time'}{'x'}{'f(x)'}")
    for i in range(num_iters):

        grad = df(init_x)
        x = init_x - learning_rate * grad
        print(f"{i+1:<10}{x:<10.2f}{f(x):<10.2f}")

    return x

# Initialize parameters
initial_x = 8
learning_rate = 0.1
num_iterations = 10

# Run gradient descent
min_x = gradient_descent(initial_x, learning_rate, num_iterations)
print(f"\nMinimum value occurs at x = {min_x:.2f}")
