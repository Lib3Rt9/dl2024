import math
import random
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, file_path):
        self.load_from_file(file_path)
        self.initialize_weights_biases()
        
    def load_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            num_layers = int(lines[0])
            self.layers = [int(lines[i + 1]) for i in range(num_layers)]

    def initialize_weights_biases(self):
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            self.weights.append([[random.random() for _ in range(self.layers[i])] for _ in range(self.layers[i + 1])])
            self.biases.append([random.random() for _ in range(self.layers[i + 1])])

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        activations = [inputs]

        for i in range(self.num_layers - 1):
            layer_activations = []
            for j in range(self.layers[i + 1]):
                weighted_sum = sum(activations[-1][k] * self.weights[i][j][k] for k in range(self.layers[i])) + self.biases[i][j]
                layer_activations.append(self.sigmoid(weighted_sum))
            activations.append(layer_activations)

        return activations

    def backpropagate(self, inputs, targets):
        activations = self.feedforward(inputs)
        output_deltas = [activations[-1][i] * (1 - activations[-1][i]) * (targets[i] - activations[-1][i]) for i in range(self.layers[-1])]
        deltas = [output_deltas]

        for i in range(self.num_layers - 2, 0, -1):
            layer_deltas = []
            for j in range(self.layers[i]):
                error = sum(deltas[-1][k] * self.weights[i][k][j] for k in range(self.layers[i + 1]))
                layer_deltas.append(activations[i][j] * (1 - activations[i][j]) * error)
            deltas.append(layer_deltas)

        deltas.reverse()

        for i in range(self.num_layers - 1):
            for j in range(self.layers[i + 1]):
                for k in range(self.layers[i]):
                    self.weights[i][j][k] += self.learning_rate * deltas[i][j] * activations[i][k]
                self.biases[i][j] += self.learning_rate * deltas[i][j]

    def train(self, inputs, targets, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(inputs)):
                self.backpropagate(inputs[i], targets[i])
                activations = self.feedforward(inputs[i])
                loss = sum((targets[i][j] - activations[-1][j])**2 for j in range(len(targets[i])))
                epoch_loss += loss
            losses.append(epoch_loss / len(inputs))
        return losses

# Example usage for XOR problem
nn = NeuralNetwork('labwork4/nn_structure.txt')

# Training data
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]

# Train the neural network and get losses
losses = nn.train(inputs, targets, epochs=10000)

print("XOR problem results after training:")
print(f"Input: [0, 0] Output: {nn.feedforward([0, 0])[-1]}")
print(f"Input: [0, 1] Output: {nn.feedforward([0, 1])[-1]}")
print(f"Input: [1, 0] Output: {nn.feedforward([1, 0])[-1]}")
print(f"Input: [1, 1] Output: {nn.feedforward([1, 1])[-1]}")

# Plot the losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()