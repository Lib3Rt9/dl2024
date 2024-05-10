from random import seed, random, uniform
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = uniform(-1, 1)
        self.output = 0
        self.input_sum = 0

    # Sigmoid activation
    def activate(self):
        self.output = 1 / (1 + (1 / self.exp(self.input_sum + self.bias)))

    # Approximate e^x for activation function
    def exp(self, x):
        n = 1
        sum = 1
        for i in range(1, 100):
            n *= x / i
            sum += n
        return sum

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

class Link:
    def __init__(self, source_neuron, target_neuron, weight):
        self.source = source_neuron
        self.target = target_neuron
        self.weight = weight

class LayerLink:
    def __init__(self, source_layer, target_layer):
        self.links = []
        for source_neuron in source_layer.neurons:
            for target_neuron in target_layer.neurons:
                link = Link(source_neuron, target_neuron, 0)
                self.links.append(link)
                target_neuron.weights.append(link.weight)

class Network:
    def __init__(self):
        self.layers = []
        self.layer_links = []

    def initialize_weights_randomly(self):
        for layer_link in self.layer_links:
            for link in layer_link.links:
                link.weight = self.random()

    def initialize_network_from_file(self, filename):
        with open(filename, 'r') as f:
            num_layers = int(f.readline().strip())  # Read the number of layers
            print(f"Number of layers: {num_layers}")

            layer_sizes = [int(f.readline().strip()) for _ in range(num_layers)]  # Read the layer sizes
            print("Number of neurons in each layer:")
            
            for i, size in enumerate(layer_sizes):
                print(f"Layer {i+1}: {size} neuron(s)")
            
            input_size = layer_sizes[0]
            for size in layer_sizes[1:]:
                self.layers.append(Layer(size, input_size))
                input_size = size

            self.layer_links = []
            for i in range(len(self.layers) - 1):
                layer_link = LayerLink(self.layers[i], self.layers[i + 1])
                self.layer_links.append(layer_link)

    def feedforward(self, inputs):
            for i, layer in enumerate(self.layers):
                next_inputs = []
                for neuron in layer.neurons:
                    neuron.input_sum = sum(w * i for w, i in zip(neuron.weights, inputs))
                    neuron.activate()
                    next_inputs.append(neuron.output)
                inputs = next_inputs
            return inputs

    def random(self):
        seed(1)
        return random()


if __name__ == "__main__":
    neural_net = Network()

    neural_net.initialize_network_from_file("labwork4/nn_structure.txt")

    neural_net.initialize_weights_randomly()
    
    input_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    neural_net.feedforward(input_data)
    
    output_values = [neuron.output for neuron in neural_net.layers[-1].neurons]
    
    print("Output values from the last layer:")
    print(output_values)

    # Plot output values
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(output_values)), output_values, marker='o')
    plt.title('Output values from the last layer')
    plt.xlabel('Neuron index')
    plt.ylabel('Output value')
    plt.grid(True)
    plt.show()