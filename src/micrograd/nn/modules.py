import random

from micrograd.engine import Value


class Neuron:
    """A single fully-connected neuron."""

    def __init__(self, nin):
        self.weights = [
            Value(random.uniform(-0.1, 0.1))
            for _ in range(nin)
        ]

        self.bias = Value(random.uniform(-0.1, 0.1))

    def __call__(self, inputs):
        activation = sum(
            weight * input_value
            for weight, input_value in zip(self.weights, inputs)
        ) + self.bias

        return activation.tanh()

    def parameters(self):
        return self.weights + [self.bias]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class Layer:
    """A fully-connected layer containing multiple neurons."""

    def __init__(self, nin, nout):
        self.neurons = [
            Neuron(nin)
            for _ in range(nout)
        ]

    def __call__(self, inputs):
        outputs = [
            neuron(inputs)
            for neuron in self.neurons
        ]

        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class MLP:
    """
    Multi-Layer Perceptron.

    Example:

        model = MLP(3, [4, 4, 1])

    creates:

        3 inputs
          ↓
        4 neurons
          ↓
        4 neurons
          ↓
        1 neuron
    """

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def parameters(self):
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()
