from micrograd.nn import Neuron, Layer, MLP
from micrograd import Value


def test_neuron():
    neuron = Neuron(3)

    inputs = [
        Value(1),
        Value(2),
        Value(3),
    ]

    output = neuron(inputs)

    assert isinstance(output, Value)


def test_layer():
    layer = Layer(3, 4)

    inputs = [
        Value(1),
        Value(2),
        Value(3),
    ]

    outputs = layer(inputs)

    assert len(outputs) == 4


def test_mlp():
    model = MLP(3, [4, 4, 1])

    inputs = [
        Value(1),
        Value(2),
        Value(3),
    ]

    output = model(inputs)

    assert isinstance(output, Value)


def test_mlp_parameters():
    model = MLP(3, [4, 4, 1])

    # 3 -> 4:
    # 4 * 3 weights + 4 biases = 16
    #
    # 4 -> 4:
    # 4 * 4 weights + 4 biases = 20
    #
    # 4 -> 1:
    # 1 * 4 weights + 1 bias = 5
    #
    # Total = 41

    assert len(model.parameters()) == 41
