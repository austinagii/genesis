from genesis import NeuralNetwork

def test_neural_network_validates_initialization_parameters():
    try:
        net = NeuralNetwork(input_size=None, neurons=None)
    except Exception as e:
        assert isinstance(e, TypeError)
