import pytest

from genesis import NeuralNetwork


class TestNeuralNetwork():
    def test_initialization_succeeds_with_valid_args(self):
        """Test initializing the neural network with valid input_size and neurons"""
        nn = NeuralNetwork(input_size=3, neurons=[1, 2, 3])
        assert isinstance(nn, NeuralNetwork)

    def test_raises_type_error_for_non_integer_input_size(self):
        """Test raising TypeError when input_size is not an integer"""
        with pytest.raises(TypeError):
            NeuralNetwork('a', [1, 2, 3])

        with pytest.raises(TypeError):
            NeuralNetwork({1}, [1, 2, 3])

        with pytest.raises(TypeError):
            NeuralNetwork(None, [1, 2, 3])

    def test_raises_value_error_for_non_positive_input_size(self):
        """Test raising ValueError when input_size is less than or equal to zero"""
        with pytest.raises(ValueError):
            NeuralNetwork(input_size=-1, neurons=[5, 3, 1])

        with pytest.raises(ValueError):
            NeuralNetwork(input_size=0, neurons=[5, 3, 1])

    def test_raises_type_error_for_invalid_neurons_sequence(self):
        """Test raising TypeError when neurons is not a valid sequence of integers"""
        with pytest.raises(TypeError):
            NeuralNetwork(input_size=3, neurons=None)

        with pytest.raises(TypeError):
            NeuralNetwork(input_size=3, neurons="5,3,1")

        with pytest.raises(TypeError):
            NeuralNetwork(input_size=3, neurons=[5, "three", 1])
    
    def test_raises_value_error_for_invalid_neuron_values(self):
        """Test raising ValueError when neurons contains zero or negative values"""
        with pytest.raises(ValueError):
            NeuralNetwork(input_size=3, neurons=[])

        with pytest.raises(ValueError):
            NeuralNetwork(input_size=3, neurons=[5, 0, 1])

        with pytest.raises(ValueError):
            NeuralNetwork(input_size=3, neurons=[5, -3, 1])

    def test_model_weights_are_initialized(self):
        """Test neural network weights are initialized correctly based on input size and neurons per layer"""
        nn = NeuralNetwork(3, [3, 2, 1])

        assert len(nn.weights) == 3

        assert nn.weights[0].shape == (3, 3)
        assert nn.weights[1].shape == (2, 3)
        assert nn.weights[2].shape == (1, 2)
