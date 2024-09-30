import numpy as np
from typing import Sequence 

class NeuralNetwork:
    def __init__(self, input_size: int, neurons: Sequence[int]) -> None:
        """Initialize a neural network with the specified number of neurons

        The `neurons` parameter dictates both the number of layer and the number 
        of neurons per layer. The value found at each index in `neurons` indicates
        the number of neurons in the layer at that given index. e.g. Specifying a 
        `neurons` value of `[5, 3, 1]` would create a neural network with three hidden 
        layers where the first layer (layer at index 0) has 5 neurons, followed by 
        a layer with 3 neurons and a final layer with 1 neuron.

        both input_size and any values specified in neurons must be postivive non zero
        integers. A ValueError will be thrown for invalid values.
        """

        if input_size is None or not isinstance(input_size, int):
            raise TypeError("Parameter 'input_size' must be an integer")
        if input_size <= 0:
            raise ValueError("Pamater 'input_size' must be greater than zero")

        if neurons is None \
            or not isinstance(neurons, Sequence) \
            or any((not isinstance(x, int) for x in neurons)):
            raise TypeError("Parameter 'neurons' must be a list of integers")
        if len(neurons) == 0 or any((x <= 0 for x in neurons)):
            raise ValueError("Parameter 'neurons' must contain one or more value \
            with each value must be greater than zero")

        self._weights = []
        # Add the first layers weights.
        self._weights.append(np.random.random(size=(neurons[0], input_size)))
        if len(neurons) > 1:
            # Add the weights for the remaining layers.
            self._weights.extend([
                np.random.random(size=(neurons[i], neurons[i-1])) for i in range(1, len(neurons))
            ]) 

    def __call__(self, inputs):
        """Computes the output by passing the specified inputs through the neural network.

        `inputs` is expected to be in the shape [batch_size, n_features].

        A ValueError is raised if the dims of the input do not match the expected input size 
        defined when the model was initialized"""
        outputs = inputs
        for weights in self.weights:
            # Iterate through the weights of each layer and compute the layer's output'
            outputs = np.matmul(outputs, weights.T) 
        return outputs


    @property
    def weights(self):
        """The weights property."""
        return self._weights
    
