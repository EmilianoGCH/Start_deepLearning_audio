import numpy as np
import math
import random


# We start a MLP as an object
class MLP:  
    """
    This is our firs Multi Layer Perceptron (MLP) from scratch in Python
    """

    def __init__(self, inputs=3, num_hidden=[5,3], num_outputs=2):
        self.inputs = inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
    
        layers = [self.inputs] + self.num_hidden + [self.num_outputs]

        weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        
        self.weights = weights

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # iterate through the network layers
        for w in self.weights:

            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

        # return output layer activation
        return activations

    def _sigmoid(self, h):
        return 1 / (1 + np.exp(-h))



if __name__ == "__main__":
    A = MLP()

    inputs = np.random.rand(A.inputs)
    # inputs = [0.5,0.25,0.75]
    outputs = A.forward_propagate(inputs)

    print(f'Result {outputs}')