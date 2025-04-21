from .layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_length, output_length):
        self.weigths = np.random.randn(output_length, input_length)
        self.biases = np.random.randn(output_length, 1)

    def forwards(self, input):
        self.input = input

        return np.dot(self.weigths, self.input) + self.biases
    
    def backwards(self, output_gradient, learning_rate):
        self.biases -= learning_rate * output_gradient

        weights_aux = self.weigths.copy()
        self.weigths -= learning_rate * np.dot(output_gradient, self.input.T)

        input_gradient = np.dot(weights_aux.T, output_gradient)

        return input_gradient