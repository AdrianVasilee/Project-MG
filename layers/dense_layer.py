from .layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_length, output_length):
        self.weights = np.random.randn(output_length, input_length)
        self.biases = np.random.randn(output_length, 1)

        self.print_info()

    def forwards(self, input):
        self.input = input

        return np.dot(self.weights, self.input) + self.biases
    
    def backwards(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weight_gradient

        self.biases -= learning_rate * output_gradient

        input_gradient = np.dot(self.weights.T, output_gradient)

        return input_gradient
    
    def print_info(self):
        print("Initializing dense layer with:")
        print("- weigth shape: ", self.weights.shape)
        print("- bias shape: ", self.biases.shape)