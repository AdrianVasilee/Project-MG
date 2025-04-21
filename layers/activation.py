from .layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forwards(self, input):
        self.input = input
        return self.activation(input)
    
    def backwards(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))