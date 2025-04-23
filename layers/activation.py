from .layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forwards(self, input):
        self.input = input
        # print("Input: ", self.input.T, "Output: ", self.activation(input).T) --> debug
        return self.activation(input)
    
    def backwards(self, output_gradient, learning_rate):
        #print("Output_gradient: ", output_gradient.T, "Input_gradient: ", self.activation_prime(self.input).T)
        return np.multiply(output_gradient, self.activation_prime(self.input))