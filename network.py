class Network:
    def __init__(self, layers, error, error_gradient):
        self.layers = layers
        self.error_func = error
        self.error_gradient_func = error_gradient

    def forward_propagation(self, input):
        output = input
        for l in self.layers:
            output = l.forwards(output)

        return output
    
    def backward_propagation(self, output_gradient, learning_rate):
        for l in reversed(self.layers):
            output_gradient = l.backwards(output_gradient, learning_rate)
        
        return output_gradient

    def train_iteration_mse(self, input, real_values, learning_rate):
        output = self.forward_propagation(input)

        error_gradient = self.error_gradient_func(output, real_values)
        self.backward_propagation(error_gradient, learning_rate)

        error = self.error_func(output, real_values)
        return error
