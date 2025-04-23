from layers.dense_layer import Dense
from layers.tanh import Tanh
from layers.ReLU import ReLU
from layers.leaky_ReLU import LeakyReLU
from debug import *
import numpy as np
from error import *

# for leaky relu
alpha = 0.05

network = [
    Dense(2, 2),
    Tanh(),
    Dense(2, 1)
]

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 0.5, 0]

learning_rate = 0.1
episodes = 10000

inputs = np.array(inputs, dtype=np.float32)
inputs = np.reshape(inputs, (4, 2, 1))

outputs = np.array(outputs, dtype=np.float32)
outputs = np.reshape(outputs, (4, 1, 1))


for e in range(episodes):
    print("Episode:", e)    
    for x, y in zip(inputs, outputs):
        # forward method
        input_aux = x
        for n in network:
            input_aux = n.forwards(input_aux)

        # Error
        print("Error: ", mse(input_aux, y))

        # backward method
        output_gradient = mse_gradient(input_aux, y)
        for n in reversed(network):
            output_gradient = n.backwards(output_gradient, learning_rate)

    #print(learning_rate)
    #learning_rate *= 0.99999


for n in network:
    if isinstance(n, Dense):
        print(f'''Dense layer:
- Weights
{n.weigths}                 
- Weight shape:  {n.weigths.shape}

- Biases: {n.biases}
- Biases shape: {n.biases.shape}

''')

# show 3d representation of the probability distribution
plot_network(20, network)