from layers.dense_layer import Dense
from layers.tanh import Tanh
import numpy as np
from error import *
from get_data import get_data

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 2),
    Tanh(),
    Dense(2, 1),
    Tanh()
]

nr_io = 4
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]

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

