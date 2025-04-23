from get_data import get_data
from layers.tanh import Tanh
from layers.ReLU import ReLU
from layers.leaky_ReLU import LeakyReLU
from layers.dense_layer import Dense
from network import Network
import numpy as np
import matplotlib.pyplot as plt
from error import mse, mse_gradient

values = get_data()

range_output = 2
time_step = 10
range_input = time_step
current_index = time_step
learning_rate = 0.005
episodes = 50
attenuation_real_value = 50
additional_neurons = 10

# for leaky relu
alpha = 0.05

layers = [
    Dense(range_input, range_input + additional_neurons),
    Tanh(),
    Dense(range_input + additional_neurons, range_input + additional_neurons),
    Tanh(),
    Dense(range_input + additional_neurons, range_input + additional_neurons),
    Tanh(),
    Dense(range_input + additional_neurons, range_input + additional_neurons),
    Tanh(),
    Dense( range_input + additional_neurons, 1),
    Tanh()
]

my_network = Network(layers, mse, mse_gradient)

output_simplifier = Tanh()

itteration_aux = []
predicted = []
real = []
percentage_grouth_values = []

for e in range(episodes):
    current_index = time_step
    i = 0
    error_sum = 0
    while (current_index + time_step + range_output < len(values)):
        # for debugging
        inputs = np.reshape([float(i) for i in values[current_index - time_step: current_index]], (range_input, 1))
        output = my_network.forward_propagation(inputs)
        
        # calculating real value
        current_value = float(values[current_index])

        next_time_step = current_index + time_step
        output_values = [float(i) for i in values[next_time_step - range_output:next_time_step + range_output]]
        avg_output = sum(output_values) / len(output_values)

        percentage_grouth = ((avg_output - current_value) / current_value) * 100

        real_output = output_simplifier.forwards(percentage_grouth / attenuation_real_value)

        # training
        i+=1
        error_sum +=my_network.train_iteration_mse(inputs, real_output, learning_rate)

        # end of loop
        current_index += 1

        # getting debug information
        if e == episodes - 1:
            itteration_aux.append(i)
            real.append(real_output)
            predicted.append(output[0][0])
            percentage_grouth_values.append(percentage_grouth)

        
    
    print(f"Avg error from episode {e}:", error_sum / i)

start_plot = 100
end_plot = 300

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(itteration_aux[start_plot:end_plot], real[start_plot:end_plot])
ax1.set_title("Real values")
ax2 = f2.add_subplot(111)
ax2.plot(itteration_aux[start_plot:end_plot], predicted[start_plot:end_plot])
ax2.set_title("Predicted values")
ax3 = f3.add_subplot(111)
ax3.plot(itteration_aux[start_plot:end_plot], percentage_grouth_values[start_plot:end_plot])
ax3.set_title("Percentege grouth")
plt.show()