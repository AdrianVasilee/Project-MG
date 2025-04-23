from get_data import get_data
from layers.tanh import Tanh
from layers.ReLU import ReLU
from layers.leaky_ReLU import LeakyReLU
from layers.dense_layer import Dense
from network import Network
import numpy as np
import matplotlib.pyplot as plt
from error import mse, mse_gradient
import glob
import random

files = glob.glob("stocks/*.csv")

range_output = 2
time_step = 10
range_input = time_step
current_index = time_step
learning_rate = 0.01
episodes = 50
attenuation_real_value = 70
additional_neurons = 10

# for leaky relu
alpha = 0.1

layers = [
    Dense(range_input, 32),
    LeakyReLU(alpha),
    Dense(32, 16),
    LeakyReLU(alpha),
    Dense( 16, 8),
    ReLU(),
    Dense( 8, 1),
    Tanh()
]

for l in layers:
    if isinstance(l, Dense): print(l.weights.shape)

my_network = Network(layers, mse, mse_gradient)

output_simplifier = Tanh()

itteration_aux = []
predicted = []
real = []
percentage_grouth_values = []

for e in range(episodes+1):
    for f in files[:1]:
        current_index = time_step
        i = 0
        error_sum = 0
        values = get_data(file=f)

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


            # ---------Start of test side --------
            '''
            output = inputs
            print("Forward method-------")
            for l in my_network.layers:
                if isinstance(l, Dense):
                    print("Dense Layer....")

                if isinstance(l, LeakyReLU):
                    print("Leaky ReLU...")
                    
                output = l.forwards(output)
                print("Output: ", [o[0] for o in output])
                print()
            

            error = mse(output, real_output)
            print("Error: ", error)

            print("Backward method-------")
            output_gradient = mse_gradient(output, real_output)
            print("Error gradient with respect to the first output: ", [o[0] for o in output_gradient])
            print()
            for l in reversed(my_network.layers):
                if isinstance(l, Dense):
                    print("Dense Layer....")

                if isinstance(l, LeakyReLU):
                    print("Leaky ReLU...")


                output_gradient = l.backwards(output_gradient, learning_rate)
                print("Gradient: ", [o[0] for o in output_gradient])

                if isinstance(l, Dense):
                    l.print_info()

            if i == 0: exit()
            '''
            #---------End of test side ---------

            # training
            i+=1
            error_sum +=my_network.train_iteration_mse(inputs, real_output, learning_rate)

            # end of loop
            current_index += 1

            # getting debug information
            itteration_aux.append(i)
            real.append(real_output)
            predicted.append(output[0][0])
            percentage_grouth_values.append(percentage_grouth)
            
        
        print(f"Avg error from episode {e}:", error_sum / i)

    
    if (e + 1) % 50 == 0:
        print("\nStudy case:")
        
        output = inputs
        for l in my_network.layers:
            print("Output: ", [o[0] for o in output])
            output = l.forwards(output)
            print()
        

        error = mse(output, real_output)
        print("Error: ", error)

        output_gradient = mse_gradient(output, real_output)
        for l in reversed(my_network.layers):
            if isinstance(l, Dense):
                print("Dense Layer....")

            if isinstance(l, LeakyReLU):
                print("Leaky ReLU...")
            
            print("Gradient: ", [o[0] for o in output_gradient])
            output_gradient = l.backwards(output_gradient, learning_rate)
        
        start_plot = 100
        end_plot = 300

        plt.plot(itteration_aux[start_plot:end_plot], real[start_plot:end_plot], label="Real values")
        plt.plot(itteration_aux[start_plot:end_plot], predicted[start_plot:end_plot], label="Predicted values")
        plt.legend()
        plt.show()
        

    itteration_aux = []
    predicted = []
    real = []