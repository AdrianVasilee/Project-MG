
import matplotlib.pyplot as plt 
import numpy as np

def plot_network(division, network):
    step = 1 / division

    x_values = []
    y_values = []
    z_values = []

    for x in range(division + 1):
        for y in range(division + 1):
            output = np.array([step * x, step * y])
            output = np.reshape(output, (2, 1))

            for n in network:
                output = n.forwards(output)

            x_values.append(step*x)
            y_values.append(step*y)
            z_values.append(output[0][0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x_values, y_values, z_values)
    ax.set_xlabel("Input A")
    ax.set_ylabel("Input B")
    ax.set_zlabel("Output")

    plt.show()