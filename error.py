import numpy as np

def mse(predicted, real):
    return np.mean(np.power(predicted - real, 2))

def mse_gradient(predicted, real):
    return 2*(predicted - real) / np.size(real)