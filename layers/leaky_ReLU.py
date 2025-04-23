from .activation import Activation
import numpy as np

class LeakyReLU(Activation):
    def __init__(self, alpha):
        max = lambda x: np.where(x >= 0, x, alpha*x)
        l_ReLU = lambda x: max(x) / np.linalg.norm(max(x), axis=0, keepdims=True)
        l_ReLU_prime = lambda x: np.where(x >= 0, 1, alpha)

        super().__init__(l_ReLU, l_ReLU_prime)