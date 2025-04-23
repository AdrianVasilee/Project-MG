from .activation import Activation
import numpy as np

class ReLU(Activation):
    def __init__(self):
        ReLU = lambda x: np.maximum(0, x)
        ReLU_gradient = lambda x: (x > 0).astype(int)
        super().__init__(ReLU, ReLU_gradient)