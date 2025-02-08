import numpy as np

def softmax(z):
    ez = np.exp(z)
    a = ez/np.sum(ez)
    return a