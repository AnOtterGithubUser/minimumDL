import numpy as np

def softmax(y):
    return np.exp(y) / np.sum(np.exp(y))
