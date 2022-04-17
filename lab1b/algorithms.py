import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, x_input, targets, bias,learningRate=0.01, seed=42, convergeThreshold=0.1, directory=""):
        self.x_input = x_input
        self.targets = targets
        self.bias = bias
        self.weighedSum = 0
        self.learningRate = learningRate 
        self.seed = seed
        np.random.seed(seed)
        self.weights = np.random.rand(1, x_input.shape[0]) 
        self.convergeThreshold = convergeThreshold
        self.directory = directory


    def twoLayerPerceptron(self):

        return None