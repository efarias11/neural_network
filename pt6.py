import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons)) 

    def forward(self, inputs): 
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: # class is designed to prevent value overflow errors called softmaxing
    def foward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities

X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3) # two input features (X,y) output is 3 but can be anything
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.foward(dense2.output)

print(activation2.output[:5])
