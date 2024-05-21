import numpy as np

np.random.seed(0)
# for all data (input, weights, and biases) we try to keep the values between -1 and 1
# for X/inputs we can "scale" the values down to have the same meaning while being between -1 and 1
# we do this to prevent an explosion of data points to make the data more consistent (for the fitting line?)
X = [[1, 2, 3, 2.5], # X is the sample/input data (standard for ML denotion)
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] 


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): # creating the weights and biases matrices 
        # since we have full control of the initialization we don't need to transpose the matrix
        self.weights = 0.10* np.random.rand(n_inputs, n_neurons) # self.weights = np.random.rand(n_inputs, n_neurons)
                                                           # (size of each sample or single batch (4 samples in each row of matrix X), number of neurons)
        # randn = is a gaussian distribution bounded around 0, we need to multiply the weights by 0.1 to make the weights between -1 and 1
        self.biases = np.zeros((1, n_neurons)) # defining 1 row and number of neurons for the columns and .zeros fills the vector with 0's
    
    def forward(self, inputs): # multiplying the input data set matrix with the created weight matrix and adding the biases vector
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5) # 4 = number of samples in X data set and 5 = number of neurons user desires (can be any value)
layer2 = Layer_Dense(5,2) # 5 = output of layer1 (has to be 5) and 2 =  any size you want

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)