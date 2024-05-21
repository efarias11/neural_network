import numpy as np
inputs = [1, 2, 3, 2.5] # vector
weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]] # grouping all weights, creating a 2D array, matrix

biases = [2, 3, 0.5] # grouping each neuron biases, makes output values stay positive (vector)

# for shape, it has to be a square/rectagular 1D/2D/3D array, no ragged arrays
# tensor is an object that can be represented as an array, not just an array

output = np.dot(weights, inputs) + biases # you pass weight first since it is a matrix or a bigger shape than inputs 
                                          # (can't multiply a matrix to a vector, but can multiply vector to a matrix, linear algebra)
# multiplying the input vector to the weights matrix to create a vector. Then you add the biases to that vector for the final output vector 
print(output)
