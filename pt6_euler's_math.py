import numpy as np
import nnfs

nnfs.init() # initailizing nnfs

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [ 1.41, 1.051, 0.026]] 
#E = math.e # euler's number e (2.71828182846) need math object imported

# we exponentiate the values so negative values don't lose meaning in the ReLU activation function
exp_values = np.exp(layer_outputs) # using euler's number to exponetiate the values and add it to a new array vector 
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) # normalization values helps us get the probability distribution of exponentiated values

print(norm_values)
# print(sum(norm_values)) # all values should sum close to 1
# print(np.sum(layer_outputs, axis=1, keepdims=True)) # axis = 1 gives us the sum of rows of the matrix, keepdims=True keeps the shape the same