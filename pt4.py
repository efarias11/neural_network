import numpy as np
# pretend these input values are temp, humidity, wind speed, etc. each a sample that makes a batch
# We turning this code into an object that can create multiple (increasing) batches at once to better control the fitting line
# bettering the learning rate for the neural network, don't give it too much batches or all data at once (over fitting)
input = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]] 

weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]] 

biases = [2, 3, 0.5] # adding to each to the new multiplied matrix (at the same row index) ()

layer1_output = np.dot(inputs, np.array(weights).T) + biases # had to transpose the weights matrix, so inputs row 0 can multiply with weights row 0 values
#print(layer1_output)

# adding another layer
weights2 = [[0.1, -0.14, 0.5], 
           [-0.5, 0.12, -0.33], 
           [-0.44, 0.73, -0.13]] 

biases2 = [-1, 2, -0.5] 

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2 # multiplying layer 1's finished matrix with the weights of layer 2's transposed matrix 
                                                                      # and adding layer's 2 biases 
print(layer2_output)