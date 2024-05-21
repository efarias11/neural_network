inputs = [1.2,5.1,2.1] # inputs for neurons
weights = [3.1, 2.1, 8.7] # weight is unique for each input
bias = 3 # every unique neuron has a unique bias

output = inputs[0]*weights[0] +inputs[1]*weights[1]+inputs[2]*weights[2]+bias # simple algorithm for neural network
print(output)