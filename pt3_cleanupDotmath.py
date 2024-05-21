import numpy as np
# numpy .dot function math breakdown
inputs = [1, 2, 3, 2.5] 
weights = [0.2, 0.8, -0.5, 1]
biases = 2 

output = np.dot(weights, inputs)+biases # doing multiplication at each corresponding array index then adding all together, then adding the bias at then end
# (0.2*1 + 0.8*2.0 + -0.5*3 + 1.0*2.5) + 2 = 4.8
print(output)

