'''
solving for x

e** x = b
'''
# import numpy as np
# import math
# b = 5.2

# print(np.log(b)) 
# print(math.e**1.6486586255873816) # prove natural logs (ln = log base of e^x) are a base euler's number to the power of x 
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0] # One-hot encoding

loss= -(math.log(softmax_output[0])*target_output[0] +
        math.log(softmax_output[1])*target_output[1] + # since you are multiplying the 0 from the vector this value becomes 0
        math.log(softmax_output[2])*target_output[2])  # since you are multiplying the 0 from the vector this value becomes 0

# print(loss)
# loss = -math.log(softmax_output[0]) # we can simplify the Categorical Cross-entropy formula to just -log()
# print(loss)

print(-math.log(0.7)) # notice when the confidence is higher the loss is lower
print(-math.log(0.5)) # notice when the confidence is lower the loss is higher (loss = measurement of error, is higher)