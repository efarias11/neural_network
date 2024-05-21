import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1] # class target gets the index value from each row 

print(-np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets])) # range(len(softmax_outputs)) goes through each row to its max length
# if there is a 0 in the outputs you will get a infinity warning