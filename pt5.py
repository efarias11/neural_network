# Import the numpy library for numerical operations
import numpy as np

# Import nnfs and the spiral_data function from nnfs.datasets
import nnfs 
from nnfs.datasets import spiral_data

# Initialize the nnfs library for consistent random values and other setup
nnfs.init()

# Define input data X as a list of feature sets (not used later in this example)
X = [[1, 2, 3, 2.5], 
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] 

# Generate a dataset of 100 points with 3 classes using the spiral_data function
X, y = spiral_data(100, 3) # 100 feature sets of 3 classes

# Define a class for a dense (fully connected) layer
class Layer_Dense:
    # Initialize the layer with the number of inputs and neurons
    def __init__(self, n_inputs, n_neurons): 
        # Initialize weights with small random values (0.10 times random values)
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons) 
        # Initialize biases with zeros
        self.biases = np.zeros((1, n_neurons)) 

    # Define the forward pass method to calculate the layer's output
    def forward(self, inputs): 
        # Calculate the output of the layer as dot product of inputs and weights plus biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Define a class for the ReLU activation function
class Activation_ReLU:
    # Define the forward pass method to apply the ReLU activation
    def forward(self, inputs):
        # Apply ReLU activation (set all negative values to 0)
        self.output = np.maximum(0, inputs)

# Create an instance of Layer_Dense with 2 input features and 5 neurons
layer1 = Layer_Dense(2, 5) 

# Create an instance of Activation_ReLU
activation1 = Activation_ReLU()

# Perform a forward pass of the data through the dense layer
layer1.forward(X)

# Perform a forward pass of the dense layer's output through the ReLU activation
activation1.forward(layer1.output)

# Print the output of the activation function (should contain no negative values)
print(activation1.output) # prints no negative values, later optimizer will fix the dead values (0)
