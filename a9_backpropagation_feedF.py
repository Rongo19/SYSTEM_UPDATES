import numpy as np

# Input dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Output (XOR)
Y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(1)
W1 = np.random.rand(2, 2)   # Input → Hidden
W2 = np.random.rand(2, 1)   # Hidden → Output

# Bias
b1 = np.zeros((1,2))
b2 = np.zeros((1,1))

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Learning rate
lr = 0.5

# Training
for epoch in range(10000):

    # -------- Feed Forward --------
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

    # -------- Error --------
    error = Y - output_layer

    # -------- Back Propagation --------
    d_output = error * sigmoid_derivative(output_layer)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_layer)

    # -------- Update Weights --------
    W2 += hidden_layer.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr


# Output
print("Final Output:")
print(output_layer)

# final output (binary)
print("\nFinal Output (Binary):")
print((output_layer > 0.5).astype(int))