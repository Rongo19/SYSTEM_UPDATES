import numpy as np

# Input (XOR problem)
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Output
Y = np.array([[0],[1],[1],[0]])

# Initialize weights randomly
np.random.seed(1)
W1 = np.random.rand(2, 2)   # input → hidden
W2 = np.random.rand(2, 1)   # hidden → output

# Biases
b1 = np.zeros((1,2))
b2 = np.zeros((1,1))

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training
lr = 0.5

for epoch in range(10000):

    # -------- Forward Propagation --------
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # -------- Error --------
    error = Y - final_output

    # -------- Backpropagation --------
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # -------- Update Weights --------
    W2 += hidden_output.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr


# Final Output
print("Final Output:")
print(final_output)