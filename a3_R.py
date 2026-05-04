import numpy as np

# Training data (ASCII values of digits 0–9)
X_ascii = np.array([48,49,50,51,52,53,54,55,56,57])

# Derive the parity feature from ASCII values
X = (X_ascii - 48) % 2

# Target output (Even = 0, Odd = 1)
Y = np.array([0,1,0,1,0,1,0,1,0,1])

# Initialize weight and bias
w = 0
b = 0
learning_rate = 0.01

# Training perceptron
for epoch in range(20):
    for i in range(len(X)):
        net = (X[i] * w) + b

        # Step activation function
        if net >= 0:
            output = 1
        else:
            output = 0

        # Update rule
        error = Y[i] - output
        w = w + learning_rate * error * X[i]
        b = b + learning_rate * error

# Testing
print("Number  Output")
for i in range(len(X_ascii)): # Use X_ascii for printing original numbers
    # Calculate net using the derived parity feature
    input_val = (X_ascii[i] - 48) % 2
    net = (input_val * w) + b
    if net >= 0:
        output = 1
    else:
        output = 0
    print(chr(X_ascii[i]), "     ", output)