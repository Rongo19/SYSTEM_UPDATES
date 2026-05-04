import numpy as np
import matplotlib.pyplot as plt

# -------- Dataset (2 classes) --------
X = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [6, 5],
    [7, 7],
    [8, 6],
    [9, 7]
])

# Labels (0 and 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# -------- Initialize --------
w = np.zeros(2)
b = 0
lr = 0.1

# -------- Training (Perceptron Learning Law) --------
for epoch in range(20):
    for i in range(len(X)):
        net = np.dot(X[i], w) + b
        
        # Step activation
        if net >= 0:
            output = 1
        else:
            output = 0
        
        # Error
        error = y[i] - output
        
        # Update rule
        w = w + lr * error * X[i]
        b = b + lr * error

# -------- Plot Decision Regions --------
x_min, x_max = 0, 10
y_min, y_max = 0, 10

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

grid = np.c_[xx.ravel(), yy.ravel()]

# Predict on grid
Z = []
for point in grid:
    net = np.dot(point, w) + b
    Z.append(1 if net >= 0 else 0)

Z = np.array(Z).reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red')
    else:
        plt.scatter(X[i][0], X[i][1], color='blue')

# Plot decision boundary
x_vals = np.linspace(0, 10, 100)
y_vals = -(w[0]*x_vals + b) / w[1]
plt.plot(x_vals, y_vals)

plt.title("Perceptron Decision Regions")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()