import numpy as np
import matplotlib.pyplot as plt

# AND gate dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Initialize
w = np.zeros(2)
b = 0
lr = 0.1

def step(x):
    return 1 if x >= 0 else 0

# Training
for epoch in range(10):
    for i in range(len(X)):
        net = np.dot(w, X[i]) + b
        pred = step(net)
        error = y[i] - pred

        # update rule
        w += lr * error * X[i]
        b += lr * error

print("Weights:", w)
print("Bias:", b)

# --- Plotting Decision Regions and Points with Colors ---

# Create a meshgrid to plot decision regions
x1_min, x1_max = -0.5, 1.5
x2_min, x2_max = -0.5, 1.5
h = 0.02 # step size in the mesh

x1_vals, x2_vals = np.meshgrid(np.arange(x1_min, x1_max, h),
                               np.arange(x2_min, x2_max, h))

# Predict class for each point in the meshgrid
Z = np.array([step(np.dot(w, [x1, x2]) + b) for x1, x2 in zip(x1_vals.ravel(), x2_vals.ravel())])
Z = Z.reshape(x1_vals.shape)

plt.figure(figsize=(10,6))

# Plot the decision regions
plt.contourf(x1_vals, x2_vals, Z, alpha=0.4, cmap='coolwarm')

# Plot original data points with distinct colors
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], marker='o', color='red', s=100, edgecolors='k', label="Class 0" if i==0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], marker='x', color='blue', s=100, edgecolors='k', label="Class 1" if i==3 else "")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Region (AND Gate)")
plt.legend()
plt.grid(True)
plt.show()