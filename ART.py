import numpy as np

# Binary input patterns
inputs = np.array([
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 0]
])

# Parameters
num_features = 4
num_clusters = 2
vigilance = 0.5

# Initialize weights
weights = np.ones((num_clusters, num_features))

# ART1 clustering
for idx, x in enumerate(inputs):
    print("\nInput:", x)

    for j in range(num_clusters):
        # Calculate match score
        match = np.sum(np.minimum(x, weights[j])) / np.sum(x)

        if match >= vigilance:
            print("Assigned to Cluster", j)

            # Update weights
            weights[j] = np.minimum(x, weights[j])
            break
    else:
        print("No matching cluster found")

# Final weights
print("\nFinal Weights:")
print(weights)