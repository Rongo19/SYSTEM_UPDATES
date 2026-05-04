import numpy as np

# Convert 0 → -1 (Hopfield uses bipolar values)
def to_bipolar(x):
    return np.where(x == 0, -1, 1)

# 4 patterns (you can change them)
patterns = np.array([
    [1, -1, 1, -1],
    [1,  1, -1, -1],
    [-1, 1, -1,  1],
    [-1, -1, 1,  1]
])

# -------- Training (Hebbian learning) --------
n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    p = p.reshape(-1, 1)
    W += p @ p.T   # outer product

# Remove self-connections
np.fill_diagonal(W, 0)

print("Weight matrix:\n", W)

# -------- Recall function --------
def recall(x, W, steps=5):
    x = x.copy()
    for _ in range(steps):
        x = np.sign(W @ x)
        x[x == 0] = 1  # handle zeros
    return x

# -------- Test with noisy input --------
test = np.array([1, -1, 1, 1])   # noisy version of first pattern
print("\nInput (noisy):", test)

output = recall(test, W)
print("Recalled output:", output)