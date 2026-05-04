import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

plt.figure(figsize = (10,6))

plt.subplot(3,1,1)
plt.plot(x, 1/(1+np.exp(-x)), label = "sigmoid", color = "blue")
plt.title('Sigmoid Activation Function')
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(x, np.maximum(0, x), label = "relu", color = "green")
plt.title('ReLU Activation Function')
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
plt.plot(x, np.tanh(x), label = "tanh", color = "red")
plt.title('Tanh Activation Function')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()