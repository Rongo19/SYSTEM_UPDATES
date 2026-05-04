import numpy as np
import pandas as pd

class Neuron:
  def __init__(self, weights, bias, activation):
    self.weights = np.array(weights)
    self.bias = np.array(bias) # Convert bias to a NumPy array as well
    self.activation = activation

  def forward(self, inputs):
      inputs = np.array(inputs)
      total = np.dot(self.weights, inputs) + self.bias
      return self.activation(total)
  
# function for activatin functinos : (sigmoid , relu, tanh):

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def relu(x):
  return np.maximum(0,x)

def tanh(x):
  return np.tanh(x)

# now creating a neuron using same class, same weights and bias for fair comparisons

weights = [1] # Changed to a single weight to match single input 'i'
bias = [0]    # Changed to a single bias to match single input 'i'

sigmoid_neuron = Neuron(weights, bias, sigmoid)
relu_neuron = Neuron(weights, bias, relu)
tanh_neuron = Neuron(weights, bias, tanh)

# plotting the activatin functions :

import matplotlib.pyplot as plt

# input range :

x = np.linspace(-10, 10, 400)

# forward pass through neurons :

y_sigmoid = [sigmoid_neuron.forward([i]) for i in x]
y_relu = [relu_neuron.forward([i]) for i in x]
y_tanh = [tanh_neuron.forward([i]) for i in x]

#plotting

plt.figure(figsize = (10,6))

plt.subplot(3,1,1)
plt.plot(x, y_sigmoid, label = "sigmoid", color = "blue")
plt.title('Sigmoid Activation Function')
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(x, y_relu, label = "relu", color = "green")
plt.title('ReLU Activation Function')
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
plt.plot(x, y_tanh, label = "tanh", color = "red")
plt.title('Tanh Activation Function')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()