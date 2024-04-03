import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')

# ReLU
def relu(x):
    return np.maximum(0, x)

plt.subplot(2, 2, 2)
plt.plot(x, relu(x))
plt.title('ReLU')

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')

# Tanh
plt.subplot(2, 2, 4)
plt.plot(x, np.tanh(x))
plt.title('Tanh')

plt.tight_layout()
plt.show()

plt.savefig('activation_functions.png')