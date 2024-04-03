import numpy as np

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

for value in random_values:
    print(f"ReLU of {value}: {relu(value)}")
    print(f"Leaky ReLU of {value}: {leaky_relu(value)}")
    print(f"Tanh of {value}: {np.tanh(value)}")

