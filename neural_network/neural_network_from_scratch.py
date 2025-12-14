import numpy as np

"""
A simple feedforward neural network implementation from scratch using NumPy.
Includes dense layers, activation functions, and a minimal sequential model.
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def my_dense(a_in, W, b, g):
    z = a_in @ W + b
    a_out = g(z)
    return a_out


def my_squential(x , W1, W2, W3, b1, b2, b3):
    a1 = my_dense(x, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return a3


if __name__ == "__main__":
    # simple test
    x = np.array([[0.5, 0.2, 0.1]])
    W1 = np.random.randn(3, 4)
    W2 = np.random.randn(4, 4)
    W3 = np.random.randn(4, 1)
    b1 = np.zeros((1, 4))
    b2 = np.zeros((1, 4))
    b3 = np.zeros((1, 1))

    output = my_squential(x, W1, W2, W3, b1, b2, b3)
    print("Output:", output)
