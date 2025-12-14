import numpy as np
import matplotlib.pyplot as plt

# Implements linear regression from scratch using gradient descent, including cost tracking and visualization.

x_train = np.array([1,2,5])
y_train = np.array([100,400,500])


def compute_cost(x, y, w, b):
    y_pred = w * x + b
    return np.sum((y_pred - y) ** 2) / (2 * len(x))


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    sum_dw = 0
    sum_db = 0
    for i in range(m):
        y_pred = w * x[i] + b
        sum_dw_i = (y_pred - y[i]) * x[i]
        sum_db_i = (y_pred - y[i])
        sum_dw += sum_dw_i
        sum_db += sum_db_i


    sum_dw = sum_dw / m
    sum_db = sum_db / m
    return sum_dw, sum_db

def gradient_descent(x, y, w, b, learning_rate, num_iterations, cost_function, gradient_function):
    j_history = []
    p_history = []

    for i in range(num_iterations):
        sum_dw, sum_db = gradient_function(x,y,w,b)

        w = w - learning_rate * sum_dw
        b = b - learning_rate * sum_db

        j_history.append(cost_function(x,y,w,b))
        p_history.append([w,b])

    return w, b, j_history, p_history


w_init = 0
b_init = 0


num_iterations = 1000
learning_rate = 0.01

w_final, b_final , j_history, p_history = gradient_descent(
    x_train, y_train, w_init, b_init, learning_rate,
    num_iterations, compute_cost, compute_gradient
)


plt.figure(figsize=(10, 4))
plt.plot(j_history, label='Cost J')
plt.title("Cost Function J over Iterations")
plt.xlabel("Iteration Number")
plt.ylabel("Cost J")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 4))
plt.scatter(x_train, y_train, marker='x', c='red', label='Training Data')

x_plot = np.linspace(0, 6, 100) # x aralığı
y_plot = w_final * x_plot + b_final
plt.plot(x_plot, y_plot, label=f'Prediction Line: y = {w_final:.2f}x + {b_final:.2f}', c='blue')

plt.title("Linear Regression Final Prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()


# parameter history
ws = [p[0] for p in p_history]
bs = [p[1] for p in p_history]

plt.figure(figsize=(10, 4))

plt.plot(ws, label="w")
plt.plot(bs, label="b")
plt.title("Parameter Updates During Gradient Descent")
plt.xlabel("Iteration Number")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()