import numpy as np
import copy
import matplotlib.pyplot as plt

# A simple logistic regression implementation from scratch with a plotted decision boundary.

x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(x, y, w, b):
    m = x.shape[0]

    z = x.dot(w) + b
    y_pred = sigmoid(z)
    cost = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).sum()

    return cost / m

def compute_gradient(x, y, w, b):
    m = x.shape[0]

    z = x.dot(w) + b
    y_pred = sigmoid(z)
    error = y_pred - y

    dj_dw = (1/m) * (x.T.dot(error))
    dj_db = (1/m) * np.sum(error)

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    j_history = []
    p_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        j_history.append(compute_cost(x, y, w, b))

        p_history.append((w.copy(), b))

    return w, b, j_history, p_history

w_init = np.zeros(x_train.shape[1])
b_init = 0
alpha = 0.01
iterations = 10000

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

print(f"Final w: {w_final}")
print(f"Final b: {b_final}")
print(f"Son Maliyet: {compute_cost(x_train, y_train, w_final, b_final):.4f}")


w1, w2 = w_final[0], w_final[1]

x1_min, x1_max = 0, 4
x1_plot = np.linspace(x1_min, x1_max, 100)

x2_boundary = (-w1 / w2) * x1_plot - (b_final / w2)
# sigmoid == 0 -> karar sınırı  z = w1 * x1 + w2 * x2 + b
# z = 0 -> w2 * x2 = -w1 * x1 - b ----> x2 = - ( w1 * x1 + b) / w2

plt.figure(figsize=(8, 6))

plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], marker='o', color='blue', label='Sınıf 0 (y=0)')

plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], marker='x', color='red', label='Sınıf 1 (y=1)')

plt.plot(x1_plot, x2_boundary, color='green', linestyle='-', label='Karar Sınırı')

plt.title("Lojistik Regresyon Karar Sınırı")
plt.xlabel("Özellik 1 ($x_1$)")
plt.ylabel("Özellik 2 ($x_2$)")
plt.xlim(x1_min, x1_max)
plt.ylim(0, 3)
plt.legend()
plt.grid(True)
plt.show()