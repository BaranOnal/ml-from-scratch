import numpy as np
import matplotlib.pyplot as plt

# A simple logistic regression model implemented from scratch using gradient descent.


x_train = np.array([1, 2, 5, 7, 10])
y_train = np.array([0, 0, 1, 1, 1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0

    for i in range(m):
        z = w * x[i] + b
        a = sigmoid(z)
        total_cost += - (y[i] * np.log(a + 1e-10) + (1 - y[i]) * np.log(1 - a + 1e-10))

    return total_cost / m


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dw = 0
    db = 0

    for i in range(m):
        z = w * x[i] + b
        a = sigmoid(z)

        dw += (a - y[i]) * x[i]
        db += (a - y[i])

    return dw / m, db / m


def gradient_descent(x, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dw, db = compute_gradient(x, y, w, b)

        w -= alpha * dw
        b -= alpha * db

        if i % 1000 == 0:
            print(f"iter {i}, cost={compute_cost(x, y, w, b):.4f}")

    return w, b


def predict(x, w, b):
    return sigmoid(w * x + b)


w_init = 0
b_init = 0
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, 0.01, 10000)


print("\nPredictions:")
print(predict(x_train, w_final, b_final))


plt.figure(figsize=(10, 4))

plt.scatter(x_train, y_train, c='red', label='Training Data')

x_plot = np.linspace(0, 12, 200)
y_plot = sigmoid(w_final * x_plot + b_final)

plt.plot(x_plot, y_plot, label='Sigmoid Prediction Curve', linewidth=2)

plt.title("Logistic Regression Prediction (Sigmoid Curve)")
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.ylim(-0.1, 1.1)

x_boundary = -b_final / w_final

plt.axvline(x=x_boundary, color='green', linestyle='--', label=f'Decision Boundary (X={x_boundary:.2f})')
plt.legend()
plt.show()