import numpy as np
import matplotlib.pyplot as plt
import copy

# Implements multivariate linear regression using gradient descent and visualizes cost and parameter updates.

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost = 0

    for i in range(m):
        y_pred = np.dot(x[i],w) + b
        cost += (y_pred - y[i])**2
    return cost/(2*m)


def compute_gradient(x, y, w, b):
    m,n = x.shape
    sum_dw = np.zeros(n)
    sum_db = 0

    for i in range(m):
        err = (np.dot(x[i],w) + b) - y[i]
        for j in range(n):
            sum_dw[j] += err * x[i][j]

        sum_db += err

    sum_dw = sum_dw / m
    sum_db = sum_db / m
    return sum_dw, sum_db


def gradient_descent(x, y, w_in, b_in, learning_rate, num_iterations, cost_function, gradient_function):
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iterations):
        sum_dw, sum_db = gradient_function(x, y, w, b)
        w -= learning_rate * sum_dw
        b -= learning_rate * sum_db

        j_history.append(cost_function(x, y, w, b))

    return w , b , j_history



b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

w_final, b_final, j_history = gradient_descent(
    X_train, y_train, w_init, b_init,
    learning_rate = 1e-7,
    num_iterations = 1000,
    cost_function=compute_cost,
    gradient_function=compute_gradient

)

m = X_train.shape[0]
for i in range(m):
    print(np.dot(X_train[i],w_final)+b_final)


#Cost func. J
plt.figure(figsize=(10, 4))
plt.plot(j_history, label='Cost J')
plt.title("Maliyet Fonksiyonu (J) İterasyonlar Boyunca")
plt.xlabel("İterasyon Sayısı")
plt.ylabel("Maliyet J")
plt.grid(True)
plt.legend()
plt.show()

#actual values and predictions
y_pred = np.dot(X_train, w_final) + b_final

plt.figure(figsize=(10, 4))
plt.scatter(y_train, y_pred, marker='o', c='purple', label='Tahminler')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', label='İdeal Tahmin (y=x)') # İdeal tahmin çizgisi

plt.title("Tahmin Edilen Y Değerleri vs. Gerçek Y Değerleri")
plt.xlabel("Gerçek Y Değerleri")
plt.ylabel("Tahmin Edilen Y Değerleri")
plt.xlim(min(y_train) - 10, max(y_train) + 10)
plt.ylim(min(y_train) - 10, max(y_train) + 10)
plt.legend()
plt.grid(True)
plt.show()

