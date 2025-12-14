import numpy as np

#Mean Squared Error cost function for a simple linear regression model.
x_train = np.array([1,2,5])
y_train = np.array([100,400,500])

w = 100
b = 200

def compute_cost(x, y, w, b):
    m = x.shape[0]

    total_cost = 0

    for i in range(m):
        y_pred = w * x[i] + b
        total_cost += (y_pred - y[i]) ** 2

    return total_cost / (2 * m)

#def compute_cost(x, y, w, b):
    y_pred = w * x + b
    return np.sum((y_pred - y)**2) / (2 * len(x))
#np.summ() np.array'leri için daha hızlı


