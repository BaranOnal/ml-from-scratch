# Implements cost and gradient functions for linear and logistic regression with L2 regularization.

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost_linear_regression(x, y, w, b, lambda_=1.0):
    """Compute regularized cost for linear regression."""
    m, n = x.shape
    f_wb = np.dot(x, w) + b

    cost = np.sum((f_wb - y) ** 2) / (2 * m)

    # L2
    reg_cost = (lambda_ / (2 * n)) * np.sum(w ** 2)

    return cost + reg_cost


def compute_cost_logistic_regression(x, y, w, b, lambda_=1.0):
    """Compute regularized cost for logistic regression."""
    m, n = x.shape

    z = np.dot(x, w) + b
    f_wb = sigmoid(z)
    cost = -np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb)) / m

    #regularization
    reg_cost = (lambda_ / (2 * n)) * np.sum(w ** 2)

    return cost + reg_cost


def compute_gradient_linear_regression(x, y, w, b, lambda_=1.0):
    """Compute gradients for linear regression with L2 regularization."""
    m, n = x.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        error = np.dot(x[i], w) + b - y[i]
        dj_db += error
        for j in range(n):
            dj_dw[j] += error * x[i][j]

    dj_dw = (dj_dw / m) + (lambda_ / m) * w
    dj_db = dj_db / m

    return dj_db, dj_dw


def compute_gradient_logistic_regression(x, y, w, b, lambda_=1.0):
    """Compute gradients for logistic regression with L2 regularization."""
    m, n = x.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        error = sigmoid(np.dot(x[i], w) + b) - y[i]
        dj_db += error
        for j in range(n):
            dj_dw[j] += error * x[i][j]

    dj_dw = (dj_dw / m) + (lambda_ / m) * w
    dj_db = dj_db / m

    return dj_db, dj_dw