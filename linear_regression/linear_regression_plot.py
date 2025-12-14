import numpy as np
import matplotlib.pyplot as plt

# Simple linear model (y = w*x + b) and visualization

x_train = np.array([1,2,5])
y_train = np.array([100,400,500])

w = 100
b = 200

plt.scatter(x_train,y_train,c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


def compute_model_output(x, w, b):

    m = x.shape[0] # or m = len(x)
    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i] = w * x[i] + b

    return y_pred

#def compute_model_output(x, w, b):
    return w * x + b
# x -> np.array olduğu için numpy arrayın tüm elemanları için aynı işelim yapıyor döndüğü elle yazmaya gerek kalmıyor.


tmp_t_pred = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_t_pred, c = 'b', label = 'prediction')

plt.scatter(x_train,y_train,c='r', label = 'data', marker='x')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()