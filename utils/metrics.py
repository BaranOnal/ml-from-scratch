import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_pred==1) & (y_true==1))
    fp = np.sum((y_pred==1) & (y_true==0))
    return tp / (tp + fp + 1e-8)

def recall(y_true, y_pred):
    tp = np.sum((y_pred==1) & (y_true==1))
    fn = np.sum((y_pred==0) & (y_true==1))
    return tp / (tp + fn + 1e-8)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p*r) / (p + r + 1e-8)
