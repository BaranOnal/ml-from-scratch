import numpy as np

def split_data(X,y,test_size=0.2,seed=None):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    test_count = int(len(X)*test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std
