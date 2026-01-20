import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def accuracy2(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return (tp + tn) / (tp + fn + fp + tn)

def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp + 1e-8)

def recall(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn + 1e-8)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p*r) / (p + r + 1e-8)


if __name__ == '__main__':
    y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0])

    print("Accuracy (Numpy): ", accuracy(y_true, y_pred))
    print("Accuracy (Manual): ", accuracy2(y_true, y_pred))
    print("Precision: ", precision(y_true, y_pred))
    print("Recall: ", recall(y_true, y_pred))
    print("F1 Score: ", f1(y_true, y_pred))
