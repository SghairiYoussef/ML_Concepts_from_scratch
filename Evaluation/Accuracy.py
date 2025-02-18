import numpy as np

def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return np.mean(np.array(y_true) == np.array(y_pred))