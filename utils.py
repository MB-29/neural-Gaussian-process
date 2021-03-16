import numpy as np

def encode_one_hot(labels):
    """Zero-mean one-hot encoding
    """
    n = labels.shape[0]
    classes = np.unique(labels)
    m = classes.shape[0]
    one_hot_labels = np.full((n, m), -1/m)
    for i in range(n):
        one_hot_labels[i, labels[i]] = 1-1/m
    return one_hot_labels

