import numpy as np

def encode_one_hot(labels):
    n = labels.shape[0]
    classes = np.unique(labels)
    m = classes.shape[0]
    one_hot_labels = np.zeros((n, m))
    for i in range(n):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

