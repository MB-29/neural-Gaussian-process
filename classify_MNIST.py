
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

from nngp import NNGP

n_train = 1000
n_test = 200

sigma_eps = 0.01
sigma_w_2 = 1.45
sigma_b_2 = 0.28
L = 3

data = MNIST('../../Data/MNIST/raw')
data_vectors, labels = data.load_training()
data_vectors, labels = np.array(data_vectors), np.array(labels)
print('Loaded MNIST')

training_data = data_vectors[:n_train]
test_data = data_vectors[n_train:n_train + n_test]
training_labels = labels[:n_train]
test_labels = labels[n_train:n_train+n_test]


classifier = NNGP(
    training_data,
    training_labels,
    test_data,
    L,
    sigma_eps_2=sigma_eps**2,
    sigma_w_2=sigma_w_2,
    sigma_b_2=sigma_b_2,
    classify=True
    )
classifier.train()
predicted_labels = classifier.classify()

accuracy = np.mean(predicted_labels == test_labels)
print(f'accuray = {accuracy}')