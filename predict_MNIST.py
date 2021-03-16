
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

from nngp import NNGP

n_train = 100
n_test = 10
Ny, Nx = 28, 28

sigma_eps = 0.01
L = 20

data = MNIST('../../Data/MNIST/raw')
data_vectors, labels = data.load_training()
data_vectors, labels = np.array(data_vectors), np.array(labels)
data_images = data_vectors.reshape((-1, Ny, Nx))
print('Loaded MNIST')

training_data = data_vectors[:n_train]
test_data = data_vectors[n_train:n_train + n_test]
training_labels = labels[:n_train]
test_labels = labels[n_train:n_train+n_test]


classifier = NNGP(training_data, training_labels, test_data, L, sigma_eps_2=sigma_eps**2, classify=True)
classifier.train()
predicted_labels = classifier.classify()