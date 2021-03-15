import numpy as np
import matplotlib.pyplot as plt 

from nngp import NNGP

L = 10
n_training = 5
n_test = 30
sigma_eps = 0.01

def signal(x):
    # return 2 * x + 3
    return np.cos(0.1*x)

# training_data = np.sort(5 * np.random.randn(n_training, 1))
training_data = np.array([-9, -4, 2, 3, 8]).reshape((n_training, 1))
training_output = signal(training_data)
training_targets = training_output + sigma_eps * np.random.randn(n_training, 1)

test_data = np.linspace(-10, 10, n_test).reshape(n_test, 1)
test_output = signal(test_data)

regression = NNGP(training_data, training_targets, test_data, L, sigma_eps**2)

print('training')
regression.train()

print('predicting')
predictions, covariance = regression.predict()
variances = np.diag(covariance)

plt.scatter(training_data, training_targets, label='train')
plt.plot(test_data, signal(test_data), label='signal', ls='--', color='black', alpha=.7, lw=1)
plt.errorbar(test_data, predictions, yerr=variances, color='red', label='prediction')
plt.legend()
plt.show()