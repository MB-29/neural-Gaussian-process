import numpy as np
from scipy.linalg import solve

from utils import encode_one_hot


def ReLU(x):
    return (abs(x) + x) / 2

class NNGP:

    def __init__(self, training_data, training_targets, test_data, L, sigma_eps_2, sigma_b_2=1, sigma_w_2=1, phi=ReLU, classify=False):

        # data
        self.n_training, self.d = training_data.shape
        self.training_data = training_data
        self.training_targets = training_targets if classify == False else encode_one_hot(training_targets)
 
        self.n_test = test_data.shape[0]
        self.test_data = test_data

        self.n_data = self.n_test + self.n_training
        self.data = np.zeros((self.n_data, self.d))
        self.data[:self.n_training] = self.training_data
        self.data[self.n_training:] = self.test_data

        # network
        self.L = L  #network depth
        self.phi = phi
        self.sigma_b_2 = sigma_b_2 
        self.sigma_w_2 = sigma_w_2 

        # inference
        self.sigma_eps_2 = sigma_eps_2

        # F_phi grid
        self.n_var, self.n_cov = 10, 10

        self.min_var, self.max_var = 1, 10
        self.min_corr, self.max_corr = 0.1, 0.9
        self.integral_bound = 10

        # kernel values
        self.K_values = np.zeros((self.n_data, self.n_data, self.L))
        self.theta_values = np.zeros((self.n_data, self.n_data, self.L))

        self.trained = False

    
    def train(self):
        """ Fill K_values with recurrence equation (5)
        """

        print('training')
        
        # initialization
        for i in range(self.n_data):
            for j in range(i+1):
                overlap = self.data[i] @ self.data[j] / self.d
                self.K_values[i, j, 0] = self.sigma_b_2 + self.sigma_w_2 * overlap
                self.K_values[j, i, 0] = self.K_values[i, j, 0]

        # recurrence

        # diagonal
        for l in range(1, self.L):
            for i in range(self.n_data):
                var = self.K_values[i, i, l-1]
                self.K_values[i, i, l] = self.ReLU_iteration_diagonal(var)

        # off-diagonal
            for i in range(self.n_data):
                for j in range(i):
                    K_xx, K_yy = self.K_values[i, i, l-1], self.K_values[j, j, l-1]
                    K_xy = self.K_values[i, j, l-1]
                    theta = self.theta_values[i, j, l-1]
                    K_xy_, theta_ = self.ReLU_iteration(K_xx, K_yy, K_xy, theta)

                    self.K_values[i, j, l] = K_xy_
                    self.K_values[j, i, l] = K_xy_

                    self.theta_values[i, j, l] = theta_
                    self.theta_values[j, i, l] = theta_

        self.trained = True
    

    def predict(self):
        """ Test data prediction

        :return: Predicted mean, predicted covariance matrix
        :rtype: (n_test, d) array, (n_test, n_test) array
        """

        assert self.trained == True

        K_DD = self.K_values[:self.n_training, :self.n_training, -1]
        K_TD = self.K_values[self.n_training:, :self.n_training, -1]
        K_TT = self.K_values[self.n_training:, self.n_training:, -1]

        noisy_kernel = K_DD + self.sigma_eps_2 * np.eye(self.n_training)
        target_inverse = solve(noisy_kernel, self.training_targets)
        kernel_inverse = solve(noisy_kernel, K_TD.T)

        predicted_mean = K_TD @ target_inverse
        predicted_cov = K_TT - K_TD @ kernel_inverse
        
        return predicted_mean, predicted_cov

    def classify(self):
        """Classify by performing regression on one-hot labels
        """

        predicted_mean, predicted_cov = self.predict()
        labels = np.argmax(predicted_mean, axis=1)

        return labels

            
    def ReLU_iteration(self, K_xx, K_yy, K_xy, theta):
        corr = K_xy / np.sqrt(K_xx * K_yy)
        angle_term = np.sin(theta) + (np.pi - theta) * np.cos(theta) 
        kernel_value = self.sigma_b_2 + self.sigma_w_2 / (2*np.pi) * np.sqrt(K_xx * K_yy) * angle_term
        return kernel_value, np.arccos(corr)

    def ReLU_iteration_diagonal(self, var):
        return self.sigma_b_2 + self.sigma_w_2 / (2*np.pi) * var * np.pi


