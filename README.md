# Neural Networks as Gaussian Processes


A NumPy implementation of the bayesian inference approach of [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165). We implement the Bayesian inference in the framework of an infinite neral net with ReLU nonlinearity, allowing for an analytic computation of the layer kernels. 

## Usage

### Installation

Clone the repository
```bash
git clone https://github.com/MB-29/NN-gaussian-process.git
```
move to the root directory

```bash
cd NN-gaussian-process
```

### Usage


```python
from nngp import NNGP

# ... 

regression = NNGP(
    training_data,              # Data
    training_targets,
    test_data,
    L,                          # Neural network depth
    sigma_eps_2=sigma_eps**2,   # Observation noise variance
    sigma_w_2=sigma_w_2,        # Weight hyperparameter
    sigma_b_2=sigma_b_2         # Bias hyperparameter
    )

regression.train()
predictions, covariance = regression.predict()

```


## Requirements
* Python 3
* numpy