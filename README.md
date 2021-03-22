# Neural Networks as Gaussian Processes


A NumPy implementation of the bayesian inference approach of [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165).

We focus on infinitely wide neural network endowed with ReLU nonlinearity function, allowing for an analytic computation of the layer kernels. 

## Usage

### Requirements
* Python 3
* numpy

### Installation

Clone the repository
```bash
git clone https://github.com/MB-29/neural-Gaussian-process.git
```
and move to the root directory

```bash
cd neural-Gaussian-process
```

### Use our module


```python
from nngp import NNGP

# ... 

regression = NNGP(
    training_data,              # data
    training_targets,
    test_data,
    L,                          # neural network depth
    sigma_eps_2=sigma_eps_2,    # observation noise variance
    sigma_w_2=sigma_w_2,        # weight hyperparameter
    sigma_b_2=sigma_b_2         # bias hyperparameter
    )

regression.train()
predictions, covariance = regression.predict()

```

## Examples
* A 1D regression script is provided in the file `1D_regression.py`.
* A classification script for MNIST is provided in the file `classify_MNIST.py`. In relies on the additional requirement `python-mnist` available on pip.



