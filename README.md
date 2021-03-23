# Neural Networks as Gaussian Processes


A NumPy implementation of the bayesian inference approach of [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165).

We focus on infinitely wide neural network endowed with ReLU nonlinearity function, allowing for an analytic computation of the layer kernels. 

## Usage

## Requirements
* Python 3
* numpy

### Installation

Clone the repository
```bash
git clone https://github.com/MB-29/NN-gaussian-process.git
```
and move to the root directory

```bash
cd NN-gaussian-process
```

### Use our module


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

## Examples
* A classification script for MNIST is provided in the file `classify_MNIST.py`. In relies on the additional requirement `python-mnist` available on pip.
* A 1D regression script is provided in the file `1D_regression.py`. We obtained the following results.

[Network expressivity](demo/expressivity.png)
[Fixed point analysis](demo/fixed_points.png)
[Test uncertainty and test error](demo/error.png)

