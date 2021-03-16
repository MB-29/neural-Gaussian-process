# Neural Networks as Gaussian Processes


A NumPy implementation of the bayesian inference approach of [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165). We implement the Bayesian inference in the framework of an infinite neral net with ReLU nonlinearity, allowing for an analytic computation of the layer kernels. 

## Usage

Clone the repository
```bash
git clone https://github.com/MB-29/NN-gaussian-process.git
```
move to the root directory

```bash
cd NN-gaussian-process
```
run the code

```bash
python predict_MNIST.py
```


## Requirements
* Python 3
* numpy