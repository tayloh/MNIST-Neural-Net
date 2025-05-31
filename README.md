# taylohs MNIST Neural network

This is a naive implementation of a basic feed forward neural net, for learning purposes.  
Runs on CPU (single-thread), with stochastic gradient descent. Matrix/vector operations are not optimized. Allocs could be worse.

## Run
`gcc -o bin/mnist_nn.exe src/main.c src/mnist.c src/neuralnet.c linalg/src/vector.c linalg/src/matrix.c`

Speeds per training sample up by one order of magnitude:  
`gcc -O3 -march=native -ffast-math -o bin/mnist_nn.exe src/main.c src/mnist.c src/neuralnet.c linalg/src/vector.c linalg/src/matrix.c`

`bin/mnist_nn.exe`

## Spec
~ 70 15 15 train validate test split with 98.4 % accuracy on test (highest measured)  

Architecture: 784, 256, 10  

Parameters:
* learning_rate = 0.01f
* lambda/l2/weight decay = 0.0001f
* patience = 3
* max_epochs = 30
* activation function: ReLU
* loss function: Cross-Entropy
* initialization: He

## Dataset
https://www.kaggle.com/datasets/hojjatk/mnist-dataset