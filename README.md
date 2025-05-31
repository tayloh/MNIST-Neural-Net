# taylohs MNIST Neural network

## Run
gcc -o bin/mnist_nn.exe src/main.c src/mnist.c src/neuralnet.c linalg/src/vector.c linalg/src/matrix.c

Speeds per training sample up by one order of magnitude:  
gcc -O3 -march=native -ffast-math -o bin/mnist_nn.exe src/main.c src/mnist.c src/neuralnet.c linalg/src/vector.c linalg/src/matrix.c

bin/mnist_nn.exe

## Dataset
https://www.kaggle.com/datasets/hojjatk/mnist-dataset