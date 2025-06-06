#ifndef NEURALNET_H
#define NEURALNET_H

#include "../linalg/include/linalg.h"
#include "mnist.h"

typedef struct
{
    int layer_input_size;
    int layer_output_size;

    int *layer_sizes;
    int num_layers;

    float (*activation)(float);
    float (*activation_derivative)(float);

    // Each layer has a weight matrix, e.g. hidden layer 0 node 0 has weights: weights[0][0] = [1, 2, ..., 1] to input layer nodes
    Matrix **weights;

    // Bias for each layer (excluding input layer)
    Vector **biases;

    // Activations are the values for each neuron after doing z = w1 * x1 + w2 * x2 + ... + wn * xn - bi
    // activation_i = activation(z)
    Vector **activations;

    // zs are the pre-activation values, that is, z above
    // They are cached to be used in backpropagation
    Vector **zs;

    // Store gradients for each weight and bias
    Matrix **grad_w;
    Vector **grad_b;

    // Track if weights biases init has been called
    int init_called;

} NeuralNet;

// Layer 0 ------------ No biases

// W0      wwwwwwwwwwww Weights weights[0]

// Layer 1 ------------ Biases biases[0] <- OBS

// W1      wwwwwwwwwwww Weights weights[1]

// .
// .
// .

// WN      wwwwwwwwwwww Weights weights[num_layers - 2]

// Layer num_layers - 1 (output layer) Biases biases[num_layers - 2]

// To setup, set_activation, set_activation_derivative, init_w_b
NeuralNet *neuralnet_create(int num_layers, int *layer_sizes);
void neuralnet_free(NeuralNet *nn);

NeuralNet *neuralnet_load_model(const char *model_fp);          // todo
void neuralnet_save_model(NeuralNet *nn, const char *model_fp); // todo
void neuralnet_set_activation(NeuralNet *nn, float (*activation)(float));
void neuralnet_set_activation_derivative(NeuralNet *nn, float (*activation_derivative)(float));

float ReLU(float x);
float ReLU_derivative(float x);

void neuralnet_init_w_b_he(NeuralNet *nn);

// inputs and targets are not altered inside of these functions
void neuralnet_train(NeuralNet *nn, Vector **inputs, Vector **targets, int num_samples, int num_validation_samples, float learning_rate, int max_epochs, float lambda, int patience);
void neuralnet_test(NeuralNet *nn, Vector **inputs, Vector **targets, int num_samples);

int neuralnet_infer(NeuralNet *nn, const Vector *input);

void neuralnet_forward(NeuralNet *nn, const Vector *input);

void neuralnet_backprop(NeuralNet *nn, const Vector *target);
void neuralnet_update_w_b(NeuralNet *nn, float learning_rate, float lambda);

// Use cross-entropy since MNIST is a classification task
float neuralnet_compute_softmax_CE(const Vector *output, const Vector *target);

void neuralnet_print(const NeuralNet *nn);
void neuralnet_print_layer(const NeuralNet *nn, int layer_index);

#endif