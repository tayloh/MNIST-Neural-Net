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

    // TODO: Need to store activations and pre-activation values (activations and zs)

} NeuralNet;

// Layer 0 ------------ No biases

// W0      wwwwwwwwwwww Weights weights[0]

// Layer 1 ------------ Biases biases[0] <- OBS

// W1      wwwwwwwwwwww Weights weights[1]

// .
// .
// .

// WN      wwwwwwwwwwww Weights weights[num_layers - 2]

// Layer num_layers - 1 (output layer) Biases biases[num_layers - 1]

NeuralNet *neuralnet_create(int num_layers, int *layer_sizes);
void neuralnet_free(NeuralNet *nn);

void neuralnet_load_model(NeuralNet *nn, const char *model_fp);
void neuralnet_set_activation(NeuralNet *nn, float (*activation)(float));
void neuralnet_set_activation_derivative(NeuralNet *nn, float (*activation_derivative)(float));

float ReLU(float x);
float ReLU_derivative(float x);

void neuralnet_init_w_b_he(NeuralNet *nn); // TODO
void neuralnet_init_w_b_xavier(NeuralNet *nn);

void neuralnet_train(NeuralNet *nn, const Mnist *mnist);
void neuralnet_test(const NeuralNet *nn, const Mnist *mnist);

int neuralnet_infer(const NeuralNet *nn, const Vector *image);

void neuralnet_forward(NeuralNet *nn, const Vector *input); // TODO

void neuralnet_backprop(NeuralNet *nn, const Vector *target, Matrix **grad_w, Vector **grad_b);      // TODO
void neuralnet_update_weights(NeuralNet *nn, Matrix **grad_w, Vector **grad_b, float learning_rate); // TODO

float neuralnet_compute_loss(const Vector *output, const Vector *target);                // TODO
void neuralnet_compute_output_error(NeuralNet *nn, const Vector *target, Vector *delta); // TODO

#endif