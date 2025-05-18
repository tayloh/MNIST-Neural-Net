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

} NeuralNet;

NeuralNet *neuralnet_create(int layer_input_size, int layer_output_size, int num_hidden_layers, int *layer_sizes);

void neuralnet_load_model(NeuralNet *nn, const char *model_fp);
void neuralnet_set_activation(NeuralNet *nn, float (*activation)(float));
void neuralnet_set_activation_derivative(NeuralNet *nn, float (*activation_derivative)(float));

float ReLU(float x);
float ReLU_derivative(float x);

void neuralnet_init_w_b(NeuralNet *nn);

void neuralnet_train(NeuralNet *nn, const Mnist *mnist);
void neuralnet_test(const NeuralNet *nn, const Mnist *mnist);

int neuralnet_infer(const NeuralNet *nn, const Vector *image);

#endif