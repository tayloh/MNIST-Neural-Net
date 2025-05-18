#include "neuralnet.h"

NeuralNet *neuralnet_create(int layer_input_size, int layer_output_size, int num_layers, int *layer_sizes)
{
}

void neuralnet_set_activation(NeuralNet *nn, float (*activation)(float))
{
    nn->activation = activation;
}

void neuralnet_set_activation_derivative(NeuralNet *nn, float (*activation_derivative)(float))
{
    nn->activation_derivative = activation_derivative;
}

float ReLU(float x)
{
    return x > 0.0f ? x : 0.0f;
}

float ReLU_derivative(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

void neuralnet_load_model(NeuralNet *nn, const char *model_fp)
{
}