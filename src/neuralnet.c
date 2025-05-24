#include "neuralnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Don't touch biases[0], input layer has no bias vector
NeuralNet *neuralnet_create(int num_layers, int *layer_sizes)
{

    if (num_layers < 2 || !layer_sizes)
    {
        fprintf(stderr, "Error: neuralnet_create invalid neural net configuration.\n");
        exit(EXIT_FAILURE);
    }

    NeuralNet *nn = (NeuralNet *)malloc(sizeof(NeuralNet));

    if (!nn)
    {
        perror("malloc NeuralNet");
        exit(EXIT_FAILURE);
    }

    nn->num_layers = num_layers;
    nn->layer_input_size = layer_sizes[0];
    nn->layer_output_size = layer_sizes[num_layers - 1];

    nn->layer_sizes = layer_sizes;

    // Set these after creating using separate function
    nn->activation = NULL;
    nn->activation_derivative = NULL;

    // Alloc Weights
    nn->weights = (Matrix **)calloc(num_layers - 1, sizeof(Matrix *));

    // Alloc biases
    nn->biases = (Vector **)calloc(num_layers - 1, sizeof(Vector *));

    if (!nn->weights || !nn->biases)
    {
        perror("calloc NeuralNet weights/biases");
        free(nn->weights);
        free(nn->biases);
        free(nn);
        exit(EXIT_FAILURE);
    }

    // Number of weight matrices is 1 less than number of layers
    // length of layer_sizes = num_layers
    for (int i = 0; i < num_layers - 1; ++i)
    {
        nn->weights[i] = linalg_matrix_create(layer_sizes[i], layer_sizes[i + 1]);
        nn->biases[i] = linalg_vector_create(layer_sizes[i + 1]);
    }

    return nn;
}

void neuralnet_free(NeuralNet *nn)
{
    if (!nn)
    {
        return;
    }

    // Call free on matrices and biases
    for (int i = 0; i < nn->num_layers - 1; ++i)
    {
        linalg_matrix_free(nn->weights[i]);

        // No vector was allocated on index 0
        linalg_vector_free(nn->biases[i]);
    }

    free(nn->weights);
    free(nn->biases);

    free(nn);
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

void neuralnet_init_w_b_he(NeuralNet *nn)
{
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_init_w_b NULL neuralnet");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nn->num_layers - 1; ++i)
    {
        linalg_vector_fill(nn->biases[i], 0.0f);
    }

    // TODO: He Init for weights...
}