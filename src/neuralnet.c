#include "neuralnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Returns a random float from sampled form a standard normal distribution (mean = 0, stddev = 1)
static float box_muller_sample()
{
    // We compute the inverse of the CDF for the standard distribution
    // CDF (x) of X = Probability that X is <= x
    // Inverse CDF (x) of X = Given a probability x, X is at most Inverse CDF (x)

    // But, the inverse of the CDF has no closed form solution (analytisk lösning)
    // Therefore, we instead use something called the Box-Muller transform
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

    // This implementation uses the "alternate form" (D. Knuth form)

    // The "spare" is the second value, z1
    static int has_spare = 0;
    static float spare;

    if (has_spare)
    {
        has_spare = 0;
        return spare;
    }

    has_spare = 1;

    float u, v, s;
    do
    {
        // Uniform distr u and v in range [-1, 1]
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        // s = R^2 = u^2 + v^2
        s = u * u + v * v;

        // Make sure we are inside of the unit circle
    } while (s >= 1.0f || s == 0.0f);

    // These are our sampled values
    float z0 = u * sqrtf(-2.0f * logf(s) / s);
    float z1 = v * sqrtf(-2.0f * logf(s) / s);

    spare = z1;
    return z0;
}

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

// ReLU is often preferred because:
//     Simple and fast: ReLU(x) = max(0, x) is cheap to compute.
//     Nonlinearity: Introduces nonlinearity, allowing neural nets to learn complex functions.
//     Avoids vanishing gradients (to some extent): Unlike sigmoid or tanh, ReLU doesn’t squash values into a tiny range, so the gradients don’t shrink as drastically.
//     Sparse activation: ReLU outputs zero for negative values, making the network more efficient and sparse.
// Dying ReLU problem: If too many neurons output zero, they might stop updating (because gradient is 0 for x < 0). Proper initialization (like He) helps mitigate this.
// Using He initialization is preferred with ReLU to prevent exploding or vanishing activations (empirically shown).

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

// Initialize weights and biases using He initialization
void neuralnet_init_w_b_he(NeuralNet *nn)
{
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_init_w_b NULL neuralnet");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nn->num_layers - 1; ++i)
    {
        // Fan-in is the number of input connections to a neuron.
        // In a fully connected layer, it’s simply the number of neurons in the previous layer.
        int fan_in = nn->layer_sizes[i];

        // Scale the initial weights to maintain a stable variance of the outputs as we go deeper into the network
        // He tries to preserve the variance of the activations
        float stddev = sqrtf(2.0f / fan_in);

        Matrix *w = nn->weights[i];
        for (int r = 0; r < w->rows; ++r)
        {
            for (int c = 0; c < w->columns; ++c)
            {
                w->data[r][c] = box_muller_sample() * stddev;
            }
        }

        // Zero initialize biases
        linalg_vector_fill(nn->biases[i], 0.0f);
    }
}