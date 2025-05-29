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
        // If layer 0 has 2 neurons and layer 1 has 5 neurons, then we have
        // Matrix_5x2
        // [   1.079    0.542 ]
        // [   0.646   -0.197 ]
        // [   1.225    0.848 ]
        // [  -0.157   -1.124 ]
        // [   0.575    0.346 ]
        // That is, row 1 holds the incoming weights for neuron 1 in layer 1, and so on
        nn->weights[i] = linalg_matrix_create(layer_sizes[i + 1], layer_sizes[i]);
        nn->biases[i] = linalg_vector_create(layer_sizes[i + 1]);
    }

    // Alloc activations
    nn->activations = (Vector **)calloc(num_layers, sizeof(Vector *));

    // Alloc pre activations
    nn->zs = (Vector **)calloc(num_layers, sizeof(Vector *));

    if (!nn->activations || !nn->zs)
    {
        perror("calloc NeuralNet activations/zs");
        free(nn->activations);
        free(nn->zs);
        free(nn);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_layers; ++i)
    {
        nn->activations[i] = linalg_vector_create(layer_sizes[i]);
        nn->zs[i] = linalg_vector_create(layer_sizes[i]);
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
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_set_activation NULL neuralnet");
        exit(EXIT_FAILURE);
    }

    nn->activation = activation;
}

void neuralnet_set_activation_derivative(NeuralNet *nn, float (*activation_derivative)(float))
{
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_set_activation_derivative NULL neuralnet");
        exit(EXIT_FAILURE);
    }

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

int neuralnet_infer(NeuralNet *nn, const Vector *input)
{
    if (!nn || !input)
    {
        fprintf(stderr, "Error: neuralnet_infer NULL nn or input.\n");
        exit(EXIT_FAILURE);
    }

    // nn->activations never depend on previous nn->activations values
    // so this is fine
    neuralnet_forward(nn, input);

    // Get final activations
    Vector *output = nn->activations[nn->num_layers - 1];

    // Find index with max value (argmax)
    int argmax = 0;
    float max = output->data[0];

    for (int i = 1; i < output->size; ++i)
    {
        if (output->data[i] > max)
        {
            max = output->data[i];
            argmax = i;
        }
    }

    // Since output should be 0-9, I can just take the argmax directly
    return argmax;
}

void neuralnet_forward(NeuralNet *nn, const Vector *input)
{
    if (!nn || !input)
    {
        fprintf(stderr, "Error: neuralnet_forward NULL nn or input.\n");
        exit(EXIT_FAILURE);
    }

    // Copy input to activations[0]
    linalg_vector_copy_into(nn->activations[0], input);
    linalg_vector_copy_into(nn->zs[0], input);

    for (int l = 0; l < nn->num_layers - 1; ++l)
    {
        // Think of l as the previous layer (size of weights and biases is one less than num layers)
        // l+1 is the current layer we are doing activations for

        // z = weights[l] (matrix) * a[l] (vector) + bias[l] (vector)
        Vector *z = linalg_vector_transform(nn->weights[l], nn->activations[l]);
        linalg_vector_add_into(z, nn->biases[l]);

        // Store zs for backpropagation
        linalg_vector_copy_into(nn->zs[l + 1], z);

        // Apply activation on current layer (memcpy is fast... so let's leave it as two calls)
        // Optimize if I ever process larger vectors... (also, remove all the if checks and so on.. in my linalg lib)
        linalg_vector_copy_into(nn->activations[l + 1], z);

        linalg_vector_apply(nn->activations[l + 1], nn->activation);

        linalg_vector_free(z);
    }
}

void neuralnet_print_layer(const NeuralNet *nn, int layer_index)
{
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_print NULL neuralnet");
        exit(EXIT_FAILURE);
    }

    if (layer_index >= nn->num_layers)
    {
        fprintf(stderr, "Error: neuralnet_print_layer layer_index out of bounds");
        exit(EXIT_FAILURE);
    }

    if (layer_index == 0)
    {
        printf("NeuralNet layer 0 (input) has no associated weights or biases");
        return;
    }

    printf("NeuralNet layer %d \n", layer_index);
    printf("biases[%d] = %3.3f\n", layer_index - 1, nn->biases[layer_index - 1]);
    printf("weights[%d] = \n", layer_index - 1);
    linalg_matrix_print(nn->weights[layer_index - 1]);
}

void neuralnet_print(const NeuralNet *nn)
{
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_print NULL neuralnet");
        exit(EXIT_FAILURE);
    }

    printf("NeuralNet ");

    printf("[");
    for (int i = 0; i < nn->num_layers - 1; ++i)
    {
        printf("%d, ", nn->layer_sizes[i]);
    }
    printf("%d", nn->layer_sizes[nn->num_layers - 1]);
    printf("]\n");
}