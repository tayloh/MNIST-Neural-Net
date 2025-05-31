#include "neuralnet.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static void swap_vectors(Vector **a, Vector **b)
{
    Vector *temp = *a;
    *a = *b;
    *b = temp;
}

// Fisher-Yates shuffle
static void shuffle_data(Vector **inputs, Vector **targets, int num_samples)
{
    for (int i = num_samples - 1; i > 0; --i)
    {
        int j = rand() % (i + 1);
        swap_vectors(&inputs[i], &inputs[j]);
        swap_vectors(&targets[i], &targets[j]);
    }
}

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

static Vector *softmax(const Vector *v)
{
    Vector *result = linalg_vector_create(v->size);

    // When you compute exp(x_i) for values of x_i that are large, exp(x_i)
    // can easily overflow, i.e., exceed the largest number representable in floating point.
    // So, shift the exponential (rescale) by the max
    float max_val = linalg_vector_max(v);

    float sum = 0.0;
    for (int i = 0; i < v->size; i++)
    {
        result->data[i] = exp(v->data[i] - max_val);
        sum += result->data[i];
    }

    for (int i = 0; i < v->size; i++)
    {
        result->data[i] /= sum;
    }

    // All values sum to 1, can be used as probability distribution
    return result;
}

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

    nn->init_called = 0;
    nn->num_layers = num_layers;
    nn->layer_input_size = layer_sizes[0];
    nn->layer_output_size = layer_sizes[num_layers - 1];

    nn->layer_sizes = layer_sizes;

    // Set these after creating using separate function
    nn->activation = NULL;
    nn->activation_derivative = NULL;

    // ----- Alloc weights and biases -----
    nn->weights = (Matrix **)calloc(num_layers - 1, sizeof(Matrix *));
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

    // ----- Alloc grad_w and grad_b -----
    // Alloc grad_w and grad_b (they need to be exact same size as weights and biases resp.)
    nn->grad_w = (Matrix **)calloc(num_layers - 1, sizeof(Matrix *));
    nn->grad_b = (Vector **)calloc(num_layers - 1, sizeof(Vector *));
    if (!nn->grad_w || !nn->grad_b)
    {
        perror("calloc NeuralNet grad_w grad_b");
        free(nn->grad_w);
        free(nn->grad_b);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_layers - 1; ++i)
    {
        nn->grad_w[i] = linalg_matrix_create(layer_sizes[i + 1], layer_sizes[i]);
        nn->grad_b[i] = linalg_vector_create(layer_sizes[i + 1]);
    }

    // ----- Alloc activations (a) and pre-activaions (zs) -----
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

void neuralnet_save_model(NeuralNet *nn, const char *model_fp)
{
    // File format something like
    // First byte: number of layers L
    // L next bytes: layer sizes
    // float (4bytes) * layer sizes [0] ... float * layer sizes[L-2] next bytes: biases
    // float * layer sizes [1] * layer sizes[0] next bytes: weights
}

// Initialize weights and biases using He initialization
// Try with and without: without (random -1 to 1) converges much slower
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

    nn->init_called = 1;
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

    // This is working directly with the logits (output layer activation values) for now
    // We don't need softmax here...

    // Get final activations
    Vector *output = nn->activations[nn->num_layers - 1];

    // Find index with max value (argmax)
    int argmax = linalg_vector_argmax(output);

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

        // Don't use ReLU on the output layer (softmax CE needs raw logits)
        if (l < nn->num_layers - 2)
        {
            linalg_vector_apply(nn->activations[l + 1], nn->activation);
        }

        linalg_vector_free(z);
    }
}

float neuralnet_compute_softmax_CE(const Vector *output, const Vector *target)
{
    // Cross-entropy measures the difference between two probability distributions:
    //     The true distribution (your labels)
    //     The predicted distribution (your model's output)
    // For classification with a one-hot encoded label y and a predicted softmax output y^​, the cross-entropy is:
    // CE(y,y^)=−∑_i yi log⁡(y^i)
    // Since y is one-hot (e.g., 0, 0, 0, 1, 0), only one term matters:
    // CE=−log⁡(y^true class)
    // So it just penalizes how wrong the probability is for the correct class.

    if (!output || !target || !output->data || !target->data || output->size != target->size)
    {
        fprintf(stderr, "Error: neuralnet_compute_softmax_CE invalid input");
        exit(EXIT_FAILURE);
    }

    Vector *output_softmax = softmax(output);

    float loss = 0;
    for (int i = 0; i < target->size; ++i)
    {
        // Note: Will be mult by 0 for all but one target->data[i] (the one 1)
        loss += output_softmax->data[i] * target->data[i];
    }

    loss = -logf(loss);

    linalg_vector_free(output_softmax);

    return loss;
}

void neuralnet_backprop(NeuralNet *nn, const Vector *target)
{
    // --------- OUTPUT LAYER ---------
    // Output layer index
    int L = nn->num_layers - 1;

    // Predicted softmax output
    Vector *y_hat = softmax(nn->zs[L]);

    // We need del L/del z since it's the first part of the chain rule
    // For layer L we have: del L/del W = del L/del z * del z/del W
    // See https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication

    // a = y_hat here (activated z), y = target
    // del L/del z = del L/del a * del a/del z = a^L - y
    Vector *delta = linalg_vector_sub(y_hat, target);

    // For layer L we have: del L del b = del L/del z * del z/del b
    // Where del L/del z = a^L - y = delta (vector above)
    // and del z/del b where z = W^l * a^(l-1) + b^l is just = 1 (first term is constant in terms of b)
    // => del L del b = delta * 1
    linalg_vector_copy_into(nn->grad_b[L - 1], delta);

    // del L/del W^L = del L/del z * del z/del W^L = delta x a^(L-1) (outer product)
    // del z/del W^L = a^(L-1) (see backprop.md why)
    // grad_w_L is a matrix of size |delta| x |a^(L-1)| (so we have a gradient for each ingoing weight in the layer)
    // this is how it must be
    Matrix *grad_w_L = linalg_vector_outer_prod(delta, nn->activations[L - 1]);
    linalg_matrix_copy_into(nn->grad_w[L - 1], grad_w_L);
    linalg_matrix_free(grad_w_L);

    // --------- HIDDEN LAYERS ---------
    // I understand this on an ok level, but should do it with pen and paper
    for (int l = L - 1; l > 0; --l)
    {
        // The delta is what is important in every layer when computing
        // del L/del W^l = delta_l x a^(l-1) = grad_w for layer l
        // and del L/del b = delta_l = grad_b for layer l

        // If we have delta_l, then (right to left)
        // delta_(l-1) =  f'( z^(l-1) ) hadamard (W^l)^t * delta_l (*)

        // intermediate term = (W^l)^t * delta_l
        // weights[0] will never run here
        // layer 0 has no delta (no error signal for input layer, obviously)
        Matrix *w_l_transpose = linalg_matrix_transpose(nn->weights[l]);
        Vector *intermediate = linalg_vector_transform(w_l_transpose, delta);

        // f'(z^(l-1)), yes nn->zs[l] is correct... zs has one more element than
        // weights, so this is consistent with (*)
        // zs[l-1] is in other words not correct
        Vector *f_prim = linalg_vector_map(nn->zs[l], nn->activation_derivative);

        // delta^l = (W^{l})^T * delta^{l+1} ∘ f'(z^l)
        Vector *new_delta = linalg_vector_hadamard(intermediate, f_prim);

        // del L/del b = delta_l = grad_b for layer l
        // First iter: l = L - 1 = num_layers -1 -1 = num_layers - 2
        //             l - 1 = num_layers - 3 (final element of grad_b has index num_layers - 2)
        linalg_vector_copy_into(nn->grad_b[l - 1], new_delta);

        // del L/del W^l = delta_l x a^(l-1) = grad_w for layer l
        Matrix *grad_w_l = linalg_vector_outer_prod(new_delta, nn->activations[l - 1]);
        linalg_matrix_copy_into(nn->grad_w[l - 1], grad_w_l);

        linalg_vector_free(delta);
        linalg_vector_free(intermediate);
        linalg_matrix_free(w_l_transpose);
        linalg_vector_free(f_prim);
        linalg_matrix_free(grad_w_l);

        // Careful...
        delta = new_delta; // move down one layer
    }

    linalg_vector_free(delta);
    linalg_vector_free(y_hat);
}

void neuralnet_update_w_b(NeuralNet *nn, float learning_rate, float lambda)
{
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_update_w_b NULL nn");
        exit(EXIT_FAILURE);
    }

    int L = nn->num_layers - 1;
    for (int l = 0; l < L; ++l)
    {
        int layer_size = nn->layer_sizes[l + 1];

        // Update biases and weights between layer l and l+1
        for (int i = 0; i < layer_size; ++i)
        {
            nn->biases[l]->data[i] -= learning_rate * nn->grad_b[l]->data[i];

            int prev_layer_size = nn->layer_sizes[l];
            for (int j = 0; j < prev_layer_size; ++j)
            {
                // Weight decay: lambda * nn->weights[l]->data[i][j]
                nn->weights[l]->data[i][j] -= learning_rate * (nn->grad_w[l]->data[i][j] + lambda * nn->weights[l]->data[i][j]);
            }
        }
    }
}

void neuralnet_train(NeuralNet *nn, Vector **inputs, Vector **targets, int num_samples, int num_validation_samples, float learning_rate, int max_epochs, float lambda, int patience)
{

    // cba to check everything here...
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_train NULL nn");
        exit(EXIT_FAILURE);
    }
    // At least check members that don't get set in create()
    if (!nn->activation)
    {
        fprintf(stderr, "Error: neuralnet_train activation function is not set");
        exit(EXIT_FAILURE);
    }
    if (!nn->activation_derivative)
    {
        fprintf(stderr, "Error: neuralnet_train activation function derivative is not set");
        exit(EXIT_FAILURE);
    }
    if (!nn->init_called)
    {
        // Thought my implementation was broken for half an hour
        printf("Warning: weights and biases have not been initialized");
    }

    int num_train_samples = num_samples - num_validation_samples;

    printf("neuralnet_train: Started training...\n\n");
    neuralnet_print(nn);
    printf("num_train_samples          = %d\n", num_train_samples);
    printf("num_validation_samples     = %d\n", num_validation_samples);
    printf("learning_rate              = %.5f\n", learning_rate);
    printf("max_epochs                 = %d\n", max_epochs);
    printf("lambda (weight decay)      = %.5f\n", lambda);
    printf("patience (validation loss) = %d\n", patience);

    int epochs_without_improvement = 0;
    float best_validation_loss = 10000.0f;
    float current_validation_loss = 10000.0f;

    for (int e = 0; e < max_epochs; ++e)
    {
        printf("\n---Epoch %d---\n", e);

        shuffle_data(inputs, targets, num_train_samples);

        float total_epoch_loss = 0.0f;

        time_t epoch_start_time = time(NULL);
        int cursor_row = get_current_cursor_row();
        int last_percentage = -1;

        for (int i = 0; i < num_train_samples; ++i)
        {
            // clock_t start = clock();

            // Feed forward
            neuralnet_forward(nn, inputs[i]);
            //  ---

            // Get the raw logits at the output layer
            // Softmax-CE expects raw logits (applying ReLU first distorts the relative score between classes)
            Vector *output_layer_logits = nn->zs[nn->num_layers - 1];

            float loss = neuralnet_compute_softmax_CE(output_layer_logits, targets[i]);
            total_epoch_loss += loss;

            // Backpropagation
            neuralnet_backprop(nn, targets[i]);
            neuralnet_update_w_b(nn, learning_rate, lambda);
            //  ---
            // neuralnet_print_layer(nn, nn->num_layers - 1);

            // Only update progress bar if percentage changed
            int percentage = (i + 1) * 100 / num_train_samples;
            if (percentage > last_percentage)
            {
                draw_progress_bar(percentage, cursor_row);
                last_percentage = percentage;

                printf(" %d s", time(NULL) - epoch_start_time);
            }

            // clock_t end = clock();
            // printf("Single sample time: %.3f ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);
        }
        // After each epoch, chech average loss, and check validation loss (also average)

        // Compute loss on validation dataset
        // Indices for validation samples start where training samples end:
        for (int i = num_train_samples; i < num_train_samples + num_validation_samples; ++i)
        {
            // I think three places do this exact thing now.. code smell
            neuralnet_forward(nn, inputs[i]);
            Vector *output_layer_logits = nn->zs[nn->num_layers - 1];
            float loss = neuralnet_compute_softmax_CE(output_layer_logits, targets[i]);
            current_validation_loss += loss;
        }
        current_validation_loss /= num_validation_samples;

        // Log loss after each epoch
        float avg_epoch_loss = total_epoch_loss / num_train_samples;
        printf("\nTraining loss: %.4f | Validation loss: %.4f\n", avg_epoch_loss, current_validation_loss);

        // Early stop if validation loss is higher than best loss after "patience" num epochs
        if (current_validation_loss < best_validation_loss)
        {
            best_validation_loss = current_validation_loss;
            epochs_without_improvement = 0;
        }
        else
        {
            epochs_without_improvement += 1;
            if (epochs_without_improvement >= patience)
            {
                printf("\nValidation loss has not decreased for %d epochs, stopping...\n", patience);
                break; // early stop
            }
        }
    }
    printf("\nneuralnet_train: Training done!\n");
}

void neuralnet_test(NeuralNet *nn, Vector **inputs, Vector **targets, int num_samples)
{
    if (!nn)
    {
        fprintf(stderr, "Error: neuralnet_test NULL nn");
        exit(EXIT_FAILURE);
    }

    printf("neuralnet_test: Testing started...\n");
    printf("%d samples\n", num_samples);

    // Calculate both loss and percentage correct guesses
    int correct_predictions = 0;
    float total_loss = 0;
    for (int i = 0; i < num_samples; ++i)
    {
        // Run inference (includes forward pass)
        int prediction = neuralnet_infer(nn, inputs[i]);

        // Get index for the 1 in the one-hot target vector
        int target = linalg_vector_argmax(targets[i]);

        if (prediction == target)
        {
            correct_predictions++;
        }

        // Compute softmax CE
        Vector *output_layer_logits = nn->zs[nn->num_layers - 1];
        float loss = neuralnet_compute_softmax_CE(output_layer_logits, targets[i]);
        total_loss += loss;
    }
    float accuracy = (correct_predictions * 100.0f) / num_samples;
    float average_loss = total_loss / num_samples;

    printf("---\n");
    printf("Results\n");
    printf("Average loss: %.3f\n", average_loss);
    printf("Accuracy: %.2f%% (%d/%d correct)\n", accuracy, correct_predictions, num_samples);
    printf("---\n");
    printf("neuralnet_test: Testing done!\n");
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
    printf("weights[%d] = \n", layer_index - 1);
    printf("biases[%d] = \n", layer_index - 1);
    linalg_vector_print(nn->biases[layer_index - 1]);
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