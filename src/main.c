#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "../linalg/include/linalg.h"
#include "mnist.h"
#include "neuralnet.h"
#include "bmp.h"

#define USE_BINARY_COLORS 1

void cli_train(const char *params_fp, const char *model_fp)
{

    // Should break this into its own function (the param file read)... but I cba
    // to make a struct for training parameters to pass around...

    // int layer_sizes[MAX_LAYERS];
    int nlayers = 0;
    float learning_rate = 0.0f;
    float lambda = 0.0f;
    int patience = 0;
    int max_epochs = 0;
    int num_validation_samples = 0;

    // ---------------------------------------
    // --- Read parameters file --------------
    // ---------------------------------------
    FILE *f = fopen(params_fp, "r");
    if (!f)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    const int MAX_LENGTH = 256;
    const int MAX_LAYERS = 100;

    char line[MAX_LENGTH];

    // Collect layer_sizes in the while loop and then copy over to smaller array
    int layer_sizes_collect[MAX_LAYERS];

    while (fgets(line, sizeof(line), f))
    {

        // Strip newline
        // 0 sets '\0' which terminates the string
        line[strcspn(line, "\n")] = 0;

        // Split line on first space
        char *key = strtok(line, " ");
        char *value = strtok(NULL, "");

        if (!key || !value)
        {
            continue;
        }

        // Read params
        if (strcmp(key, "nlayers") == 0)
        {
            nlayers = atoi(value);
        }
        else if (strcmp(key, "learning_rate") == 0)
        {
            learning_rate = strtof(value, NULL);
        }
        else if (strcmp(key, "lambda") == 0)
        {
            lambda = strtof(value, NULL);
        }
        else if (strcmp(key, "patience") == 0)
        {
            patience = atoi(value);
        }
        else if (strcmp(key, "max_epochs") == 0)
        {
            max_epochs = atoi(value);
        }
        else if (strcmp(key, "num_validation_samples") == 0)
        {
            num_validation_samples = atoi(value);
        }

        if (strcmp(key, "layers") == 0)
        {
            char *token = strtok(value, ",");
            for (int i = 0; token && i < nlayers; ++i)
            {
                layer_sizes_collect[i] = atoi(token);
                token = strtok(NULL, ",");
            }
        }
    }

    fclose(f);

    // Set layer_sizes here
    int layer_sizes[nlayers];
    memcpy(layer_sizes, layer_sizes_collect, nlayers * sizeof(int));

    printf("Read parameters file...\n");
    printf("nlayers                %d\n", nlayers);
    printf("layers                 ");
    for (int i = 0; i < nlayers; ++i)
    {
        printf("%d ", layer_sizes[i]);
    }
    printf("\n");
    printf("learning_rate          %f\n", learning_rate);
    printf("lambda                 %f\n", lambda);
    printf("patience               %d\n", patience);
    printf("max_epochs             %d\n", max_epochs);
    printf("num_validation_samples %d\n", num_validation_samples);

    // ---------------------------------------
    // --- Train modell ----------------------
    // ---------------------------------------

    // ---- Load Mnist datasets ----
    printf("\nLoading training dataset...");
    Mnist *mnist_train = mnist_create(MNIST_NUM_TRAIN);
    mnist_load_images(mnist_train, "./mnist-dataset/train-images.idx3-ubyte", USE_BINARY_COLORS);
    mnist_load_labels(mnist_train, "./mnist-dataset/train-labels.idx1-ubyte");
    printf(" done!\n\n");

    // ---- Allocate neuralnet ----
    NeuralNet *nn = neuralnet_create(nlayers, layer_sizes);
    neuralnet_set_activation(nn, ReLU);
    neuralnet_set_activation_derivative(nn, ReLU_derivative);
    neuralnet_init_w_b_he(nn);

    // ---- Synthesize one-hot vectors from mnist ----
    Vector **targets_train = (Vector **)calloc(MNIST_NUM_TRAIN, sizeof(Vector *));
    if (!targets_train)
    {
        perror("calloc train targets");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < MNIST_NUM_TRAIN; ++i)
    {
        // Create a one-hot vector for each target (element at index=label is set to 1)
        targets_train[i] = linalg_vector_create(nn->layer_output_size);
        linalg_vector_fill(targets_train[i], 0.0f);

        int onehot_index = mnist_train->labels[i];
        targets_train[i]->data[onehot_index] = 1.0f;
    }

    neuralnet_train(nn, mnist_train->images, targets_train, MNIST_NUM_TRAIN, num_validation_samples, learning_rate, max_epochs, lambda, patience);

    // ---------------------------------------
    // --- Save model ------------------------
    // ---------------------------------------
    neuralnet_save_model(nn, model_fp);

    // ---------------------------------------
    // --- Frees -----------------------------
    // ---------------------------------------
    for (int i = 0; i < MNIST_NUM_TRAIN; ++i)
    {
        linalg_vector_free(targets_train[i]);
    }

    free(targets_train);
    mnist_free(mnist_train);
    neuralnet_free(nn);
}

void cli_test(const char *model_fp)
{
    // Read modelfile
    printf("Loading modelfile...");

    NeuralNet *nn = neuralnet_load_model(model_fp);
    neuralnet_set_activation(nn, ReLU);

    printf(" done!\n\n");
    printf("Allocated ");
    neuralnet_print(nn);
    printf("\n");

    // Read test dataset
    printf("Loading testing dataset...");
    Mnist *mnist_test = mnist_create(MNIST_NUM_TEST);
    mnist_load_images(mnist_test, "./mnist-dataset/t10k-images.idx3-ubyte", USE_BINARY_COLORS);
    mnist_load_labels(mnist_test, "./mnist-dataset/t10k-labels.idx1-ubyte");
    printf(" done!\n\n");

    // Synthesize targets from the mnist structs
    // mnist stores labels as uint8, but nn expects each target to be a vector with the samme size
    // as its output layer
    // ----------------------------------
    Vector **targets_test = (Vector **)calloc(MNIST_NUM_TEST, sizeof(Vector *));
    if (!targets_test)
    {
        perror("calloc test targets");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < MNIST_NUM_TEST; ++i)
    {
        // Create a one-hot vector for each target (element at index=label is set to 1)
        targets_test[i] = linalg_vector_create(nn->layer_output_size);
        linalg_vector_fill(targets_test[i], 0.0f);

        int onehot_index = mnist_test->labels[i];
        targets_test[i]->data[onehot_index] = 1.0f;
    }
    // ----------------------------------

    // Test model
    neuralnet_test(nn, mnist_test->images, targets_test, MNIST_NUM_TEST);

    // Frees
    for (int i = 0; i < MNIST_NUM_TEST; ++i)
    {
        linalg_vector_free(targets_test[i]);
    }

    free(targets_test);
    mnist_free(mnist_test);
    neuralnet_free(nn);
}

void prettyprint_scaled_bmp_vector(const Vector *bmp_vector)
{
    for (int i = 0; i < bmp_vector->size; ++i)
    {
        if (i % MNIST_IMAGE_COLS == 0)
        {
            printf("\n");
        }

        if (bmp_vector->data[i] == 0)
        {
            printf("=");
        }
        else
        {
            printf(" ");
        }
    }
}

float invert_bmp_color(float grayscale_color)
{
    return 1.0f - grayscale_color;
}

void cli_infer(const char *model_fp, const char *bmp_fp)
{
    // Read bmp into a matrix
    Matrix *bmp_matrix = bmp_create_matrix(bmp_fp);

    // Normalize values
    linalg_matrix_scale(bmp_matrix, 1.0f / 255.0f);

    // MNIST images have black background and white digits
    // So we need to invert the input (assume digits are black and background is white)
    linalg_matrix_apply(bmp_matrix, invert_bmp_color);

    // Downsample to 28x28
    Matrix *bmp_matrix_downsampled = bmp_downsample_maxpooling(bmp_matrix, MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS);

    // Flatten to array for input to nn
    Vector *nn_input = linalg_matrix_flatten(bmp_matrix_downsampled);
    // linalg_vector_scale(nn_input, 1.0f / 255.0f);

    // Read modelfile and allocate nn
    NeuralNet *nn = neuralnet_load_model(model_fp);
    neuralnet_set_activation(nn, ReLU);

    // Run inference
    int prediction = neuralnet_infer(nn, nn_input);

    printf("Input bmp (%s) as ASCII:", bmp_fp);
    prettyprint_scaled_bmp_vector(nn_input);

    printf("\n\nNeuralNet prediction: %d", prediction);

    linalg_matrix_free(bmp_matrix);
    linalg_matrix_free(bmp_matrix_downsampled);
    linalg_vector_free(nn_input);
    neuralnet_free(nn);
}

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));

    // mnist_nn --train params.txt modelfile
    // mnist_nn --test modelfile
    // mnist_nn --infer img.bmp modelfile

    if (argc == 1)
    {
        printf("Usage:\n");
        printf("mnist_nn --train paramsfile.txt\n");
        printf("mnist_nn --test modelfile\n");
        printf("mnist_nn --infer modelfile img.bmp\n");
        return 0;
    }

    const char *TRAIN = "--train";
    const char *TEST = "--test";
    const char *INFER = "--infer";

    char *command = argv[1];

    if (strcmp(command, TRAIN) == 0)
    {
        // Is supposed to train a model given the parameters listed in
        // a params.txt file

        if (argc != 4)
        {
            printf("Usage: mnist_nn --train params.txt modelfile");
            return 1;
        }

        char *params_fp = argv[2];
        char *model_fp = argv[3];
        cli_train(params_fp, model_fp);
    }
    else if (strcmp(command, TEST) == 0)
    {
        // Is supposed to test a model given a modelfile

        if (argc != 3)
        {
            printf("Usage: mnist_nn --test modelfile");
            return 2;
        }

        char *modelfile_fp = argv[2];
        cli_test(modelfile_fp);
    }
    else if (strcmp(command, INFER) == 0)
    {
        // Is supposed to infer a number based on the input modelfile and .bmp image

        if (argc != 4)
        {
            printf("Usage: mnist_nn --infer img.bmp modelfile");
            return 3;
        }

        char *bmp_fp = argv[2];
        char *model_fp = argv[3];
        cli_infer(model_fp, bmp_fp);
    }
    else
    {
        printf("Usage:\n");
        printf("mnist_nn --train params.txt modelfile\n");
        printf("mnist_nn --test modelfile\n");
        printf("mnist_nn --infer img.bmp modelfile\n");
    }

    return 0;
}