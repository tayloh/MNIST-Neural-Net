#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "../linalg/include/linalg.h"
#include "mnist.h"
#include "neuralnet.h"

void mnist_nn_training_driver()
{
    // ---- Load Mnist datasets ----
    printf("mnist_nn_training_driver: Starting training driver...\n\n");
    printf("Loading training dataset...");
    Mnist *mnist_train = mnist_create(MNIST_NUM_TRAIN);
    mnist_load_images(mnist_train, "./mnist-dataset/train-images.idx3-ubyte");
    mnist_load_labels(mnist_train, "./mnist-dataset/train-labels.idx1-ubyte");
    printf(" done!\n");

    printf("Loading testing dataset...");
    Mnist *mnist_test = mnist_create(MNIST_NUM_TEST);
    mnist_load_images(mnist_test, "./mnist-dataset/t10k-images.idx3-ubyte");
    mnist_load_labels(mnist_test, "./mnist-dataset/t10k-labels.idx1-ubyte");
    printf(" done!\n\n");

    // ---- Init neuralnet ----
    printf("Allocating neuralnet...");

    // EDIT NEURALNET AND LEARNING PARAMETERS HERE

    // Best config I've found so far is
    // layer_sizes = {784, 256, 10}
    // learning_rate = 0.01f
    // lambda/l2/weight decay = 0 (but for generalization, I guess pick 0.0001f)
    // patience = 3
    // epochs = 30

    // activation function: ReLU
    // loss function: Cross-Entropy
    // initialization: He

    // => Best accuracy on test dataset: 98.4 %
    srand((unsigned)time(NULL));
    int layer_sizes[] = {784, 256, 10};
    float learning_rate = 0.01f;
    float lambda = 0.0001f;
    int patience = 3;
    int max_epochs = 1;                 // 30
    int num_validation_samples = 10000; // 10000 => ~ 70 15 15 train val test split

    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    NeuralNet *nn = neuralnet_create(num_layers, layer_sizes);
    neuralnet_set_activation(nn, ReLU);
    neuralnet_set_activation_derivative(nn, ReLU_derivative);
    neuralnet_init_w_b_he(nn);

    printf(" done!\n");
    printf("\n");

    // Synthesize targets from the mnist structs
    // mnist stores labels as uint8, but nn expects each target to be a vector with the samme size
    // as its output layer
    // ----------------------------------
    Vector **targets_train = (Vector **)calloc(MNIST_NUM_TRAIN, sizeof(Vector *));
    Vector **targets_test = (Vector **)calloc(MNIST_NUM_TEST, sizeof(Vector *));
    if (!targets_train || !targets_test)
    {
        perror("calloc train/test targets");
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
    for (int i = 0; i < MNIST_NUM_TEST; ++i)
    {
        // Create a one-hot vector for each target (element at index=label is set to 1)
        targets_test[i] = linalg_vector_create(nn->layer_output_size);
        linalg_vector_fill(targets_test[i], 0.0f);

        int onehot_index = mnist_test->labels[i];
        targets_test[i]->data[onehot_index] = 1.0f;
    }
    // ----------------------------------

    // ---- Train and test neuralnet ----
    neuralnet_train(nn, mnist_train->images, targets_train, MNIST_NUM_TRAIN, num_validation_samples, learning_rate, max_epochs, lambda, patience);
    printf("\n");
    neuralnet_test(nn, mnist_test->images, targets_test, MNIST_NUM_TEST);

    //
    //
    //

    // This works now! nice, can build cli --train --test tomorrow
    // TODO: function to read training parameters from file
    // separate train and test into different functions here in main.c
    neuralnet_save_model(nn, "testmodel");
    NeuralNet *nn2 = neuralnet_load_model("testmodel");
    neuralnet_set_activation(nn2, ReLU); // backprop is only run during train so no need to set it

    neuralnet_print(nn2);
    neuralnet_test(nn2, mnist_test->images, targets_test, MNIST_NUM_TEST);

    // ---- Free train/test ----
    for (int i = 0; i < MNIST_NUM_TRAIN; ++i)
    {
        linalg_vector_free(targets_train[i]);
    }
    for (int i = 0; i < MNIST_NUM_TEST; ++i)
    {
        linalg_vector_free(targets_test[i]);
    }
    free(targets_train);
    free(targets_test);

    // ---- Free other ----
    mnist_free(mnist_train);
    mnist_free(mnist_test);

    neuralnet_free(nn);

    // return nn;
}

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
    mnist_load_images(mnist_train, "./mnist-dataset/train-images.idx3-ubyte");
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
    mnist_load_images(mnist_test, "./mnist-dataset/t10k-images.idx3-ubyte");
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

    // frees
    for (int i = 0; i < MNIST_NUM_TEST; ++i)
    {
        linalg_vector_free(targets_test[i]);
    }

    free(targets_test);
    mnist_free(mnist_test);
    neuralnet_free(nn);
}

void cli_infer()
{

    // Read model file
    // Convert image to 28x28
    // Run inference
}

void random_experimentation();

int main(int argc, char **argv)
{
    // mnist_nn -train modeldefs
    // mnist_nn -infer modelfile img

    if (argc == 1)
    {
        mnist_nn_training_driver();
        return 0;
    }

    const char *TRAIN = "--train";
    const char *TEST = "--test";
    const char *INFER = "--infer";
    const char *LIST = "--list";

    char *command = argv[1];

    if (strcmp(command, TRAIN) == 0)
    {
        // Is supposed to train a model given the parameters listed in
        // a modeldefs file

        if (argc != 4)
        {
            printf("Usage: mnist_nn --train params.txt modelfile");
            exit(EXIT_FAILURE);
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
            exit(EXIT_FAILURE);
        }

        char *modelfile_fp = argv[2];
        cli_test(modelfile_fp);
    }
    else if (strcmp(command, INFER) == 0)
    {
        // Is supposed to infer a number based on the input modelfile and .bmp image
        printf("-infer is not yet implemented.");
    }
    else if (strcmp(command, LIST) == 0)
    {
        // Is supposed to list the metadata inside of the modelfile
        // Training parameters, layer sizes, etc
    }
    else
    {
        printf("Usage:\n");
        printf("mnist_nn --train paramsfile.txt\n");
        printf("mnist_nn --test modelfile\n");
        printf("mnist_nn --infer modelfile img.bmp\n");
        printf("mnist_nn --list modelfile");
    }

    return 0;
}

void random_experimentation()
{

    srand(time(NULL));

    // LINALG
    Vector *v1 = linalg_vector_create(3);
    Vector *v2 = linalg_vector_create(3);

    linalg_vector_fill(v1, 2.0f);
    linalg_vector_fill(v2, 3.0f);

    linalg_vector_print(v1);
    linalg_vector_print(v2);

    float dot = linalg_vector_dot(v1, v2);
    printf("Dot product: %.2f\n", dot);

    linalg_vector_normalize(v1);
    linalg_vector_print(v1);

    linalg_vector_free(v1);
    linalg_vector_free(v2);

    Matrix *m = linalg_matrix_identity(4);
    linalg_matrix_print(m);

    Matrix *m1 = linalg_matrix_create(2, 3);
    m1->data[0][0] = 1;
    m1->data[0][1] = 2;
    m1->data[0][2] = 3;
    m1->data[1][0] = 4;
    m1->data[1][1] = 5;
    m1->data[1][2] = 6;
    linalg_matrix_print(m1);
    Matrix *m1_transpose = linalg_matrix_transpose(m1);
    linalg_matrix_print(m1_transpose);

    linalg_matrix_free(m);
    linalg_matrix_free(m1);
    linalg_matrix_free(m1_transpose);

    Matrix *a = linalg_matrix_create(2, 3);
    Matrix *b = linalg_matrix_create(3, 2);
    a->data[0][0] = 1;
    a->data[0][1] = 2;
    a->data[0][2] = 3;
    a->data[1][0] = 4;
    a->data[1][1] = 5;
    a->data[1][2] = 6;

    b->data[0][0] = 7;
    b->data[0][1] = 8;
    b->data[1][0] = 9;
    b->data[1][1] = 10;
    b->data[2][0] = 11;
    b->data[2][1] = 12;
    Matrix *ab = linalg_matrix_multiply(a, b);
    linalg_matrix_print(ab);

    Vector *row1 = linalg_matrix_get_row(ab, 1);
    Vector *col0 = linalg_matrix_get_column(ab, 0);
    linalg_vector_print(row1);
    linalg_vector_print(col0);
    linalg_vector_free(row1);
    linalg_vector_free(col0);

    linalg_matrix_free(a);
    linalg_matrix_free(b);
    linalg_matrix_free(ab);

    Matrix *I3 = linalg_matrix_identity(3);
    linalg_matrix_scale(I3, 2);
    Vector *v3 = linalg_vector_create(3);
    linalg_vector_fill(v3, 3);
    Vector *transformed = linalg_vector_transform(I3, v3);
    linalg_vector_print(transformed);

    linalg_matrix_free(I3);

    // MNIST
    Mnist *mnist = mnist_create(MNIST_NUM_TEST);
    mnist_load_images(mnist, "./mnist-dataset/t10k-images.idx3-ubyte");
    mnist_load_labels(mnist, "./mnist-dataset/t10k-labels.idx1-ubyte");
    mnist_print_image_x(mnist, MNIST_NUM_TEST - 1);

    // NEURALNET
    int nn_layers[5] = {MNIST_IMAGE_SIZE, 2, 5, 8, 10};
    NeuralNet *nn = neuralnet_create(5, nn_layers);
    neuralnet_set_activation(nn, ReLU);
    neuralnet_set_activation_derivative(nn, ReLU_derivative);

    neuralnet_init_w_b_he(nn);

    neuralnet_print(nn);
    neuralnet_print_layer(nn, 2);

    int inferred_value = neuralnet_infer(nn, mnist->images[MNIST_NUM_TEST - 1]);
    printf("NeuralNet inferred %d\n", inferred_value);
    Vector *target = linalg_vector_create(10);
    linalg_vector_fill(target, 0.0f);
    target->data[mnist->labels[MNIST_NUM_TEST - 1]] = 1.0f;
    printf("Loss: %.3f", neuralnet_compute_softmax_CE(nn->zs[nn->num_layers - 1], target));

    // neuralnet_train(nn, mnist);

    linalg_vector_free(target);
    mnist_free(mnist);
    neuralnet_free(nn);
}
