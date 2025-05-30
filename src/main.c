#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../linalg/include/linalg.h"
#include "mnist.h"
#include "neuralnet.h"

int main(int argc, char **argv)
{
    printf("MNIST Neural net\n");

    srand(time(NULL));

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

    Mnist *mnist = mnist_create(MNIST_NUM_TEST);
    mnist_load_images(mnist, "./mnist-dataset/t10k-images.idx3-ubyte");
    mnist_load_labels(mnist, "./mnist-dataset/t10k-labels.idx1-ubyte");
    mnist_print_image_x(mnist, MNIST_NUM_TEST - 1);

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

    neuralnet_train(nn, mnist);

    linalg_vector_free(target);
    mnist_free(mnist);
    neuralnet_free(nn);

    return 0;
}
