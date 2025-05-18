#ifndef MNIST_H
#define MNIST_H

#include "stdint.h"
#include "../linalg/include/linalg.h"

// Dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

enum
{
    MNIST_NUM_TEST = 10000,
    MNIST_NUM_TRAIN = 60000,
    MNIST_IMAGE_SIZE = 28 * 28,
    MNIST_IMAGE_ROWS = 28,
    MNIST_IMAGE_COLS = 28,
    MNIST_MAGIC_IMAGE = 2051,
    MNIST_MAGIC_LABEL = 2049
};

typedef struct
{
    int size;

    // Load each image as a vector with 28x28 size
    Vector **images;

    // One label for each image
    uint8_t *labels;

} Mnist;

Mnist *mnist_create(int size);
void mnist_load_images(Mnist *mnist, const char *mnist_images_fp);
void mnist_load_labels(Mnist *mnist, const char *mnist_labels_fp);
void mnist_free(Mnist *mnist);

void mnist_print_image(const Mnist *mnist, int index);
void mnist_print_image_x(const Mnist *mnist, int index);

#endif