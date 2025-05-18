#include "mnist.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

static uint32_t read_uint32_big_endian(FILE *f)
{
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, f) != 4)
    {
        fprintf(stderr, "Error: read_uint32_big_endian failed to read 4 bytes from file.\n");
        exit(EXIT_FAILURE);
    }

    return ((uint32_t)bytes[0] << 24) |
           ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) |
           (uint32_t)bytes[3];
}

Mnist *mnist_create(int size)
{
    Mnist *mnist = (Mnist *)malloc(sizeof(Mnist));

    if (!mnist)
    {
        perror("malloc Mnist");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for each image-label pair
    mnist->size = size;
    mnist->labels = (uint8_t *)calloc(mnist->size, sizeof(uint8_t));
    mnist->images = (Vector **)calloc(mnist->size, sizeof(Vector *));

    if (!mnist->labels || !mnist->images)
    {
        perror("calloc Mnist labels/images");
        free(mnist->labels);
        free(mnist->images);
        free(mnist);
        exit(EXIT_FAILURE);
    }

    // Create and allocate image vector for each image
    for (int i = 0; i < mnist->size; ++i)
    {
        mnist->images[i] = linalg_vector_create(MNIST_IMAGE_SIZE);
    }

    return mnist;
}

void mnist_free(Mnist *mnist)
{
    if (!mnist)
    {
        return;
    }

    // Call free on each vector
    for (int i = 0; i < mnist->size; ++i)
    {
        linalg_vector_free(mnist->images[i]);
    }

    // Free allocated arrays
    free(mnist->images);
    free(mnist->labels);

    free(mnist);
}

void mnist_load_images(Mnist *mnist, const char *mnist_images_fp)
{
    FILE *f = fopen(mnist_images_fp, "rb");
    if (!f)
    {
        perror("fopen images");
        exit(EXIT_FAILURE);
    }

    // Read header (big-endian/network byte order, most significant byte on lower address/first)
    // Need to convert to little-endian
    // index 0:  4 bytes magic number
    // index 4:  4 bytes num images
    // index 8:  4 bytes num rows
    // index 12: 4 bytes num cols
    uint32_t magic = read_uint32_big_endian(f);
    uint32_t num_images = read_uint32_big_endian(f);
    uint32_t num_rows = read_uint32_big_endian(f);
    uint32_t num_cols = read_uint32_big_endian(f);

    if (magic != MNIST_MAGIC_IMAGE || num_rows != MNIST_IMAGE_ROWS || num_cols != MNIST_IMAGE_COLS || num_images != mnist->size)
    {
        fprintf(stderr, "Error: mnist_load_images invalid MNIST image file format or size mismatch\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // index 16: 784 bytes per image (1 byte per pixel, so don't need to worry about endianness)
    uint8_t buffer[MNIST_IMAGE_SIZE];
    for (int i = 0; i < mnist->size; ++i)
    {
        // Read one image at a time
        if (fread(buffer, 1, MNIST_IMAGE_SIZE, f) != MNIST_IMAGE_SIZE)
        {
            fprintf(stderr, "Error: mnist_load_images failed to read image %d\n", i);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        // Load it into mnist-images[i]->data (the vector)
        for (int j = 0; j < MNIST_IMAGE_SIZE; ++j)
        {
            mnist->images[i]->data[j] = buffer[j] / 255.0f; // normalize
        }
    }

    fclose(f);
}

void mnist_load_labels(Mnist *mnist, const char *mnist_labels_fp)
{
    FILE *f = fopen(mnist_labels_fp, "rb");
    if (!f)
    {
        perror("fopen labels");
        exit(EXIT_FAILURE);
    }

    // Read header (big-endian/network byte order, most significant byte on lower address/first)
    // Need to convert to little-endian
    // index 0:  4 bytes magic number
    // index 4:  4 bytes num images
    uint32_t magic = read_uint32_big_endian(f);
    uint32_t num_labels = read_uint32_big_endian(f);

    if (magic != MNIST_MAGIC_LABEL || num_labels != mnist->size)
    {
        fprintf(stderr, "Error: mnist_load_labels invalid MNIST label file format or size mismatch\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // No need to worry about endianness
    // index 8:  1 byte per label from here on out
    if (fread(mnist->labels, sizeof(uint8_t), mnist->size, f) != (size_t)mnist->size)
    {
        fprintf(stderr, "Error: mnist_load_labels failed to read labels\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);
}

void mnist_print_image(const Mnist *mnist, int index)
{
    if (!mnist || !mnist->images || !mnist->labels || !mnist->images[index] || !mnist->labels[index])
    {
        fprintf(stderr, "Error: mnist_print_image NULL mnist or mnist image");
        exit(EXIT_FAILURE);
    }

    printf("MNIST Label %d\n", mnist->labels[index]);
    printf("MNIST Image");

    for (int i = 0; i < MNIST_IMAGE_SIZE; ++i)
    {
        if (i % 28 == 0)
        {
            printf("\n");
        }
        printf("%.2f ", mnist->images[index]->data[i]);
    }

    printf("\n");
}

void mnist_print_image_x(const Mnist *mnist, int index)
{
    if (!mnist || !mnist->images || !mnist->labels || !mnist->images[index]->data)
    {
        fprintf(stderr, "Error: mnist_print_image NULL mnist or mnist image");
        exit(EXIT_FAILURE);
    }

    printf("MNIST Label %d\n", mnist->labels[index]);
    printf("MNIST Image");

    for (int i = 0; i < MNIST_IMAGE_SIZE; ++i)
    {
        if (i % 28 == 0)
        {
            printf("\n");
        }

        if (mnist->images[index]->data[i] > 0)
        {
            printf("%d", mnist->labels[index]);
        }
        else
        {
            printf(" ");
        }
    }

    printf("\n");
}