#ifndef BMP_H
#define BMP_H

#include <stdio.h>

#include "../linalg/include/linalg.h"

// bmp.h bmp.c is only used for inference on user input bmps
// for training/testing data, see mnist.h mnist.c

// Tells the compiler to pack struct members with 1-byte alignment — i.e., no padding bytes between fields.
// push saves the previous packing setting (so it can be restored later with pop).
#pragma pack(push, 1)
typedef struct
{
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct
{
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

int bmp_read_file_header(FILE *f, BITMAPFILEHEADER *file_header);
int bmp_read_info_header(FILE *f, BITMAPINFOHEADER *info_header);

Matrix *bmp_create_matrix(const char *bmp_fp);

// Downsample using max pooling
// Produces too harsh edges
Matrix *bmp_downsample_maxpooling(Matrix *bmp_matrix, int to_height, int to_width);

// Downscale using boxfilter
// Produces too smooth edges
Matrix *bmp_downsample_boxfilter(Matrix *bmp_matrix, int to_height, int to_width);

void bmp_print(const char *filepath);

#endif