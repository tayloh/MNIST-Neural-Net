#include "bmp.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int bmp_read_file_header(FILE *f, BITMAPFILEHEADER *file_header)
{
    if (fread(file_header, sizeof(BITMAPFILEHEADER), 1, f) != 1)
    {
        fprintf(stderr, "Error: bmp_read_file_header read BITMAPFILEHEADER\n");
        return 1;
    }

    if (file_header->bfType != 0x4D42) // 'BM'
    {
        fprintf(stderr, "Error: bmp_read_file_header file is not a bmp\n");
        return 2;
    }

    return 0;
}

int bmp_read_info_header(FILE *f, BITMAPINFOHEADER *info_header)
{
    if (fread(info_header, sizeof(BITMAPINFOHEADER), 1, f) != 1)
    {
        fprintf(stderr, "Error: bmp_read_info_header read BITMAPINFOHEADER\n");
        return 1;
    }

    if (info_header->biBitCount != 8)
    {
        fprintf(stderr, "Error: bmp_read_info_header only 8-bit bmps are supported (got %d-bit)\n", info_header->biBitCount);
        return 2;
    }

    return 0;
}

Matrix *bmp_create_matrix(const char *bmp_fp)
{

    FILE *f = fopen(bmp_fp, "rb");
    if (!f)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    int ret = 0;

    BITMAPFILEHEADER file_header;
    ret |= bmp_read_file_header(f, &file_header);

    BITMAPINFOHEADER info_header;
    ret |= bmp_read_info_header(f, &info_header);

    if (ret != 0)
    {
        fprintf(stderr, "Error: bmp_create_matrix issue when reading bmp headers");
        exit(EXIT_FAILURE);
    }

    int bmp_width = info_header.biWidth;

    // Height can be negative in bmps to indicate that the image is stored top-down instead of bottom-up
    int bmp_height = abs(info_header.biHeight);
    int is_top_down = info_header.biHeight < 0 ? 1 : 0;

    fseek(f, file_header.bfOffBits, SEEK_SET);

    // Pads to 4-byte multiple
    // Add 3 to make sure we get the upper boundary (otherwise e.g. 35 => 32 bytes but it has to be 36)
    int row_size = ((info_header.biWidth + 3) / 4) * 4;
    uint8_t row_buffer[row_size];

    Matrix *result = linalg_matrix_create(bmp_width, bmp_height);

    if (is_top_down)
    {
        for (int y = 0; y < bmp_height; ++y)
        {

            if (fread(row_buffer, 1, row_size, f) != row_size)
            {
                fprintf(stderr, "Error: bmp_create_vector read bmp pixel row");

                fclose(f);
                exit(EXIT_FAILURE);
            }

            for (int x = 0; x < bmp_width; x++)
            {
                result->data[y][x] = row_buffer[x];
            }
        }
    }
    else
    {
        // bmp stores pixels bottom-to-top by default
        for (int y = bmp_height - 1; y >= 0; --y)
        {
            if (fread(row_buffer, 1, row_size, f) != row_size)
            {
                fprintf(stderr, "Error: bmp_create_vector read bmp pixel row");

                fclose(f);
                exit(EXIT_FAILURE);
            }

            for (int x = 0; x < bmp_width; x++)
            {
                result->data[y][x] = row_buffer[x];
            }
        }
    }

    fclose(f);
    return result;
}

Matrix *bmp_downsample_maxpooling(Matrix *bmp_matrix, int to_height, int to_width)
{
    int src_w = bmp_matrix->columns;
    int src_h = bmp_matrix->rows;

    float scale_x = (float)src_w / to_width;
    float scale_y = (float)src_h / to_height;

    Matrix *result = linalg_matrix_create(to_height, to_width);

    for (int y = 0; y < to_height; y++)
    {
        for (int x = 0; x < to_width; x++)
        {

            // Define box indices
            int x_start = (int)(x * scale_x);
            int x_end = (int)((x + 1) * scale_x);
            int y_start = (int)(y * scale_y);
            int y_end = (int)((y + 1) * scale_y);

            if (x_end >= src_w)
            {
                x_end = src_w - 1;
            }
            if (y_end >= src_h)
            {
                y_end = src_h - 1;
            }

            // Maxpooling inside of the box
            float max_val = 0;
            for (int j = y_start; j <= y_end; j++)
            {
                for (int i = x_start; i <= x_end; i++)
                {
                    float val = bmp_matrix->data[j][i];
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                }
            }

            result->data[y][x] = max_val;
        }
    }

    return result;
}

Matrix *bmp_downsample_boxfilter(Matrix *bmp_matrix, int to_height, int to_width)
{
    int src_w = bmp_matrix->columns;
    int src_h = bmp_matrix->rows;

    float scale_x = (float)src_w / to_width;
    float scale_y = (float)src_h / to_height;

    Matrix *result = linalg_matrix_create(to_height, to_width);

    for (int y = 0; y < to_height; y++)
    {
        for (int x = 0; x < to_width; x++)
        {

            // Define box indices
            int x_start = (int)(x * scale_x);
            int x_end = (int)((x + 1) * scale_x);
            int y_start = (int)(y * scale_y);
            int y_end = (int)((y + 1) * scale_y);

            if (x_end >= src_w)
            {
                x_end = src_w - 1;
            }
            if (y_end >= src_h)
            {
                y_end = src_h - 1;
            }

            // Boxfilter
            float sum = 0;
            int count = 0;
            for (int j = y_start; j <= y_end; j++)
            {
                for (int i = x_start; i <= x_end; i++)
                {
                    float val = bmp_matrix->data[j][i];
                    sum += val;

                    count++;
                }
            }

            result->data[y][x] = sum / count;
        }
    }

    return result;
}

void bmp_print(const char *filepath)
{
    Matrix *bmp_matrix = bmp_create_matrix(filepath);

    printf("BMP as ASCII\n");

    for (int i = 0; i < bmp_matrix->rows; ++i)
    {
        for (int j = 0; j < bmp_matrix->columns; ++j)
        {
            if (bmp_matrix->data[i][j] > 0)
            {
                printf("x");
            }
            else
            {
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("\n");

    linalg_matrix_free(bmp_matrix);
}
