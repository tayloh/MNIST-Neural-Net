#include "../include/linalg.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Matrix *linalg_matrix_create(int rows, int columns)
{
    if (rows < 1 || columns < 1)
    {
        fprintf(stderr, "Error: matrix_create rows or columns are less than 1.\n");
        exit(EXIT_FAILURE);
    }

    // Allocates 32 bit (int) + 32 bit (int) + 64 bit (memory address on 64 bit system)
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));

    if (!m)
    {
        perror("malloc Matrix");
        exit(EXIT_FAILURE);
    }
    m->rows = rows;
    m->columns = columns;

    m->data = (float **)calloc(rows, sizeof(float *));

    if (!m->data)
    {
        perror("calloc Matrix data");
        free(m);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; ++i)
    {
        m->data[i] = (float *)calloc(columns, sizeof(float));
        if (!m->data[i])
        {
            perror("calloc Matrix row");

            // Free previously allocated rows
            for (int j = 0; j < i; ++j)
            {
                free(m->data[j]);
            }
            free(m->data);
            free(m);
            exit(EXIT_FAILURE);
        }
    }

    return m;
}

Matrix *linalg_matrix_identity(int n)
{
    Matrix *identity = linalg_matrix_create(n, n);
    for (int i = 0; i < n; ++i)
    {
        identity->data[i][i] = 1.0f;
    }

    return identity;
}

void linalg_matrix_free(Matrix *m)
{
    if (!m)
    {
        return;
    }

    // Free each row
    for (int i = 0; i < m->rows; ++i)
    {
        free(m->data[i]);
    }

    // Free the array of row pointers
    free(m->data);

    // Free the Matrix struct itself
    free(m);
}

void linalg_matrix_fill(Matrix *m, float value)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: matrix_fill NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m->rows; ++i)
    {
        for (int j = 0; j < m->columns; ++j)
        {
            m->data[i][j] = value;
        }
    }
}

void linalg_matrix_print(const Matrix *m)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: matrix_print NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    printf("Matrix_%dx%d\n", m->rows, m->columns);
    for (int i = 0; i < m->rows; ++i)
    {
        printf("[");
        for (int j = 0; j < m->columns; ++j)
        {
            printf("%8.3f ", m->data[i][j]);
        }
        printf("]\n");
    }
    printf("\n");
}

void linalg_matrix_scale(Matrix *m, float scalar)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: matrix_scale NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m->rows; ++i)
    {
        for (int j = 0; j < m->columns; ++j)
        {
            m->data[i][j] *= scalar;
        }
    }
}

void linalg_matrix_add_scalar(Matrix *m, float scalar)
{

    if (!m || !m->data)
    {
        fprintf(stderr, "Error: add_scalar scalar NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m->rows; ++i)
    {
        for (int j = 0; j < m->columns; ++j)
        {
            m->data[i][j] += scalar;
        }
    }
}

// TODO: This is actually not that easy, I need GAUSS ELIM, implement if necessary
Matrix *linalg_matrix_inverse(const Matrix *m)
{

    if (m->rows != m->columns)
    {
        fprintf(stderr, "Error: matrix_inverse only square matrices can be inverted.\n");
        exit(EXIT_FAILURE);
    }
}

Matrix *linalg_matrix_transpose(const Matrix *m)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: matrix_transpose NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    Matrix *result = linalg_matrix_create(m->columns, m->rows);

    for (int i = 0; i < result->rows; ++i)
    {
        for (int j = 0; j < result->columns; ++j)
        {
            result->data[i][j] = m->data[j][i];
        }
    }

    return result;
}

Matrix *linalg_matrix_multiply(const Matrix *a, const Matrix *b)
{
    if (!a || !a->data)
    {
        fprintf(stderr, "Error: matrix_multiply a NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: matrix_multiply b NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    // mn x np => mp
    if (a->columns != b->rows)
    {
        fprintf(stderr, "Error: matrix_multiply incompatible matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    int m = a->rows;    // i
    int n = a->columns; // = b->rows, k
    int p = b->columns; // j

    Matrix *result = linalg_matrix_create(m, p);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            float sum = 0;
            for (int k = 0; k < n; ++k)
            {
                sum += a->data[i][k] * b->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }
    return result;
}

Matrix *linalg_matrix_add(const Matrix *a, const Matrix *b)
{

    if (!a || !a->data)
    {
        fprintf(stderr, "Error: matrix_add a NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: matrix_add b NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    if (a->columns != b->columns || a->rows != b->rows)
    {
        fprintf(stderr, "Error: matrix_add incompatible matrix dimensions");
        exit(EXIT_FAILURE);
    }

    Matrix *result = linalg_matrix_create(a->rows, a->columns);
    for (int i = 0; i < result->rows; ++i)
    {
        for (int j = 0; j < result->columns; ++j)
        {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }

    return result;
}

Matrix *linalg_matrix_sub(const Matrix *a, const Matrix *b)
{

    if (!a || !a->data)
    {
        fprintf(stderr, "Error: matrix_sub a NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: matrix_sub b NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    if (a->columns != b->columns || a->rows != b->rows)
    {
        fprintf(stderr, "Error: matrix_sub incompatible matrix dimensions");
        exit(EXIT_FAILURE);
    }

    Matrix *result = linalg_matrix_create(a->rows, a->columns);
    for (int i = 0; i < result->rows; ++i)
    {
        for (int j = 0; j < result->columns; ++j)
        {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }

    return result;
}

Matrix *linalg_matrix_copy(const Matrix *m)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: matrix_copy NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    Matrix *result = linalg_matrix_create(m->rows, m->columns);
    for (int i = 0; i < result->rows; ++i)
    {
        for (int j = 0; j < result->columns; ++j)
        {
            result->data[i][j] = m->data[i][j];
        }
    }
    return result;
}

Matrix *linalg_matrix_hadamard(const Matrix *a, const Matrix *b)
{

    if (!a || !a->data)
    {
        fprintf(stderr, "Error: matrix_hadamard a NULL matrix.\n");
        exit(EXIT_FAILURE);
    }
    if (!b || !b->data)
    {
        fprintf(stderr, "Error: matrix_hadamard b NULL matrix.\n");
        exit(EXIT_FAILURE);
    }
    if (a->rows != b->rows || a->columns != b->columns)
    {
        fprintf(stderr, "Error: matrix_hadamard incompatible matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    Matrix *result = linalg_matrix_create(a->rows, a->columns);
    for (int i = 0; i < result->rows; ++i)
    {
        for (int j = 0; j < result->columns; ++j)
        {
            result->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
    return result;
}

void linalg_matrix_apply(Matrix *m, float (*func)(float))
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: matrix_apply NULL matrix");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m->rows; ++i)
    {
        for (int j = 0; j < m->columns; ++j)
        {
            // Function pointers are automatically dereferenced in C
            m->data[i][j] = func(m->data[i][j]);
        }
    }
}

Vector *linalg_matrix_get_column(const Matrix *m, int column)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: get_column NULL matrix");
        exit(EXIT_FAILURE);
    }

    if (column < 0 || column >= m->columns)
    {
        fprintf(stderr, "Error: get_column column out of bounds");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(m->rows);
    for (int i = 0; i < result->size; ++i)
    {
        result->data[i] = m->data[i][column];
    }
    return result;
}

Vector *linalg_matrix_get_row(const Matrix *m, int row)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: get_row NULL matrix");
        exit(EXIT_FAILURE);
    }

    if (row < 0 || row >= m->rows)
    {
        fprintf(stderr, "Error: get_row row out of bounds");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(m->columns);
    for (int i = 0; i < result->size; ++i)
    {
        result->data[i] = m->data[row][i];
    }
    return result;
}