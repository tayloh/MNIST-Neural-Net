#include "../include/linalg.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

Vector *linalg_vector_create(int size)
{
    Vector *v = (Vector *)malloc(sizeof(Vector));
    if (!v)
    {
        perror("malloc Vector");
        exit(EXIT_FAILURE);
    }
    v->size = size;
    v->data = (float *)calloc(size, sizeof(float));
    if (!v->data)
    {
        perror("calloc Vector data");
        free(v);
        exit(EXIT_FAILURE);
    }

    return v;
}

Vector *linalg_vector_copy(const Vector *v)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_copy NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(v->size);
    for (int i = 0; i < result->size; i++)
    {
        result->data[i] = v->data[i];
    }
    return result;
}

void linalg_vector_free(Vector *v)
{
    if (v)
    {
        free(v->data);
        free(v);
    }
}

void linalg_vector_fill(Vector *v, float value)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_fill NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < v->size; ++i)
    {
        v->data[i] = value;
    }
}

void linalg_vector_scale(Vector *v, float scalar)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_scale NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < v->size; ++i)
    {
        v->data[i] *= scalar;
    }
}

void linalg_vector_normalize(Vector *v)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_normalize NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    float length = sqrtf(linalg_vector_dot(v, v));
    if (length > ZERO_EPS)
    {
        linalg_vector_scale(v, 1.0f / length);
    }
    else
    {
        fprintf(stderr, "Error: vector_normalize division by zero\n");
        exit(EXIT_FAILURE);
    }
}

float linalg_vector_dot(const Vector *a, const Vector *b)
{
    if (!a || !b->data)
    {
        fprintf(stderr, "Error: vector_dot a NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: vector_dot b NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    if (a->size != b->size)
    {
        fprintf(stderr, "Error: vector_dot vector sizes incompatible\n");
        exit(EXIT_FAILURE);
    }
    float sum = 0.0f;
    for (int i = 0; i < a->size; ++i)
    {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

void linalg_vector_print(const Vector *v)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_print NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    printf("Vector_%d [ ", v->size);
    for (int i = 0; i < v->size; ++i)
    {
        printf("%8.3f ", v->data[i]);
    }
    printf("]\n\n");
}

Vector *linalg_vector_transform(const Matrix *m, const Vector *v)
{
    if (!m || !m->data)
    {
        fprintf(stderr, "Error: vector_transform NULL matrix.\n");
        exit(EXIT_FAILURE);
    }

    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_transform NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    if (m->columns != v->size)
    {
        fprintf(stderr, "Error: vector_transform incompatible vector and matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(m->rows);
    for (int i = 0; i < m->rows; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < m->columns; ++j)
        {
            sum += m->data[i][j] * v->data[j];
        }
        result->data[i] = sum;
    }
    return result;
}

float linalg_vector_sum(const Vector *v)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_sum NULL vector");
        exit(EXIT_FAILURE);
    }

    float sum = 0;
    for (int i = 0; i < v->size; ++i)
    {
        sum += v->data[i];
    }
    return sum;
}

Vector *linalg_vector_add(const Vector *a, const Vector *b)
{
    if (!a || !a->data)
    {
        fprintf(stderr, "Error: vector_add a NULL vector");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: vector_add b NULL vector");
        exit(EXIT_FAILURE);
    }

    if (a->size != b->size)
    {
        fprintf(stderr, "Error: vector_add incompatible vector sizes");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(a->size);
    for (int i = 0; i < result->size; ++i)
    {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

Vector *linalg_vector_sub(const Vector *a, const Vector *b)
{

    if (!a || !a->data)
    {
        fprintf(stderr, "Error: vector_sub a NULL vector");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: vector_sub b NULL vector");
        exit(EXIT_FAILURE);
    }

    if (a->size != b->size)
    {
        fprintf(stderr, "Error: vector_sub incompatible vector sizes");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(a->size);
    for (int i = 0; i < result->size; ++i)
    {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

Vector *linalg_vector_hadamard(const Vector *a, const Vector *b)
{
    if (!a || !a->data)
    {
        fprintf(stderr, "Error: vector_hadamard a NULL vector");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: vector_hadamard b NULL vector");
        exit(EXIT_FAILURE);
    }

    if (a->size != b->size)
    {
        fprintf(stderr, "Error: vector_hadamard incompatible vector sizes");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(a->size);
    for (int i = 0; i < result->size; ++i)
    {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

void linalg_vector_add_scalar(Vector *v, float scalar)
{
    if (scalar < ZERO_EPS)
    {
        fprintf(stderr, "Error: vector_add_scalar division by 0.\n");
        exit(EXIT_FAILURE);
    }

    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_add_scalar NULL vector.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < v->size; ++i)
    {
        v->data[i] += scalar;
    }
}

void linalg_vector_apply(Vector *v, float (*func)(float))
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_apply NULL vector");
        exit(EXIT_FAILURE);
    }

    if (!func)
    {
        fprintf(stderr, "Error: vector_apply NULL func");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < v->size; ++i)
    {
        v->data[i] = func(v->data[i]);
    }
}

// Copy values of b into a
void linalg_vector_copy_into(Vector *a, const Vector *b)
{
    if (!a || !a->data)
    {
        fprintf(stderr, "Error: vector_copy_into a NULL vector");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: vector_copy_into b NULL vector");
        exit(EXIT_FAILURE);
    }

    if (a->size != b->size)
    {
        fprintf(stderr, "Error: vector_copy_into size mismatch");
        exit(EXIT_FAILURE);
    }

    memcpy(a->data, b->data, b->size * sizeof(float));
}

// Add values of b onto values of a
void linalg_vector_add_into(Vector *a, const Vector *b)
{
    if (!a || !a->data)
    {
        fprintf(stderr, "Error: vector_copy_into a NULL vector");
        exit(EXIT_FAILURE);
    }

    if (!b || !b->data)
    {
        fprintf(stderr, "Error: vector_copy_into b NULL vector");
        exit(EXIT_FAILURE);
    }

    if (a->size != b->size)
    {
        fprintf(stderr, "Error: vector_copy_into size mismatch");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < a->size; ++i)
    {
        a->data[i] += b->data[i];
    }
}

int linalg_vector_argmax(const Vector *v)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_argmax NULL v");
        exit(EXIT_FAILURE);
    }

    int argmax = 0;
    float max = v->data[0];

    for (int i = 1; i < v->size; ++i)
    {
        if (v->data[i] > max)
        {
            max = v->data[i];
            argmax = i;
        }
    }

    return argmax;
}

float linalg_vector_max(const Vector *v)
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_max NULL v");
        exit(EXIT_FAILURE);
    }

    float max = v->data[0];
    for (int i = 0; i < v->size; ++i)
    {
        if (v->data[i] > max)
        {
            max = v->data[i];
        }
    }

    return max;
}

Matrix *linalg_vector_outer_prod(const Vector *a, const Vector *b)
{
    if (!a || !a->data)
    {
        fprintf(stderr, "Error: vector_outer_prod a NULL vector");
        exit(EXIT_FAILURE);
    }
    if (!b || !b->data)
    {
        fprintf(stderr, "Error: vector_outer_prod b NULL vector");
        exit(EXIT_FAILURE);
    }

    Matrix *result = linalg_matrix_create(a->size, b->size);
    for (int i = 0; i < result->rows; ++i)
    {
        for (int j = 0; j < result->columns; ++j)
        {
            result->data[i][j] = a->data[i] * b->data[j];
        }
    }

    return result;
}

Vector *linalg_vector_map(const Vector *v, float (*func)(float))
{
    if (!v || !v->data)
    {
        fprintf(stderr, "Error: vector_map NULL vector");
        exit(EXIT_FAILURE);
    }

    if (!func)
    {
        fprintf(stderr, "Error: vector_map NULL func");
        exit(EXIT_FAILURE);
    }

    Vector *result = linalg_vector_create(v->size);
    for (int i = 0; i < v->size; ++i)
    {
        result->data[i] = func(v->data[i]);
    }

    return result;
}