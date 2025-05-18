#ifndef LINALG_H
#define LINALG_H

#define ZERO_EPS 1e-8f

typedef struct
{
    int size;
    float *data;
} Vector;

typedef struct
{
    int rows;
    int columns;
    float **data;
} Matrix;

// ---------------------
// VECTOR
// ---------------------

Vector *linalg_vector_create(int size);
Vector *linalg_vector_copy(const Vector *v);

void linalg_vector_free(Vector *v);
void linalg_vector_fill(Vector *v, float value);
void linalg_vector_print(const Vector *v);

void linalg_vector_scale(Vector *v, float scalar);
void linalg_vector_add_scalar(Vector *v, float scalar);
void linalg_vector_normalize(Vector *v);

float linalg_vector_dot(const Vector *a, const Vector *b);
float linalg_vector_sum(const Vector *v);

Vector *linalg_vector_add(const Vector *a, const Vector *b);
Vector *linalg_vector_sub(const Vector *a, const Vector *b);
Vector *linalg_vector_transform(const Matrix *m, const Vector *v);
Vector *linalg_vector_hadamard(const Vector *a, const Vector *b);

void linalg_vector_apply(Vector *v, float (*func)(float));

// ---------------------
// MATRIX
// ---------------------

Matrix *linalg_matrix_create(int rows, int columns);
Matrix *linalg_matrix_copy(const Matrix *m);
Matrix *linalg_matrix_identity(int n);

void linalg_matrix_free(Matrix *m);
void linalg_matrix_fill(Matrix *m, float value);
void linalg_matrix_print(const Matrix *m);

void linalg_matrix_scale(Matrix *m, float scale);
void linalg_matrix_add_scalar(Matrix *m, float scalar);

Matrix *linalg_matrix_inverse(const Matrix *m);
Matrix *linalg_matrix_transpose(const Matrix *m);
Matrix *linalg_matrix_multiply(const Matrix *a, const Matrix *b);
Matrix *linalg_matrix_add(const Matrix *a, const Matrix *b);
Matrix *linalg_matrix_sub(const Matrix *a, const Matrix *b);
Matrix *linalg_matrix_hadamard(const Matrix *a, const Matrix *b);

void linalg_matrix_apply(Matrix *m, float (*func)(float));

Vector *linalg_matrix_get_column(const Matrix *m, int column);
Vector *linalg_matrix_get_row(const Matrix *m, int row);

#endif