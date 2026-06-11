#ifndef NN_H
#define NN_H

#include <stdbool.h>

__global__ void linear(float* x, float* w, float* b, float* y, int row_size, int hid_size, int col_size);

__global__ void relu(float* y, float* x, int size);

__global__ void softmax(float* y, float* x, int num_rows, int num_cols);

__global__ void cross_entropy(float* x, bool* target, float* result, int num_rows, int num_cols);

__global__ void cross_entropy_softmax_back(float* softmax_output, bool* t, float* dy_by_dx, size_t size);

#endif
