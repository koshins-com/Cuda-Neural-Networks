#ifndef NN_H
#define NN_H

__global__ void mul_acc(float* x, float* w, float* b, float* y, int row_size, int hid_size, int col_size);

__global__ void relu(float* y, float* x, int size);

__global__ void softmax(float* y, float* x, int num_rows, int num_cols);

__global__ void loss(float* x, bool* target, float* result, int num_rows, int num_cols);

#endif
