#include "nn.h"

__global__ void linear(float* x, float* w, float* b, float* y, int row_size, int hid_size, int col_size)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    float result = b[col];
    if (row < row_size && col < col_size)
    {
        for (int i = 0; i < hid_size; i++)
        {
            result +=  x[row * hid_size + i] * w[i * col_size + col];
        }
        y[row * col_size + col] = result;
    }
}

__global__ void relu(float* y, float* x, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        if (x[id] < 0)
        {
            y[id] = 0;
        }
        else
        {
            y[id] = x[id];
        }
    }
}

__global__ void softmax(float* y, float* x, int num_rows, int num_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_rows && col < num_cols)
    {
        x += row * num_cols;
        y += row * num_cols;
        float x_max = *x;
        for (int i = 0; i < num_cols; i++)
        {
            if (x[i] > x_max)
            {
                x_max = x[i];
            }
        }
        float sum = 0;
        for (int i = 0; i < num_cols; i++)
        {
            sum += exp(x[i] - x_max);
        }
        y[col] = exp(x[col] - x_max) / sum;
    }
}

__global__ void cross_entropy(float* x, bool* target, float* result, int batch_size, int num_cols)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < batch_size)
    {
        float loss = 0;
        for (int i = 0; i < num_cols; i++)
        {
            loss -= log(x[id * num_cols + i]) * target[id * num_cols + i];
        }
        result[id] = loss;
    }
}

//__global__ cross_entropy_back(float)
