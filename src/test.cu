#include "nn.h"
#include "cucall.h"
#include "test.h"
#include <cstddef>
#include <limits.h>
#include <curand.h>
#include <stdio.h>
#include <float.h>

#define BLOCK_SIZE1D 1024
#define BLOCK_SIZE2D 32
#define MIN(x, y) (y < x ? y : x)
#define MAX(x, y) (y > x ? y : x)

__global__ void to_float(uint* in, float* out, size_t count)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < count)
    {
        out[id] = ((float) in[id] / UINT_MAX - .5) * 2;
    }
}

__global__ void to_bool(uint* in, bool* out, size_t count)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < count)
    {
        out[id] = in[id] > UINT_MAX / 2;
    }
}

bool check(float* a, float* b, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        if (abs(a[i] - b[i]) > 1E-1)
        {
            return false;
        }
    }
    return true;
}

bool check(float* y, float* t, size_t size, bool print)
{
    if (!print)
    {
        return check(y, t, size);
    }
    bool r = true;
    printf("\t\t");
    for (int i = 0; i < size; i++)
    {
        if (abs(y[i] - t[i]) > 1E-1)
        {
            printf("(\x1b[32m%f\x1b[0m, \x1b[31m%f\x1b[0m) ", t[i], y[i]);
            r = false;
        }
        else
        {
            printf("%f ", y[i]);
        }
    }
    printf("\n");
    return r;
}

bool check(float* y, float* t, size_t rows, size_t cols, bool print)
{
    if (!print)
    {
        return check(y, t, rows * cols);
    }
    bool r = true;
    for (int i = 0; i < rows; i++)
    {
        printf("\t\t");
        for (int j = 0; j < cols; j++)
        {
            int idx = i * cols + j;
            if (abs(y[idx] - t[idx]) > 1E-1)
            {
                printf("(\x1b[32m%f\x1b[0m, \x1b[31m%f\x1b[0m) ", t[idx], y[idx]);
                r = false;
            }
            else
            {
                printf("%f ", y[idx]);
            }
        }
        printf("\n");
    }
    return r;
}

void print_array(float* a, size_t rows, size_t cols)
{
    for (int i = 0; i < rows; i++)
    {
        printf("\t\t");
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", a[i * cols + j]);
        }
        printf("\n");
    }
}

void print_array(bool* a, size_t rows, size_t cols)
{
    for (int i = 0; i < rows; i++)
    {
        printf("\t\t");
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", a[i * cols + j]);
        }
        printf("\n");
    }
}

void print_array(float* a, size_t size)
{
    printf("\t\t");
    for (int i = 0; i < size; i++)
    {
        printf("%f ", a[i]);
    }
    printf("\n");
}

void print_array(bool* a, size_t size)
{
    printf("\t\t");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
}

bool test_linear(curandGenerator_t generator)
{
    bool print = false;
    constexpr uint rows = 1 << 6;
    constexpr uint cols = 1 << 8;
    constexpr uint hids = 1 << 7;
    uint* temp;
    CUDA_CALL(cudaMalloc(&temp, sizeof(uint) * rows * cols * hids / MIN(rows, MIN(cols, hids))));

    float* x;
    constexpr uint countx = rows * hids;
    CUDA_CALL(cudaMalloc(&x, countx * sizeof(float)));
    CURAND_CALL(curandGenerate(generator, temp, countx));
    to_float<<<(countx + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(temp, x, countx);

    float* w;
    constexpr uint countw = hids * cols;
    CUDA_CALL(cudaMalloc(&w, countw * sizeof(float)));
    CURAND_CALL(curandGenerate(generator, temp, countw));
    to_float<<<(countw + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(temp, w, countw);

    float* b;
    constexpr uint countb = hids * cols;
    CUDA_CALL(cudaMalloc(&b, countb * sizeof(float)));
    CURAND_CALL(curandGenerate(generator, temp, countb));
    to_float<<<(countb + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(temp, b, countb);
    CUDA_CALL(cudaFree(temp));

    float* y;
    constexpr uint county = rows * cols;
    CUDA_CALL(cudaMalloc(&y, county * sizeof(float)));

    constexpr dim3 bs(BLOCK_SIZE2D, BLOCK_SIZE2D);
    constexpr dim3 gs((rows + BLOCK_SIZE2D -1) / BLOCK_SIZE2D, (cols + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D);
    linear<<<gs, bs>>>(x, w, b, y, rows, hids, cols);

    float* dy = (float*) malloc(sizeof(float) * county);
    CUDA_CALL(cudaMemcpy(dy, y, sizeof(float) * county, cudaMemcpyDeviceToHost));
    float* dx = (float*) malloc(sizeof(float) * countx);
    CUDA_CALL(cudaMemcpy(dx, x, sizeof(float) * countx, cudaMemcpyDeviceToHost));
    float* dw = (float*) malloc(sizeof(float) * countw);
    CUDA_CALL(cudaMemcpy(dw, w, sizeof(float) * countw, cudaMemcpyDeviceToHost));
    float* db = (float*) malloc(sizeof(float) * countb);
    CUDA_CALL(cudaMemcpy(db, b, sizeof(float) * countb, cudaMemcpyDeviceToHost));
    float* dt = (float*) malloc(sizeof(float) * county);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float sum = db[j];
            for (int k = 0; k < hids; k++)
            {
                sum += dx[i * hids + k] * dw[k * cols + j];
            }
            dt[i * cols + j] = sum;
        }
    }

    if (print)
    {
        printf("\tx:\n");
        print_array(dx, rows, hids);
        printf("\tw:\n");
        print_array(dw, hids, cols);
        printf("\tb:\n");
        print_array(db, 1, cols);
        printf("\tExpected:\n");
        print_array(dt, rows, cols);
        printf("\tCalculated:\n");
        print_array(dy, rows, cols);
    }
    return check(dt, dy, county);
}

bool test_relu(curandGenerator_t generator)
{
    bool print = false;
    constexpr int size = 1 << 11;
    uint* temp;
    CUDA_CALL(cudaMalloc(&temp, sizeof(uint) * size));
    CURAND_CALL(curandGenerate(generator, temp, size));
    float* x;
    CUDA_CALL(cudaMalloc(&x, sizeof(float) * size));
    to_float<<<(size + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(temp, x, size);
    float* y;
    CUDA_CALL(cudaMalloc(&y, sizeof(float) * size));
    relu<<<(size + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(y, x, size);
    float* dx = (float*) malloc(sizeof(float) * size);
    CUDA_CALL(cudaMemcpy(dx, x, sizeof(float) * size, cudaMemcpyDeviceToHost));
    float* dy = (float*) malloc(sizeof(float) * size);
    CUDA_CALL(cudaMemcpy(dy, y, sizeof(float) * size, cudaMemcpyDeviceToHost));
    float* dt = (float*) malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++)
    {
        if (dx[i] < 0)
        {
            dt[i] = 0;
        }
        else
        {
            dt[i] = dx[i];
        }
    }
    if (print)
    {
        printf("\tx:\t\t");
        print_array(dx, size);
        printf("\tExpected:\t");
        print_array(dt, size);
        printf("\tCalculated:\t");
        print_array(dy, size);
    }
    return check(dt, dy, size);
}


bool test_softmax(curandGenerator_t generator)
{
    bool print = false;
    constexpr size_t rows = 1 << 6;
    constexpr size_t cols = 1 << 7;
    constexpr size_t size = rows * cols;
    uint* temp;
    CUDA_CALL(cudaMalloc(&temp, size * sizeof(uint)));
    CURAND_CALL(curandGenerate(generator, temp, size));
    float* x;
    CUDA_CALL(cudaMalloc(&x, sizeof(float) * size));
    to_float<<<(size + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(temp, x, size);
    float* y;
    CUDA_CALL(cudaMalloc(&y, sizeof(float) * size));
    dim3 bs(BLOCK_SIZE2D, BLOCK_SIZE2D);
    dim3 gs((rows + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D, (cols + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D);
    softmax<<<gs, bs>>>(y, x, rows, cols);
    float* dx = (float*) malloc(sizeof(float) * size);
    CUDA_CALL(cudaMemcpy(dx, x, sizeof(float) * size, cudaMemcpyDeviceToHost));
    float* dy = (float*) malloc(sizeof(float) * size);
    CUDA_CALL(cudaMemcpy(dy, y, sizeof(float) * size, cudaMemcpyDeviceToHost));
    float* dt = (float*) malloc(sizeof(float) * size);
    for (int i = 0; i < rows; i++)
    {
        float max = FLT_MIN;
        for (int j = 0; j < cols; j++)
        {
            if (max < dx[i * cols + j])
            {
                max = dx[i * cols + j];
            }
        }
        float sum = 0;
        for (int j = 0; j < cols; j++)
        {
            sum += exp(dx[i * cols + j] - max);
        }
        for (int j = 0; j < cols; j++)
        {
            dt[i * cols + j] = exp(dx[i * cols + j] - max) / sum;
        }
    }
    if (print)
    {
        printf("\tx:\n");
        print_array(dx, rows, cols);
        printf("\tExpected:\n");
        print_array(dt, rows, cols);
        printf("\tCalculated:\n");
        print_array(dy, rows, cols);
    }
    return check(dt, dy, size);
}

bool test_cross_entropy(curandGenerator_t generator)
{
    bool print = false;
    constexpr size_t rows = 1 << 10;
    constexpr size_t cols = 1 << 9;
    constexpr size_t size = rows * cols;
    uint* temp;
    CUDA_CALL(cudaMalloc(&temp, sizeof(uint) * size));
    CURAND_CALL(curandGenerate(generator, temp, size));

    float* y;
    CUDA_CALL(cudaMalloc(&y, sizeof(float) * size));
    to_float<<<(size + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(temp, y, size);
    dim3 gs((rows + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D, (cols + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D);
    dim3 bs(BLOCK_SIZE2D, BLOCK_SIZE2D);
    softmax<<<gs, bs>>>(y, y, rows, cols);

    bool* t;
    CUDA_CALL(cudaMalloc(&t, size * sizeof(bool)));
    CURAND_CALL(curandGenerate(generator, temp, size));
    to_bool<<<(size + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(temp, t, size);
    float* l;
    CUDA_CALL(cudaMalloc(&l, rows * sizeof(float)));

    cudaDeviceSynchronize();
    cross_entropy<<<(rows + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D, BLOCK_SIZE1D>>>(y, t, l, rows, cols);
    cudaDeviceSynchronize();

    float* dy = (float*) malloc(sizeof(float) * size);
    CUDA_CALL(cudaMemcpy(dy, y, sizeof(float) * size, cudaMemcpyDeviceToHost));
    bool* dt = (bool*) malloc(sizeof(bool) * size);
    CUDA_CALL(cudaMemcpy(dt, t, sizeof(bool) * size, cudaMemcpyDeviceToHost));
    float* dl = (float*) malloc(sizeof(float) * rows);
    CUDA_CALL(cudaMemcpy(dl, l, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    float* dtl = (float*) malloc(sizeof(float) * rows);

    for (int i = 0; i < rows; i++)
    {
        dtl[i] = 0;
        for (int j = 0; j < cols; j++)
        {
            dtl[i] -= log(dy[i * cols + j]) * dt[i * cols + j];
        }
    }

    if (print)
    {
        printf("\ty:\n");
        print_array(dy, rows, cols);
        printf("\tt:\n");
        print_array(dt, rows, cols);
    }
    return check(dl, dtl, rows, print);
}

bool test_cross_entropy_softmax_back(curandGenerator_t generator)
{
    bool print = true;
    size_t rows = 2;
    size_t cols = 3;
    size_t size = rows * cols;
    uint* temp;
    CUDA_CALL(cudaMalloc(&temp, sizeof(uint) * size));
    CURAND_CALL(curandGenerate(generator, temp, size));
    float* softmax_in;
    CUDA_CALL(cudaMalloc(&softmax_in, sizeof(float) * size));
    dim3 gs((size + BLOCK_SIZE1D - 1) / BLOCK_SIZE1D);
    dim3 bs(BLOCK_SIZE1D);
    to_float<<<(size + BLOCK_SIZE1D - 1), BLOCK_SIZE1D>>>(temp, softmax_in, size);
    float* softmax_output;
    CUDA_CALL(cudaMalloc(&softmax_output, sizeof(float) * size));
    dim3 gs2d((rows + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D, (cols + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D);
    dim3 bs2d(BLOCK_SIZE2D, BLOCK_SIZE2D);
    softmax<<<gs2d, bs2d>>>(softmax_output, softmax_in, rows, cols);
    bool* t;
    CUDA_CALL(cudaMalloc(&t, sizeof(float) * size));
    CURAND_CALL(curandGenerate(generator, temp, size));
    to_bool<<<gs, bs>>>(temp, t, size);
    float* dy_by_dx;
    CUDA_CALL(cudaMalloc(&dy_by_dx, sizeof(float) * size));
    cross_entropy_softmax_back<<<gs, bs>>>(softmax_output, t, dy_by_dx, size);
    float* dso = (float*) malloc(sizeof(float) * size);
    CUDA_CALL(cudaMemcpy(dso, softmax_output, sizeof(float) * size, cudaMemcpyDeviceToHost));
    bool* dt = (bool*) malloc(sizeof(bool) * size);
    CUDA_CALL(cudaMemcpy(dt, t, sizeof(bool) * size, cudaMemcpyDeviceToHost));
    float* dd = (float*) malloc(sizeof(float) * size);
    CUDA_CALL(cudaMemcpy(dd, dy_by_dx, sizeof(float) * size, cudaMemcpyDeviceToHost));
    float* dtd = (float*) malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++)
    {
        dtd[i] = dso[i] - dt[i];
    }
    if (print)
    {
        printf("\tsoft out:\n");
        print_array(dso, rows, cols);
        printf("\tt:\n");
        print_array(dso, rows, cols);
        printf("\tOutput:\n");
    }
    return check(dd, dtd, rows, cols, print);
}
