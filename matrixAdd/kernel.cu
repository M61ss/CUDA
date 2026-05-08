#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<int> data_;

public:
    Matrix(const int &n, const int &m, const int &val = 0) : rows_(n), cols_(m), data_(n * m) {
        if (val != 0) {
            fillMatrix(val);
        }
    }

    const size_t& rows() const { return rows_; }
    const size_t& cols() const { return cols_; }
    const std::vector<int>& data() const { return data_; }
    const int* rawData() const { return data_.data(); }

    size_t& rows() { return rows_; }
    size_t& cols() { return cols_; }
    std::vector<int>& data() { return data_; }
    int* rawData() { return data_.data(); }

    int& operator[](const int& pos) {
        return data_[pos];
    }

    void fillMatrix(const int val) {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                data_[i * cols_ + j] = val;
            }
        }
    }
};

cudaError_t matrixAdd(int *c, const int *a, const int *b, const size_t rows, const size_t cols);

__global__ void addKernel(int *c, const int *a, const int *b, const size_t cols)
{
    size_t idx = threadIdx.x * cols + threadIdx.y;
    c[idx] = a[idx] + b[idx];
}

int main()
{
    const size_t N = 5;
    const size_t M = 4;
    Matrix a(N, M, 1);
    Matrix b(N, M, 2);
    Matrix c(N, M);

    cudaError_t cudaStatus = matrixAdd(c.rawData(), a.rawData(), b.rawData(), N, M);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixAdd failed!");
        return 1;
    }

    for (int i = 0; i < c.rows(); i++) {
        for (int j = 0; j < c.cols(); j++) {
            printf("%d ", c[i * c.cols() + j]);
        }
        printf("\n");
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t matrixAdd(int *c, const int *a, const int *b, const size_t rows, const size_t cols)
{
    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, rows * cols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, rows * cols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, rows * cols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threadsPerBlock(rows, cols);
    addKernel<<<1, threadsPerBlock>>>(dev_c, dev_a, dev_b, cols);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
