
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define M 5
#define N 4

cudaError_t matrixAdd(int ca[][M], const int a[][M], const int b[][M]);

__global__ void addKernel(int c[][M], const int a[][M], const int b[][M])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    c[i][j] = a[i][j] + b[i][j];
}

int main()
{
    const int a[N][M] = {
        { 1, 1, 1, 1, 1},
        { 1, 1, 1, 1, 1},
        { 1, 1, 1, 1, 1},
        { 1, 1, 1, 1, 1},
    };
    const int b[N][M] = {
        { 2, 2, 2, 2, 2},
        { 2, 2, 2, 2, 2},
        { 2, 2, 2, 2, 2},
        { 2, 2, 2, 2, 2},
    };
    int c[N][M] = { 0 };

    cudaError_t cudaStatus = matrixAdd(c, a, b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixAdd failed!");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%d ", c[i][j]);
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

cudaError_t matrixAdd(int c[][M], const int a[][M], const int b[][M])
{
    int(*dev_a)[M] = 0;
    int(*dev_b)[M] = 0;
    int(*dev_c)[M] = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, N * M * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * M * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * M * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, N * M * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, N * M * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threadsPerBlock(N, M);
    addKernel<<<1, threadsPerBlock>>>(dev_c, dev_a, dev_b);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);
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
