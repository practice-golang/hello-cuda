#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "hello.h"

// #define N 10000000 // 1meg
#define N 100000000 // 10meg
#define MAX_ERR 1e-6

void printStatus() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("Device %d:\n", i);
        printf("  Name: %s\n", deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);

        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        printf("  Used Memory: %zu bytes\n", totalMem - freeMem);
        printf("  Free Memory: %zu bytes\n", freeMem);
        printf("  Total Memory: %zu bytes\n", totalMem);
        printf("\n");
    }
}

__global__ void vector_add(float *out, float *a, float *b, int n) {
    printf("Hello World from GPU!\n");

    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int *hostData;
int *deviceData;

float *a, *b, *out;
float *d_a, *d_b, *d_out;

int sayHello() {
    // CUDA malloc host memory
    // size_t memSize = N * sizeof(int);
    size_t memSize = 1024 * sizeof(int);
    cudaError_t error = cudaMallocHost((void**)&hostData, memSize);
    if (error != cudaSuccess) {
        printf("cudaMallocHost returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // CUDA malloc device memory
    error = cudaMalloc((void**)&deviceData, memSize);
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }


    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, sizeof(float) * N);
    cudaMalloc((void **)&d_b, sizeof(float) * N);
    cudaMalloc((void **)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<1, 1>>>(d_out, d_a, d_b, N);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    printf("GPU Status:\n");
    printStatus();

    return 0;
}

int freeMem() {
    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);

    // CUDA freee host memory
    cudaFreeHost(hostData);
    cudaFree(deviceData);

    printf("GPU Status:\n");
    printStatus();

    return 0;
}
