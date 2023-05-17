#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "hello.h"

// #define N 10000000 // 1meg
#define N 100000000  // 10meg
#define MAX_ERR 1e-6

struct container {
    int *hostData;
    int *deviceData;

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;
};

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

void *getContainer() {
    struct container *c = (struct container *)malloc(sizeof(struct container));
    c->hostData = NULL;
    c->deviceData = NULL;
    c->a = NULL;
    c->b = NULL;
    c->out = NULL;
    c->d_a = NULL;
    c->d_b = NULL;
    c->d_out = NULL;

    return (void *)c;
}

int sayHello(void *void_container) {
    struct container *c = (struct container *)void_container;

    // CUDA malloc host memory
    size_t memSize = N * sizeof(int);
    cudaError_t error = cudaMallocHost((void **)&c->hostData, memSize);
    if (error != cudaSuccess) {
        printf("cudaMallocHost returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // CUDA malloc device memory
    error = cudaMalloc((void **)&c->deviceData, memSize);
    if (error != cudaSuccess) {
        printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    c->a = (float *)malloc(sizeof(float) * N);
    c->b = (float *)malloc(sizeof(float) * N);
    c->out = (float *)malloc(sizeof(float) * N);

    // Initialize array
    for (int i = 0; i < N; i++) {
        c->a[i] = 1.0f;
        c->b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void **)&c->d_a, sizeof(float) * N);
    cudaMalloc((void **)&c->d_b, sizeof(float) * N);
    cudaMalloc((void **)&c->d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpyAsync(c->d_a, c->a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(c->d_b, c->b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<1, 1>>>(c->d_out, c->d_a, c->d_b, N);

    // Transfer data back to host memory
    cudaMemcpyAsync(c->out, c->d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; i++) {
        assert(fabs(c->out[i] - c->a[i] - c->b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", c->out[0]);
    printf("PASSED\n");

    printf("GPU Status:\n");
    printStatus();

    return 0;
}

int freeMem(void *void_container) {
    struct container *c = (struct container *)void_container;
    cudaDeviceSynchronize();

    // Deallocate device memory
    cudaFree(c->d_a);
    cudaFree(c->d_b);
    cudaFree(c->d_out);
    cudaFree(c->deviceData);

    // Deallocate host memory
    free(c->a);
    free(c->b);
    free(c->out);

    // CUDA freee host memory
    cudaFreeHost(c->hostData);

    // delete container
    free(c);

    printf("GPU Status:\n");
    printStatus();

    return 0;
}
