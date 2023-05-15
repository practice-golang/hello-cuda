#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>

// #define N 10000000 // 1meg
#define N 100000000 // 10meg
#define MAX_ERR 1e-6


int printStatus() {
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

    return 0;
}

__global__ void vector_add(float *out, float *a, float *b, int n) {
    printf("Hello World from GPU!\n");

    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

#ifdef __cplusplus
extern "C" {
#endif

float *a, *b, *out;
float *d_a, *d_b, *d_out;

#ifdef _WIN32
__declspec(dllexport)
#endif
int sayHello() {
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

#ifdef _WIN32
__declspec(dllexport)
#endif
int freeMem() {
    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);

    printf("GPU Status:\n");
    printStatus();

    return 0;
}

#ifdef __cplusplus
}
#endif
