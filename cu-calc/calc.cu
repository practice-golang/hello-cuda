#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

extern "C" {
    __declspec(dllexport)  int sayHello() {
        cuda_hello<<<1,1>>>(); 
        return 0;
    }
}
