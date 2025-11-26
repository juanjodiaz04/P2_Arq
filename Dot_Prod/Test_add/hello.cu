#include <stdio.h>
__global__ void kernelHello() {
    printf("Hi, I’m the GPU!\n");
}

int main() {
    kernelHello<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Hi, I’m the CPU!\n");
    return 0;
}