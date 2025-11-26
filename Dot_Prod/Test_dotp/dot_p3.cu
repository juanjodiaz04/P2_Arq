#include <iostream>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

// Global accumulator version (atomicAdd)
__global__
void dot_atomic(int n, float *x, float *y, float *result){
    int idx  = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    for(int i = idx; i < n; i += stride){
        sum += x[i] * y[i];
    }

    atomicAdd(result, sum);
}

int main() {
    int N = 1 << 25; // ~33M elements

    float *x = new float[N];
    float *y = new float[N];

    for(int i=0;i<N;i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_result;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Inicializar resultado en GPU
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256; // threads per block
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    int numBlocks = 32 * numSMs; // total blocks

    auto t0 = std::chrono::high_resolution_clock::now();

    dot_atomic<<<numBlocks, blockSize>>>(N, d_x, d_y, d_result);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    printf("Result = %f\n", result);
    printf("Expected = %f\n", 2.0 * N);
    printf("Execution time: %.3f ms\n", elapsed.count());

    delete[] x;
    delete[] y;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    return 0;
}
