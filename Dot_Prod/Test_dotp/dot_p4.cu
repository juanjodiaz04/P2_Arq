#include <iostream>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

// Log Reduction using GPU

// Grid-Stride Loop + Iterative Reduction
__global__
void partial_prod_stride(int n, float *x, float *y, float *partial){
    int index  = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride){
        partial[i] = x[i] * y[i];
    }
}

// N/2 reduction kernel

__global__
void reduce_kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = n / 2;

    if(i < stride){
        data[i] += data[i + stride];
    }
}


int main() {
    int N = 1 << 25; // ~33M

    float *x = new float[N];
    float *y = new float[N];

    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_partial;
    int size = N * sizeof(float);

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_partial, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    int numBlocks = 32 * numSMs; // total blocks

    // Partial product with grid-stride loops
    auto t0 = std::chrono::high_resolution_clock::now();

    partial_prod_stride<<<numBlocks, blockSize>>>(N, d_x, d_y, d_partial);
    cudaDeviceSynchronize();

    // Iterative reduction on GPU (n -> n/2 -> n/4 -> ... -> 1)
    int n_remaining = N;

    while(n_remaining > 1){
        int threads = 256;
        int blocks = (n_remaining / 2 + threads - 1) / threads;

        reduce_kernel<<<blocks, threads>>>(d_partial, n_remaining);
        cudaDeviceSynchronize();

        n_remaining /= 2;
    }


    // Final Result copy to Host
    float result = 0.0f;
    cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    printf("Result = %f\n", result);
    printf("Expected = %f\n", 2.0 * N);
    printf("Execution time: %.3f ms\n", elapsed.count());

    delete[] x;
    delete[] y;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial);

    return 0;
}
