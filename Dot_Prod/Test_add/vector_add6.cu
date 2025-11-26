#include <iostream>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

// GPU vector add using grid-stride loop
__global__
void add(int n, float *x, float *y){
    int index  = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){
        y[i] = x[i] + y[i];
    }
}

int main(void){
    int N = 1 << 25; // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int size = N * sizeof(float);
    float *d_x, *d_y;

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Grid Config
    int blockSize = 256;        // threads per block
    int numSMs;                 // number of multiprocessors on the GPU

    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    // Time the operation
    auto t0 = std::chrono::high_resolution_clock::now();
    add<<<32 * numSMs, blockSize>>>(N, d_x, d_y);
    
    cudaDeviceSynchronize(); // wait for kernel to finish

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    std::printf("y[0] = %f\n", y[0]);
    std::printf("Execution time: %.3f ms\n", elapsed.count());

    delete [] x;
    delete [] y;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
