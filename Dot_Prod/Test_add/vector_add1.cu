#include <iostream>
#include <chrono>
#include <cstdio>

// Simple CPU vector add
void add(int N, float *x, float *y){
    for (int i = 0; i < N; i++)
        y[i] = x[i] + y[i];
}

int main(void){
    int N = 1<<25; // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y on the CPU
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Time the add() call using high-resolution clock
    auto t0 = std::chrono::high_resolution_clock::now();
    add(N, x, y);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    // Print a sample result and the elapsed time
    std::printf("y[0] = %f\n", y[0]);
    std::printf("Execution time: %.3f ms\n", elapsed.count());

    // Free memory
    delete [] x;
    delete [] y;

    std::printf("Completed successfully\n");

    return 0;
}