#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

#define MAX_POINTS 10000000

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ---------------- DUMP FUNCTION (same as complex version) ----------------
void dump_iteration_data(
    int it,
    int K,
    int N,
    const float *d_centroids, 
    const int   *d_labels)
{
    
    static float *centroids_dump = NULL; 
    static int   *labels_dump    = NULL;

    if (centroids_dump == NULL) {
        
        centroids_dump = (float*)malloc(2 * K * sizeof(float));
        labels_dump = (int*)malloc(N * sizeof(int));
    }


    CUDA_CHECK(cudaMemcpy(centroids_dump, d_centroids, 2 * K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels_dump, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost));

    char fname[256];
    sprintf(fname, "iterations_sim/iteration_%03d.csv", it);

    FILE *f = fopen(fname, "w");
    if (!f) return;

    fprintf(f, "# centroids:\n");
    for (int c = 0; c < K; c++)
        // Indexación para AoS: x está en 2*c, y está en 2*c + 1
        fprintf(f, "C%d,%f,%f\n", c, centroids_dump[2*c], centroids_dump[2*c + 1]);

    fprintf(f, "# labels:\n");
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", labels_dump[i]);

    fclose(f);
}

// ---------------- Load CSV ----------------
int load_csv(const char *filename, float **out_points) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;

    float *points = (float*)malloc(MAX_POINTS * 2 * sizeof(float));
    if (!points) return -1;

    int count = 0;
    float x, y;
    while (fscanf(f, "%f,%f", &x, &y) == 2) {
        points[2 * count]     = x;
        points[2 * count + 1] = y;
        if (++count >= MAX_POINTS) break;
    }

    fclose(f);
    *out_points = points;
    return count;
}

// ---------------- Kernel: assign + accumulate ----------------
__global__
void assign_and_accumulate_kernel(
    const float *points,    
    const float *centroids, 
    int   *labels,          
    float *sum,             
    int   *count,           
    int N,
    int K)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {

        float px = points[2 * i];
        float py = points[2 * i + 1];

        float best_dist = FLT_MAX;
        int   best_c    = -1;

        for (int c = 0; c < K; ++c) {
            float cx = centroids[2 * c];
            float cy = centroids[2 * c + 1];

            float dx = px - cx;
            float dy = py - cy;
            float d2 = dx * dx + dy * dy;

            if (d2 < best_dist) {
                best_dist = d2;
                best_c    = c;
            }
        }

        labels[i] = best_c;

        atomicAdd(&sum[2 * best_c],     px);
        atomicAdd(&sum[2 * best_c + 1], py);
        atomicAdd(&count[best_c],       1);
    }
}

// ---------------- GPU K-means ----------------
void kmeans_gpu(
    float *h_points,
    float *h_centroids,
    int N,
    int K,
    int max_iters,
    float epsilon)
{
    system("rm -rf iterations_sim");
    system("mkdir -p iterations_sim");

    float *d_points, *d_centroids;
    int   *d_labels;
    float *d_sum;
    int   *d_count;

    CUDA_CHECK(cudaMalloc(&d_points,    2 * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, 2 * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels,    N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum,       2 * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count,     K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points,    h_points,    2 * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, 2 * K * sizeof(float), cudaMemcpyHostToDevice));

    float *h_sum   = (float*)malloc(2 * K * sizeof(float));
    int   *h_count = (int*)malloc(K * sizeof(int));

    int blockSize = 512;
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    int numBlocks = 32 * numSMs;

    printf("K-means GPU (1 kernel): N=%d, K=%d\n", N, K);

    // Kernel execution timing
    auto start_kernel = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iters; ++it) {

        CUDA_CHECK(cudaMemset(d_sum,   0, 2 * K * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_count, 0,     K * sizeof(int)));

        assign_and_accumulate_kernel<<<numBlocks, blockSize>>>(d_points, d_centroids, d_labels, d_sum, d_count, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_sum,   d_sum,   2 * K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_count, d_count,     K * sizeof(int),   cudaMemcpyDeviceToHost));

        float movement = 0.f;
        for (int c = 0; c < K; ++c) {

            float oldx = h_centroids[2 * c];
            float oldy = h_centroids[2 * c + 1];

            float newx = oldx, newy = oldy;

            if (h_count[c] > 0) {
                newx = h_sum[2 * c]     / h_count[c];
                newy = h_sum[2 * c + 1] / h_count[c];
            }

            float dx = newx - oldx;
            float dy = newy - oldy;
            movement += dx*dx + dy*dy;

            h_centroids[2 * c]     = newx;
            h_centroids[2 * c + 1] = newy;
        }

        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, 2 * K * sizeof(float), cudaMemcpyHostToDevice));

        //DUMP CURRENT STATE TO FILE
        //dump_iteration_data(it, K, N, d_centroids, d_labels);

        if (movement < epsilon) break;
    }

    // Kernel end timing
    auto end_kernel = std::chrono::high_resolution_clock::now();
    double elapsed_kernel = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    printf("Elapsed time kernel: %.3f ms\n", elapsed_kernel);

    free(h_sum);
    free(h_count);

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_count));
}

// ---------------- Main ----------------
int main(int argc, char **argv) {
    if (argc < 2) return 1;

    float *points = NULL;
    int N = load_csv(argv[1], &points);
    if (N <= 0) return 1;

    int K = 3;
    float centroids[6] = {0.0f, 0.0f, 5.0f, 5.0f, 10.0f, 10.0f};

    // Measure execution time in GPU
    auto start = std::chrono::high_resolution_clock::now();

    kmeans_gpu(points, centroids, N, K, 50, 1e-4);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Elapsed time: %.3f ms\n", elapsed);

    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; ++c)
        printf("C%d = (%f, %f)\n", c, centroids[2*c], centroids[2*c+1]);

    free(points);
    return 0;
}
