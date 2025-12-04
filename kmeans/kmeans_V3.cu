#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

#define MAX_POINTS 10000000

//------------ Macro for CUDA errors-------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

//-------------------- Read CSV points in two arrays ---------------------
int load_csv_soa(const char *filename, float **out_x, float **out_y) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Error: cannot open file %s\n", filename);
        return -1;
    }

    float *x = (float *)malloc(MAX_POINTS * sizeof(float));
    float *y = (float *)malloc(MAX_POINTS * sizeof(float));

    if (!x || !y) {
        printf("Error allocating memory\n");
        fclose(f);
        free(x);
        free(y);
        return -1;
    }

    int count = 0;
    float px, py;
    while (fscanf(f, "%f,%f", &px, &py) == 2) {
        if (count >= MAX_POINTS) break;
        x[count] = px;
        y[count] = py;
        count++;
    }

    fclose(f);
    *out_x = x;
    *out_y = y;
    return count;
}

//------------------- Dump iteration data ------------------------
void dump_iteration_data(
    int it,
    int K,
    int N,
    const float *d_cx,
    const float *d_cy,
    const int   *d_labels)
{
    static float *cx_dump     = NULL;
    static float *cy_dump     = NULL;
    static int   *labels_dump = NULL;

    if (cx_dump == NULL) {
        cx_dump     = (float*)malloc(K * sizeof(float));
        cy_dump     = (float*)malloc(K * sizeof(float));
        labels_dump = (int*)malloc(N * sizeof(int));
    }

    CUDA_CHECK(cudaMemcpy(cx_dump,     d_cx,    K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cy_dump,     d_cy,    K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels_dump, d_labels, N * sizeof(int),  cudaMemcpyDeviceToHost));

    char fname[256];
    sprintf(fname, "iterations_opt2/iteration_%03d.csv", it);

    FILE *f = fopen(fname, "w");
    if (!f) return;

    fprintf(f, "# centroids:\n");
    for (int c = 0; c < K; c++)
        fprintf(f, "C%d,%f,%f\n", c, cx_dump[c], cy_dump[c]);

    fprintf(f, "# labels:\n");
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", labels_dump[i]);

    fclose(f);
}

// =====================================================================
//   Fused Kernels
// =====================================================================

__global__
void assign_reduce_kernel(
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ cx,
    const float *__restrict__ cy,
    int *__restrict__ labels,
    float *sum_x,
    float *sum_y,
    int   *count,
    int N,
    int K)
{
    extern __shared__ float smem[];

    // shared memory layout:
    // 0..K-1   : cx_s
    // K..2K-1  : cy_s
    // 2K..3K-1 : sum_x_b
    // 3K..4K-1 : sum_y_b
    // 4K..5K-1 : count_b (int)
    float *cx_s = smem;
    float *cy_s = smem + K;
    float *sum_x_b = smem + 2*K;
    float *sum_y_b = smem + 3*K;
    int   *count_b = (int*)(smem + 4*K);

   // Copy centroids to shared memory and initialize accumulators
    for (int c = threadIdx.x; c < K; c += blockDim.x)
    {
        // Copy centroids to shared memory
        cx_s[c] = cx[c];
        cy_s[c] = cy[c];

        // Initialize accumulators
        sum_x_b[c] = 0.0f;
        sum_y_b[c] = 0.0f;
        count_b[c] = 0;
    }

    __syncthreads();

    // Assign clusters and accumulate in shared memory
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        float px = x[i];
        float py = y[i];

        float best_dist = FLT_MAX;
        int best_c = -1;

        for (int c = 0; c < K; ++c) {
            float dx = px - cx_s[c];
            float dy = py - cy_s[c];
            float d2 = dx*dx + dy*dy;

            if (d2 < best_dist) {
                best_dist = d2;
                best_c = c;
            }
        }

        labels[i] = best_c;

        // Accumulate to shared memory
        atomicAdd(&sum_x_b[best_c], px);
        atomicAdd(&sum_y_b[best_c], py);
        atomicAdd(&count_b[best_c], 1);
    }

    __syncthreads();

    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        atomicAdd(&sum_x[c], sum_x_b[c]);
        atomicAdd(&sum_y[c], sum_y_b[c]);
        atomicAdd(&count[c], count_b[c]);
    }
}


// =====================================================================
//    UPDATE CENTROIDS KERNEL
// =====================================================================
__global__
void update_centroids_kernel(
    float *cx,
    float *cy,
    const float *sum_x,
    const float *sum_y,
    const int   *count,
    float *movement,
    int K)
{
    int c = threadIdx.x;
    if (c < K) {
        if (count[c] > 0) {
            float oldx = cx[c];
            float oldy = cy[c];

            float newx = sum_x[c] / count[c];
            float newy = sum_y[c] / count[c];

            float dx = newx - oldx;
            float dy = newy - oldy;

            cx[c] = newx;
            cy[c] = newy;

            atomicAdd(movement, dx*dx + dy*dy);
        }
    }
}

// =====================================================================
//                    KMEANS GPU
// =====================================================================
void kmeans_gpu(
    float *x_host,
    float *y_host,
    float *centroids_host,
    int N,
    int K,
    int max_iters,
    float epsilon)
{
    float *cx_host = (float*)malloc(K * sizeof(float));
    float *cy_host = (float*)malloc(K * sizeof(float));

    for (int c = 0; c < K; ++c) {
        cx_host[c] = centroids_host[2*c];
        cy_host[c] = centroids_host[2*c+1];
    }

    system("rm -rf iterations_opt2");
    system("mkdir -p iterations_opt2");

    // device memory
    float *d_x, *d_y, *d_cx, *d_cy;
    int   *d_labels;
    float *d_sum_x, *d_sum_y;
    int   *d_count;
    float *d_movement;

    CUDA_CHECK(cudaMalloc(&d_x, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cx, K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy, K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_x, K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count, K*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_movement, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, x_host, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y_host, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cx, cx_host, K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, cy_host, K*sizeof(float), cudaMemcpyHostToDevice));

    int threads_points = 256;   // Number of threads for the assign+reduce kernel
    int numSms;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, 0));
    int blocks_points = 32* numSms; // Number of blocks for the assign+reduce kernel

    int threads_clusters = 64;      // Number of threads for the update centroids kernel
    int blocks_clusters = 1;        // Number of blocks for the update centroids kernel

    size_t shared_fused = (5 * K) * sizeof(float); // cx, cy, sum_x_b, sum_y_b, count_b

    printf("K-means GPU fused: N=%d, K=%d\n", N, K);

    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iters; ++it) {

        // Reset movement
        CUDA_CHECK(cudaMemset(d_movement, 0, sizeof(float)));

        // Reset global sums and counts
        CUDA_CHECK(cudaMemset(d_sum_x, 0, K*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_y, 0, K*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_count, 0, K*sizeof(int)));

        // Fused assign and reduce kernel
        assign_reduce_kernel<<<blocks_points, threads_points, shared_fused>>>(
            d_x, d_y, d_cx, d_cy, d_labels,
            d_sum_x, d_sum_y, d_count,
            N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update centroids and compute movement
        update_centroids_kernel<<<blocks_clusters, threads_clusters>>>(
            d_cx, d_cy,
            d_sum_x, d_sum_y,
            d_count, d_movement,
            K);
        CUDA_CHECK(cudaDeviceSynchronize());

        float movement_host = 0;
        CUDA_CHECK(cudaMemcpy(&movement_host, d_movement, sizeof(float), cudaMemcpyDeviceToHost));

        //dump_iteration_data(it, K, N, d_cx, d_cy, d_labels);

        if (movement_host < epsilon) break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double,std::milli>(end-start).count();
    printf("Elapsed time: %.3f ms\n", elapsed);

    // copy centroids back
    CUDA_CHECK(cudaMemcpy(cx_host, d_cx, K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cy_host, d_cy, K*sizeof(float), cudaMemcpyDeviceToHost));

    for (int c = 0; c < K; ++c) {
        centroids_host[2*c]     = cx_host[c];
        centroids_host[2*c + 1] = cy_host[c];
    }

}


// =====================================================================
//                                MAIN
// =====================================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s data.csv\n", argv[0]);
        return 1;
    }

    float *x = NULL;
    float *y = NULL;
    int N = load_csv_soa(argv[1], &x, &y);
    if (N <= 0) return 1;

    int K = 3;
    float centroids[6] = {0,0, 5,5, 10,10};

    kmeans_gpu(x, y, centroids, N, K, 50, 1e-4);

    for (int c = 0; c < K; ++c)
        printf("C%d = (%f,%f)\n", c, centroids[2*c], centroids[2*c+1]);

    free(x);
    free(y);
    return 0;
}
