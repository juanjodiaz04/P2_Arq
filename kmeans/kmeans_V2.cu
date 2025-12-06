#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
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

// Load 2D points from a CSV file into two separate arrays (Structure of Arrays - SoA)
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

// --------------- Dump iteration data to CSV file ---------------------

void dump_iteration_data(
    int it,
    int K,
    int N,
    const float *d_cx,
    const float *d_cy,
    const int   *d_labels)
{
    // ---------- static buffers (allocated once) ----------
    static float *cx_dump     = NULL;
    static float *cy_dump     = NULL;
    static int   *labels_dump = NULL;

    if (cx_dump == NULL) {
        cx_dump     = (float*)malloc(K * sizeof(float));
        cy_dump     = (float*)malloc(K * sizeof(float));
        labels_dump = (int*)malloc(N * sizeof(int));
        if (!cx_dump || !cy_dump || !labels_dump) {
            printf("ERROR: host dump allocation failed.\n");
            exit(EXIT_FAILURE);
        }
    }

    // ---------- Copy from device ----------
    CUDA_CHECK(cudaMemcpy(cx_dump,     d_cx,    K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cy_dump,     d_cy,    K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels_dump, d_labels, N * sizeof(int),  cudaMemcpyDeviceToHost));

    // ---------- Build filename ----------
    char fname[256];
    sprintf(fname, "iterations_opt/iteration_%03d.csv", it);

    // ---------- Write file ----------
    FILE *f = fopen(fname, "w");
    if (!f) {
        printf("WARNING: could not write file %s\n", fname);
        return;
    }

    // Write centroids
    fprintf(f, "# centroids:\n");
    for (int c = 0; c < K; c++) {
        fprintf(f, "C%d,%f,%f\n", c, cx_dump[c], cy_dump[c]);
    }

    // Write labels
    fprintf(f, "# labels:\n");
    for (int i = 0; i < N; i++) {
        fprintf(f, "%d\n", labels_dump[i]);
    }

    fclose(f);
}


//-------------------------- KERNELS CUDA -----------------------------

// Kernel 1: reset global sums and counts
// sum_x[c] = 0, sum_y[c] = 0, count[c] = 0

__global__
void reset_sums_kernel(float *sum_x, float *sum_y, int *count, int K) {
    // Global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread resets one centroid
    if (tid < K) {
        sum_x[tid] = 0.0f;
        sum_y[tid] = 0.0f;
        count[tid] = 0;
    }
}

// Kernel 2: assign each point to the nearest centroid "(1 point per thread)" → Grid - Stride Loop
// Uses:
//  - SoA: x[i], y[i]
//  - centroids in shared memory
//  - grid-stride loop over N

__global__
void assign_clusters_kernel(
    const float *__restrict__ x, // memory alignment for better performance
    const float *__restrict__ y,
    const float *__restrict__ cx,
    const float *__restrict__ cy,
    int *__restrict__ labels,
    int N,
    int K)
{
    extern __shared__ float shared_centroids[]; // Dynamic shared memory between threads of the same block
    float *cx_s = shared_centroids;             // [0..K-1]=cx
    float *cy_s = shared_centroids + K;         // [K..2K-1]=cy

    // Copy centroids to shared memory (once per block)
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        cx_s[c] = cx[c];
        cy_s[c] = cy[c];
    }
    __syncthreads(); // Wait until all centroids are in shared memory

    int tid    = threadIdx.x + blockIdx.x * blockDim.x; // global thread ID
    int stride = blockDim.x * gridDim.x;                // total number of threads

    for (int i = tid; i < N; i += stride) {
        float px = x[i];
        float py = y[i];

        float best_dist = FLT_MAX;
        int best_c = -1;

        // Search nearest centroid
        for (int c = 0; c < K; ++c) {
            float dx = px - cx_s[c];
            float dy = py - cy_s[c];
            float d2 = dx * dx + dy * dy;
            if (d2 < best_dist) {
                best_dist = d2;
                best_c = c;
            }
        }

        labels[i] = best_c;
    }
}

// Kernel 3: block-wise reduction with atomics
// - each block has sum_x_b, sum_y_b, count_b in shared memory
// - each thread processes points with a grid-stride loop
// - then merges into global sum_x, sum_y, and count with atomics

__global__
void reduce_centroids_kernel(
    const float *__restrict__ x,
    const float *__restrict__ y,
    const int   *__restrict__ labels,
    float *sum_x,
    float *sum_y,
    int   *count,
    int N,
    int K)
{
    // Shared memory per block
    extern __shared__ unsigned char smem[];                 // Dynamic shared memory between threads of the same block
    float *sum_x_b = (float*)smem;                          // [0..K-1]=sum_x_b
    float *sum_y_b = (float*)(smem + K * sizeof(float));    // [K..2K-1]=sum_y_b
    int   *count_b = (int*)(smem + 2 * K * sizeof(float));  // [2K..3K-1]=count_b

    // Init shared memory accumulators (1 per block)
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        sum_x_b[c] = 0.0f;
        sum_y_b[c] = 0.0f;
        count_b[c] = 0;
    }
    __syncthreads(); // Wait until each block's shared memory is initialized (Avoid trash)

    int tid    = threadIdx.x + blockIdx.x * blockDim.x; // global thread ID
    int stride = blockDim.x * gridDim.x;                // total number of threads

    // Accumulate in block shared memory each thread's points
    // grid-stride loop over N
    for (int i = tid; i < N; i += stride) {
        int c = labels[i];
        if (c >= 0 && c < K) {
            float px = x[i];
            float py = y[i];
            atomicAdd(&sum_x_b[c], px);
            atomicAdd(&sum_y_b[c], py);
            atomicAdd(&count_b[c], 1);
        }
    }

    __syncthreads(); // Wait until all threads have accumulated their points

    // Merge block shared memory into global memory with atomics
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        atomicAdd(&sum_x[c], sum_x_b[c]);
        atomicAdd(&sum_y[c], sum_y_b[c]);
        atomicAdd(&count[c], count_b[c]);
    }
}

// Kernel 4: update centroids and compute total movement

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

            float newx = sum_x[c] / (float)count[c];
            float newy = sum_y[c] / (float)count[c];

            float dx = newx - oldx;
            float dy = newy - oldy;

            cx[c] = newx;
            cy[c] = newy;

            atomicAdd(movement, dx*dx + dy*dy); // keep track of total movement
        }
    }
}


//--------------------- K-MEANS GPU  ------------------------

void kmeans_gpu(
    float *x_host,
    float *y_host,
    float *centroids_host, // Size 2*K, AoS format (cx0,cy0,cx1,cy1,...)
    int N,
    int K,
    int max_iters,
    float epsilon)
{
    // Separate host centroids into SoA cx_host, cy_host
    float *cx_host = (float*)malloc(K * sizeof(float));
    float *cy_host = (float*)malloc(K * sizeof(float));
    for (int c = 0; c < K; ++c) {
        cx_host[c] = centroids_host[2 * c];
        cy_host[c] = centroids_host[2 * c + 1];
    }

    // Create directory for iterations
    system("rm -rf iterations_opt");
    system("mkdir -p iterations_opt");

    // Allocate device memory
    float *d_x, *d_y;
    float *d_cx, *d_cy;
    int   *d_labels;
    float *d_sum_x, *d_sum_y;
    int   *d_count;
    float *d_movement;

    CUDA_CHECK(cudaMalloc(&d_x,      N * sizeof(float))); // points x
    CUDA_CHECK(cudaMalloc(&d_y,      N * sizeof(float))); // points y
    CUDA_CHECK(cudaMalloc(&d_cx,     K * sizeof(float))); // centroids x
    CUDA_CHECK(cudaMalloc(&d_cy,     K * sizeof(float))); // centroids y
    CUDA_CHECK(cudaMalloc(&d_labels, N * sizeof(int)));   // labels
    CUDA_CHECK(cudaMalloc(&d_sum_x,  K * sizeof(float))); // sum x
    CUDA_CHECK(cudaMalloc(&d_sum_y,  K * sizeof(float))); // sum y
    CUDA_CHECK(cudaMalloc(&d_count,  K * sizeof(int)));   // count
    CUDA_CHECK(cudaMalloc(&d_movement, sizeof(float)));   // movement

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_x,  x_host, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  y_host, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cx, cx_host, K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, cy_host, K * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threads_points = 512;   // Number of threads for the assign+reduce kernel
    int numSms;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, 0));
    int blocks_points = 32* numSms; // Number of blocks for the assign+reduce kernel

    int threads_clusters = 64;
    int blocks_clusters  = 1; // K between 1 and 128

    size_t shared_assign  = 2 * K * sizeof(float);                   // cx_s, cy_s
    size_t shared_reduce  = 2 * K * sizeof(float) + K * sizeof(int); // sum_x_b, sum_y_b, count_b

    printf("K-means GPU: N=%d, K=%d, blocks_points=%d, threads_points=%d\n", N, K, blocks_points, threads_points);

    // Kernel execution timing
    auto start_kernel = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iters; ++it) {
        // 1) reset sums and counts
        reset_sums_kernel<<<blocks_clusters, threads_clusters>>>(d_sum_x, d_sum_y, d_count, K);
        CUDA_CHECK(cudaGetLastError());

        // 2) set movement to 0
        CUDA_CHECK(cudaMemset(d_movement, 0, sizeof(float)));

        // 3) assign clusters (use shared memory for centroids)
        assign_clusters_kernel<<<blocks_points, threads_points, shared_assign>>>(d_x, d_y, d_cx, d_cy, d_labels, N, K);
        CUDA_CHECK(cudaGetLastError());

        // 4) block-wise reduction with atomics
        reduce_centroids_kernel<<<blocks_points, threads_points, shared_reduce>>>(d_x, d_y, d_labels, d_sum_x, d_sum_y, d_count, N, K);
        CUDA_CHECK(cudaGetLastError());

        // 5) update centroids and accumulate movement
        update_centroids_kernel<<<blocks_clusters, threads_clusters>>>(d_cx, d_cy, d_sum_x, d_sum_y, d_count, d_movement, K);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all kernels to finish

        // Copy movement to host
        float movement_host = 0.0f;
        CUDA_CHECK(cudaMemcpy(&movement_host, d_movement, sizeof(float), cudaMemcpyDeviceToHost)); 

        //printf("Iter %d - centroid movement = %.6f\n", it, movement_host);


        // DUMP CURRENT STATE TO FILE
        dump_iteration_data(it, K, N, d_cx, d_cy, d_labels);


        if (movement_host < epsilon) {
            //printf("Converged after %d iterations.\n", it);
            break;
        }
    }

    // Kernel end timing
    auto end_kernel = std::chrono::high_resolution_clock::now();
    double elapsed_kernel = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    printf("Elapsed time kernel: %.3f ms\n", elapsed_kernel);

    // Copy final centroids back to host
    CUDA_CHECK(cudaMemcpy(cx_host, d_cx, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cy_host, d_cy, K * sizeof(float), cudaMemcpyDeviceToHost));

    for (int c = 0; c < K; ++c) {
        centroids_host[2 * c]     = cx_host[c];
        centroids_host[2 * c + 1] = cy_host[c];
    }

    // Free
    free(cx_host);
    free(cy_host);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_cx));
    CUDA_CHECK(cudaFree(d_cy));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_sum_x));
    CUDA_CHECK(cudaFree(d_sum_y));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_movement));
}

//------------------------------- MAIN --------------------------------

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s data.csv K [seed]\n", argv[0]);
        printf("  data.csv: input CSV file with points\n");
        printf("  K: number of clusters\n");
        printf("  seed (optional): random seed for centroid initialization\n");
        return 1;
    }

    float *x = NULL;
    float *y = NULL;
    int N = load_csv_soa(argv[1], &x, &y);
    if (N <= 0) {
        return 1;
    }

    // Read K from command line
    int K = atoi(argv[2]);
    if (K <= 0) {
        printf("Error: K must be positive\n");
        free(x);
        free(y);
        return 1;
    }

    // Read seed (optional, default = current time)
    unsigned int seed = (argc >= 4) ? (unsigned int)atoi(argv[3]) : (unsigned int)time(NULL);
    srand(seed);
    printf("Using random seed: %u\n", seed);

    // Find min/max of data points to initialize centroids within range
    float min_x = x[0], max_x = x[0];
    float min_y = y[0], max_y = y[0];
    for (int i = 1; i < N; ++i) {
        if (x[i] < min_x) min_x = x[i];
        if (x[i] > max_x) max_x = x[i];
        if (y[i] < min_y) min_y = y[i];
        if (y[i] > max_y) max_y = y[i];
    }

    // Generate random centroids within data range
    float *centroids = (float*)malloc(2 * K * sizeof(float));
    if (!centroids) {
        printf("Error allocating centroids\n");
        free(x);
        free(y);
        return 1;
    }

    printf("Initializing %d random centroids in range [%.2f, %.2f] x [%.2f, %.2f]\n",
           K, min_x, max_x, min_y, max_y);
    
    for (int c = 0; c < K; ++c) {
        centroids[2 * c]     = min_x + (max_x - min_x) * ((float)rand() / RAND_MAX);
        centroids[2 * c + 1] = min_y + (max_y - min_y) * ((float)rand() / RAND_MAX);
    }

    printf("Loaded %d points.\n", N);

    int   max_iters = 50;
    float epsilon   = 1e-4f;

    // Time measurement
    auto start = std::chrono::high_resolution_clock::now();

    kmeans_gpu(x, y, centroids, N, K, max_iters, epsilon);

    // Time measurement end
    auto end =  std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Elapsed time: %.3f ms\n", elapsed);

    printf("\nFinal centroids (GPU):\n");
    for (int c = 0; c < K; ++c) {
        printf("C%d = (%f, %f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }

    free(centroids);
    free(x);
    free(y);

    return 0;
}
