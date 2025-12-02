#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include <cuda_runtime.h>

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

//-------------------- Read CSV points (AoS) ---------------------
// points: [x0,y0,x1,y1,...]

int load_csv(const char *filename, float **out_points) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Error: cannot open file %s\n", filename);
        return -1;
    }

    float *points = (float *)malloc(MAX_POINTS * 2 * sizeof(float));
    if (!points) {
        printf("Error allocating memory\n");
        fclose(f);
        return -1;
    }

    int count = 0;
    float x, y;
    while (fscanf(f, "%f,%f", &x, &y) == 2) {
        if (count >= MAX_POINTS) break;
        points[2 * count]     = x;
        points[2 * count + 1] = y;
        count++;
    }

    fclose(f);
    *out_points = points;
    return count;
}

//-------------------------- KERNEL CUDA -----------------------------
// Kernel único por iteración:
//  - Asigna label al punto i
//  - Actualiza sum_x, sum_y, count con atomicAdd

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

        // buscar centroide más cercano
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

        // acumulación global directa 
        atomicAdd(&sum[2 * best_c],     px);
        atomicAdd(&sum[2 * best_c + 1], py);
        atomicAdd(&count[best_c],       1);
    }
}

//--------------------- K-MEANS GPU ------------------------

void kmeans_gpu(
    float *h_points,     // N*2
    float *h_centroids,  // 2*K
    int N,
    int K,
    int max_iters,
    float epsilon)
{
    // device arrays 
    float *d_points   = nullptr;
    float *d_centroids= nullptr;
    int   *d_labels   = nullptr;
    float *d_sum      = nullptr; // 2*K
    int   *d_count    = nullptr; // K

    CUDA_CHECK(cudaMalloc(&d_points,    2 * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, 2 * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels,    N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum,       2 * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count,     K * sizeof(int)));

    // copiar datos iniciales
    CUDA_CHECK(cudaMemcpy(d_points,   h_points,    2 * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, 2 * K * sizeof(float),
                          cudaMemcpyHostToDevice));

    // host buffers para sumas y conteos
    float *h_sum   = (float*)malloc(2 * K * sizeof(float));
    int   *h_count = (int*)  malloc(    K * sizeof(int));

    // configuración kernels
    int blockSize = 256;
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,
                                      cudaDevAttrMultiProcessorCount, 0));
    int numBlocks = 32 * numSMs;

    printf("K-means GPU fused: N=%d, K=%d, blocks=%d, threads=%d\n",
           N, K, numBlocks, blockSize);

    for (int it = 0; it < max_iters; ++it) {

        // limpiar sum y count en device
        CUDA_CHECK(cudaMemset(d_sum,   0, 2 * K * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_count, 0,     K * sizeof(int)));

        // kernel único: asigna + acumula
        assign_and_accumulate_kernel<<<numBlocks, blockSize>>>(
            d_points, d_centroids, d_labels, d_sum, d_count, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // copiar sumas y conteos a host
        CUDA_CHECK(cudaMemcpy(h_sum,   d_sum,   2 * K * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_count, d_count,     K * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // actualizar centroides en host y calcular movimiento
        float movement = 0.0f;
        for (int c = 0; c < K; ++c) {

            float oldx = h_centroids[2 * c];
            float oldy = h_centroids[2 * c + 1];

            float newx = oldx;
            float newy = oldy;

            if (h_count[c] > 0) {
                newx = h_sum[2 * c]     / h_count[c];
                newy = h_sum[2 * c + 1] / h_count[c];
            }

            float dx = newx - oldx;
            float dy = newy - oldy;
            movement += dx * dx + dy * dy;

            h_centroids[2 * c]     = newx;
            h_centroids[2 * c + 1] = newy;
        }

        // mandar centroides actualizados al device
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids,
                              2 * K * sizeof(float),
                              cudaMemcpyHostToDevice));

        printf("Iter %d - movement = %.6f\n", it, movement);

        if (movement < epsilon) {
            printf("Converged after %d iterations.\n", it);
            break;
        }
    }

    free(h_sum);
    free(h_count);

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_count));
}

//------------------------------- MAIN --------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s data.csv\n", argv[0]);
        return 1;
    }

    float *points = NULL;
    int N = load_csv(argv[1], &points);
    if (N <= 0) {
        return 1;
    }

    int K = 3;
    float centroids[6] = {0.0f, 0.0f, 5.0f, 5.0f, 10.0f, 10.0f};

    printf("Loaded %d points.\n", N);

    int   max_iters = 50;
    float epsilon   = 1e-4f;

    kmeans_gpu(points, centroids, N, K, max_iters, epsilon);

    printf("\nFinal centroids (GPU):\n");
    for (int c = 0; c < K; ++c) {
        printf("C%d = (%f, %f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }

    free(points);
    return 0;
}
