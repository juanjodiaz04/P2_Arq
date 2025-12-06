#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define MAX_POINTS 10000000 // maximum number of points allowed

// Compute squared Euclidean distance (no sqrt for speed)
static inline float dist2(float px, float py, float cx, float cy)
{
    float dx = px - cx;
    float dy = py - cy;
    return dx * dx + dy * dy;
}

// Load points from a CSV file
int load_csv(const char *filename, float **out_points)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        printf("Error: cannot open file %s\n", filename);
        return -1;
    }

    float *points = malloc(MAX_POINTS * 2 * sizeof(float));
    if (!points)
    {
        printf("Error allocating memory\n");
        fclose(f);
        return -1;
    }

    int count = 0;
    float x, y;

    while (fscanf(f, "%f,%f", &x, &y) == 2)
    {
        if (count >= MAX_POINTS)
            break;
        points[2 * count] = x;
        points[2 * count + 1] = y;
        count++;
    }

    fclose(f);
    *out_points = points;
    return count;
}

// Assign each point to its nearest centroid
void assign_clusters_cpu(
    const float *points,    // size N*2
    const float *centroids, // size K*2
    int *labels,            // size N
    int N, int K)
{
    for (int i = 0; i < N; i++)
    {

        float px = points[2 * i];
        float py = points[2 * i + 1];

        float best_dist = FLT_MAX;
        int best_c = -1;

        for (int c = 0; c < K; c++)
        {
            float cx = centroids[2 * c];
            float cy = centroids[2 * c + 1];

            float d = dist2(px, py, cx, cy);
            if (d < best_dist)
            {
                best_dist = d;
                best_c = c;
            }
        }

        labels[i] = best_c;
    }
}

// Recompute centroids and return total centroid movement
float update_centroids_cpu(
    const float *points,
    float *centroids,
    const int *labels,
    int N, int K)
{
    float *sum = calloc(K * 2, sizeof(float));
    int *count = calloc(K, sizeof(int));

    // Accumulate positions of points per cluster
    for (int i = 0; i < N; i++)
    {
        int c = labels[i];
        sum[2 * c] += points[2 * i];
        sum[2 * c + 1] += points[2 * i + 1];
        count[c]++;
    }

    // Compute new centroids + track movement for convergence
    float movement = 0.0f;

    for (int c = 0; c < K; c++)
    {

        float newx = centroids[2 * c];
        float newy = centroids[2 * c + 1];

        if (count[c] > 0)
        {
            newx = sum[2 * c] / count[c];
            newy = sum[2 * c + 1] / count[c];
        }

        float dx = newx - centroids[2 * c];
        float dy = newy - centroids[2 * c + 1];
        movement += dx * dx + dy * dy;

        centroids[2 * c] = newx;
        centroids[2 * c + 1] = newy;
    }

    free(sum);
    free(count);
    return movement;
}

// Main K-means loop with convergence test
void kmeans_cpu(
    float *points,
    float *centroids,
    int N, int K,
    int max_iters,
    float epsilon)
{
    int *labels = malloc(N * sizeof(int));

    for (int it = 0; it < max_iters; it++)
    {

        assign_clusters_cpu(points, centroids, labels, N, K);
        float movement = update_centroids_cpu(points, centroids, labels, N, K);

        //printf("Iteration %d - centroid movement = %.6f\n", it, movement);

        if (movement < epsilon)
        {
            printf("Converged after %d iterations.\n", it);
            break;
        }
    }

    free(labels);
}

// Example usage: read CSV points, run K-means with fixed K
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s data.csv\n", argv[0]);
        return 1;
    }

    float *points = NULL;
    int N = load_csv(argv[1], &points);
    if (N <= 0)
        return 1;

    int K = 3;
    float centroids[6] = {0, 0, 5, 5, 10, 10}; // simple initial seeds

    printf("Loaded %d points.\n", N);

    int max_iters = 10;
    float ep = 1e-4f;

    // Measure execution time in CPU
    clock_t start = clock();
    
    kmeans_cpu(points, centroids, N, K, max_iters, ep);

    // End timing
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Elapsed time: %.3f ms\n", elapsed);

    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++)
    {
        printf("C%d = (%f, %f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }

    free(points);
    return 0;
}
