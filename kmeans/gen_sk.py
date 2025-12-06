import numpy as np
import csv
import argparse
from sklearn.datasets import make_blobs

def generate_points_with_blobs(N, K, std, outfile, center_min=-10.0, center_max=10.0, seed=None):
    """
    Genera N puntos en 2D usando sklearn.make_blobs con K clusters.
    Guarda los puntos en un archivo CSV.
    """

    X, y = make_blobs(
        n_samples=N,
        n_features=2,
        centers=K,
        cluster_std=std,
        center_box=(center_min, center_max),
        random_state=seed
    )

    # Guardar archivo CSV
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        for x, y_ in X:
            writer.writerow([x, y_])

    print(f"Archivo generado: {outfile}")

    # Los centroides generados realmente (calculados a partir de los datos)
    centroids = np.array([X[y == i].mean(axis=0) for i in range(K)])

    print("Centroides aproximados generados:")
    for i, (cx, cy) in enumerate(centroids):
        print(f"  C{i}: ({cx:.3f}, {cy:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de puntos 2D usando sklearn.make_blobs.")
    parser.add_argument("-n", type=int, required=True, help="Número total de puntos")
    parser.add_argument("-k", type=int, required=True, help="Número de clusters")
    parser.add_argument("-s", type=float, required=True, help="Desviación estándar de los clusters")
    parser.add_argument("-o", type=str, default="data.csv", help="Archivo CSV de salida")
    parser.add_argument("--min", type=float, default=-10.0, help="Límite inferior para las coordenadas")
    parser.add_argument("--max", type=float, default=10.0, help="Límite superior para las coordenadas")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad")

    args = parser.parse_args()

    generate_points_with_blobs(
        args.n, args.k, args.s, args.o,
        center_min=args.min,
        center_max=args.max,
        seed=args.seed
    )

# python gen_sk.py -n 200000 -k 3 -s 1.5 -o dataset_blobs.csv --seed 42