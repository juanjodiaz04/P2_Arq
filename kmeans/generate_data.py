import numpy as np
import csv
import argparse

def generate_points(N, K, std, outfile, radius=10.0):
    """
    Genera N puntos en 2D alrededor de K centroides equidistantes (ubicados en un círculo),
    con desviación estándar std, y los escribe en CSV.
    """

    # Centroides equidistantes distribuidos en un círculo
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    centroids = np.c_[radius * np.cos(angles), radius * np.sin(angles)]

    # Número de puntos por cluster (último recibe el resto)
    base = N // K
    remainder = N % K

    points = []

    for i in range(K):
        n_i = base + (1 if i < remainder else 0)
        cx, cy = centroids[i]

        # Generar puntos alrededor del centroide i
        cluster_points = np.random.normal(loc=[cx, cy], scale=std, size=(n_i, 2))
        points.append(cluster_points)

    points = np.vstack(points)

    # Guardar archivo CSV
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        for x, y in points:
            writer.writerow([x, y])

    print(f"Archivo generado: {outfile}")
    print(f"Centroides equidistantes usados:")
    for i, (cx, cy) in enumerate(centroids):
        print(f"  C{i}: ({cx:.3f}, {cy:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de puntos 2D con centroides equidistantes.")
    parser.add_argument("-n", type=int, required=True, help="Número total de puntos")
    parser.add_argument("-k", type=int, required=True, help="Número de clusters")
    parser.add_argument("-s", type=float, required=True, help="Desviación estándar")
    parser.add_argument("-o", type=str, default="data.csv", help="Archivo CSV de salida")
    parser.add_argument("--radius", type=float, default=10.0, help="Radio del círculo para centroids")

    args = parser.parse_args()

    generate_points(args.n, args.k, args.s, args.o, radius=args.radius)

# python generate_data.py -n 200000 -k 3 -s 1.5 -o dataset.csv --radius 5.0