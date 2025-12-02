import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import imageio


# -----------------------------------------------------------
# Load the original 2D points (x,y) used for K-means
# -----------------------------------------------------------
def load_points(csv_path):
    xs = []
    ys = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Robust parser: supports "x,y", "x, y", "x y", "x;y", "x\t y"
            line = line.replace(',', ' ').replace(';', ' ')
            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except:
                pass

    return np.array(xs), np.array(ys)


# -----------------------------------------------------------
# Parse one iteration CSV file
# -----------------------------------------------------------
def load_iteration_csv(path):
    centroids = []
    labels = []

    section = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                if "centroids" in line:
                    section = "centroids"
                elif "labels" in line:
                    section = "labels"
                continue

            if section == "centroids":
                parts = line.split(',')
                if len(parts) == 3:
                    cx = float(parts[1])
                    cy = float(parts[2])
                    centroids.append([cx, cy])

            elif section == "labels":
                labels.append(int(line))

    return np.array(centroids), np.array(labels)


# -----------------------------------------------------------
# Plot a single frame and return it as an RGB array
# -----------------------------------------------------------
def plot_frame(points_x, points_y, centroids, labels, title):
    K = len(centroids)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points_x, points_y, c=labels, s=8, cmap='tab10', alpha=0.75)

    # Plot centroids
    ax.scatter(centroids[:,0], centroids[:,1], 
               s=200, c='black', marker='X', edgecolor='white', linewidth=1.5)

    # Plot Voronoi diagram if possible
    if K >= 2:
        try:
            vor = Voronoi(centroids)
            for ridge in vor.ridge_vertices:
                if -1 in ridge:
                    continue
                vtx = vor.vertices[ridge]
                ax.plot(vtx[:,0], vtx[:,1], 'k-', linewidth=1)
        except:
            pass  # ignore Voronoi errors

    ax.set_title(title)
    ax.set_xlim(min(points_x), max(points_x))
    ax.set_ylim(min(points_y), max(points_y))
    fig.tight_layout()

    # *** CRITICAL FIX: FORCE RENDER BEFORE CONVERTING ***
    fig.canvas.draw()

    # Convert canvas to RGB numpy array
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return frame


# -----------------------------------------------------------
# MAIN: Build GIF from iteration CSV files
# -----------------------------------------------------------
def main():
    # Load the original dataset of points
    points_x, points_y = load_points("input2.csv")
    print(points_x.shape, points_y.shape)

    print(points_x[:5], points_y[:5])

    # List all CSVs inside "iterations"
    iteration_files = sorted([
        f for f in os.listdir("iterations")
        if f.startswith("iteration_") and f.endswith(".csv")
    ])
    print(f"Found {len(iteration_files)} iteration files.")

    frames = []

    for fname in iteration_files:
        path = os.path.join("iterations", fname)
        centroids, labels = load_iteration_csv(path)

        print(f"Iteration {fname}: centroids shape {centroids.shape}, labels shape {labels.shape}")

        frame = plot_frame(points_x, points_y, centroids, labels, title=f"{fname}")
        frames.append(frame)

    # Save GIF
    imageio.mimsave("kmeans_convergence.gif", frames, fps=2, loop=0)
    print("GIF saved as kmeans_convergence.gif")


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
