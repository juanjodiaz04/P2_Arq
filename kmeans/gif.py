import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio


# -----------------------------------------------------------
# Load the original 2D points (x,y)
# -----------------------------------------------------------
def load_points(csv_path):
    xs = []
    ys = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Robust parser
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

    ax.scatter(centroids[:, 0], centroids[:, 1],
               s=200, c='black', marker='X',
               edgecolor='white', linewidth=1.5)

    ax.set_title(title)
    ax.set_xlim(min(points_x), max(points_x))
    ax.set_ylim(min(points_y), max(points_y))
    fig.tight_layout()

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return frame


# -----------------------------------------------------------
# MAIN: Build GIF from iteration CSVs
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate GIF from K-Means iteration CSVs.")
    parser.add_argument("--csv", default="dataset.csv", help="Path to the original dataset CSV.")
    parser.add_argument("--iter", default="iterations_opt", help="Folder containing iteration_XXX.csv files.")
    parser.add_argument("--out", default="kmeans_V1.gif", help="Output GIF filename.")

    args = parser.parse_args()

    # Load the dataset
    points_x, points_y = load_points(args.csv)
    print(points_x.shape, points_y.shape)

    # Read iteration CSVs
    iteration_files = sorted([
        f for f in os.listdir(args.iter)
        if f.startswith("iteration_") and f.endswith(".csv")
    ])
    print(f"Found {len(iteration_files)} iteration files.")

    frames = []

    for fname in iteration_files:
        path = os.path.join(args.iter, fname)
        centroids, labels = load_iteration_csv(path)

        print(f"Iteration {fname}: centroids {centroids.shape}, labels {labels.shape}")

        frame = plot_frame(points_x, points_y, centroids, labels, title=f"{fname}")
        frames.append(frame)

    # Save GIF
    imageio.mimsave(args.out, frames, fps=2, loop=0)
    print(f"GIF saved as {args.out}")


# -----------------------------------------------------------
if __name__ == "__main__":
    main()

# example
# python gif.py --iter iterations_sim --out kmeans_V1.gif
# python gif.py --iter iterations_opt --out kmeans_V2.gif
# python gif.py --iter iterations_opt2 --out kmeans_V3.gif