"""
visualize.py

Load results from .npz and plot key metrics and grid snapshots.

Usage:
    python visualize.py results/your_world_N256_seed42.npz
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


def find_latest_result(directory="results"):
    files = glob.glob(os.path.join(directory, "*.npz"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def plot_results(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    print(f"Loading {filename} ...")
    data = np.load(filename, allow_pickle=True)

    required_keys = [
        "phi_history",
        "defect_counts",
        "elegance_history",
        "grid_history",
        "R_detected",
        "config",
    ]
    for key in required_keys:
        if key not in data:
            print(f"Missing key '{key}' in {filename}.")
            return

    phi_history = data["phi_history"]
    defect_counts = data["defect_counts"]
    elegance_history = data["elegance_history"]
    grid_history = data["grid_history"]
    R_detected = data["R_detected"].item() if hasattr(data["R_detected"], "item") else data["R_detected"]
    config = data["config"].item() if hasattr(data["config"], "item") else data["config"]

    sim_name = config.get("simulation", {}).get("name", "Your World")
    lattice_size = config.get("simulation", {}).get("lattice", {}).get("size", "?")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Simulation: {sim_name}  (N={lattice_size})", fontsize=16)

    axes[0, 0].plot(phi_history, color="teal")
    axes[0, 0].set_title("Φ in time")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Φ")
    axes[0, 0].grid(True)

    axes[0, 1].plot(defect_counts, color="crimson")
    axes[0, 1].set_title("Defects in time")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Defects")
    axes[0, 1].grid(True)

    axes[0, 2].plot(elegance_history, color="navy")
    axes[0, 2].set_title("Elegance in time")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("E = C / K")
    axes[0, 2].set_yscale("log")
    axes[0, 2].grid(True, which="both", linestyle="--", alpha=0.5)

    n_snapshots = len(grid_history)
    if n_snapshots > 0:
        snapshot_indices = [0, n_snapshots // 2, n_snapshots - 1]
        for i, idx in enumerate(snapshot_indices):
            ax = axes[1, i]
            im = ax.imshow(grid_history[idx], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Grid (snapshot {idx})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        for i in range(3):
            axes[1, i].axis("off")
            axes[1, i].text(0.5, 0.5, "No snapshots", ha="center", va="center")

    axes[1, 2].axis("off")
    text = "Recursion detection:\n\n"
    for r in range(1, 6):
        val = R_detected.get(r, "not detected")
        text += f"R{r}: {val}\n"
    axes[1, 2].text(0.1, 0.5, text, fontsize=12, verticalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = find_latest_result("results")
        if filename is None:
            print("No .npz files found in results/.")
            print("Usage: python visualize.py [path/to/file.npz]")
            sys.exit(1)
        print(f"Using latest result: {filename}")

    plot_results(filename)
