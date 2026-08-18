"""
Integration of Quantum Baby Universe with the Φ‑Fabric simulator.
Measures latency, estimated energy, and elegance on a CCP tile.
"""
import time
import numpy as np
from scipy.signal import convolve2d
from dataclasses import dataclass

KERNEL = np.ones((3, 3), dtype=np.float32) / 9.0


@dataclass
class TileMetrics:
    energy_per_op: float
    ops_per_second: float
    tile_type: str


CCP_TILE = TileMetrics(energy_per_op=1e-12, ops_per_second=1e10, tile_type='CCP')
STC_TILE = TileMetrics(energy_per_op=5e-13, ops_per_second=1e12, tile_type='STC')


class FabricAcceleratedUniverse:
    """Quantum Baby Universe running on a Φ‑Fabric tile."""

    def __init__(self, N: int = 128, K_init: float = 0.9, seed: int = 42,
                 tile: TileMetrics = CCP_TILE):
        self.N = N
        self.tile = tile
        np.random.seed(seed)
        self.grid = np.random.choice(
            [0, 1], size=(N, N), p=[1 - K_init, K_init]
        ).astype(np.float32)
        self.iteration = 0
        self.phi_history = []
        self.elegance_history = []
        self.energy_consumed = 0.0
        self.total_latency = 0.0
        self.ops_per_step = N * N * 9

    def compute_step(self, threshold, noise):
        """One Compute step on the CCP tile."""
        start = time.perf_counter()

        phi_local = convolve2d(self.grid, KERNEL, mode='same', boundary='wrap')
        repair_mask = phi_local < threshold
        noise_mask = np.random.random(self.grid.shape) < (noise * (1.0 - phi_local))
        new_grid = self.grid.copy()
        new_grid[repair_mask] = 1
        new_grid[noise_mask & ~repair_mask] = 0
        self.grid = new_grid

        end = time.perf_counter()
        latency = end - start
        energy = self.ops_per_step * self.tile.energy_per_op

        self.iteration += 1
        self.total_latency += latency
        self.energy_consumed += energy

        avg_phi = float(np.mean(self.grid))
        defects = int(np.sum(self.grid == 0))
        C = defects / (self.N * self.N) + energy / 1000.0
        K = avg_phi
        elegance = C / (K + 1e-6)

        self.phi_history.append(avg_phi)
        self.elegance_history.append(elegance)

        return {'latency': latency, 'energy': energy, 'elegance': elegance}

    def report(self):
        print("=" * 70)
        print("Φ‑FABRIC REPORT – QUANTUM BABY UNIVERSE")
        print("=" * 70)
        print(f"Tile: {self.tile.tile_type}")
        print(f"Lattice: {self.N}×{self.N}")
        print(f"Steps: {self.iteration}")
        print(f"Operations per step: {self.ops_per_step:,}")
        print(f"Total operations: {self.iteration * self.ops_per_step:,}")
        print(f"Total latency: {self.total_latency:.3f} s")
        print(f"Average latency per step: {self.total_latency / max(self.iteration, 1):.6f} s")
        print(f"Energy consumption: {self.energy_consumed:.6f} J")
        print(f"Average energy per step: {self.energy_consumed / max(self.iteration, 1):.9f} J")
        print(f"Final Φ: {self.phi_history[-1]:.4f}")
        print(f"Final elegance: {self.elegance_history[-1]:.6f}")
        print("=" * 70)


def run_on_fabric(N=128, steps=1000, seed=42):
    """Run Quantum Baby Universe on simulated Φ‑Fabric."""
    print("Starting Quantum Baby Universe on Φ‑Fabric (CCP tile)...")
    print()

    universe = FabricAcceleratedUniverse(N=N, seed=seed, tile=CCP_TILE)
    threshold = 0.8
    noise = 0.01

    for step in range(steps):
        result = universe.compute_step(threshold, noise)
        if step % 200 == 0:
            print(f"Step {step:4d}: latency={result['latency']:.6f} s, "
                  f"energy={result['energy']:.9f} J, elegance={result['elegance']:.6f}")

    print()
    universe.report()
    return universe


if __name__ == '__main__':
    run_on_fabric(N=128, steps=1000, seed=42)
