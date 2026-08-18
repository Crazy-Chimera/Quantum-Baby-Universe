"""
Quantum Baby Universe – V3, lattice 128×128, vectorised Compute step,
detection of R=1 to R=4.
"""
import numpy as np
import time
from scipy.signal import convolve2d

KERNEL = np.ones((3, 3), dtype=np.float32) / 9.0


def compute_step_vectorized(grid, threshold, noise):
    """One fully vectorised Compute step using a 3×3 convolution."""
    phi_local = convolve2d(grid, KERNEL, mode='same', boundary='wrap')
    repair_mask = phi_local < threshold
    noise_mask = np.random.random(grid.shape) < (noise * (1.0 - phi_local))
    new_grid = grid.copy()
    new_grid[repair_mask] = 1
    new_grid[noise_mask & ~repair_mask] = 0
    return new_grid


class BabyUniverseV3:
    """Large baby universe 128×128 with vectorised Compute step."""

    def __init__(self, N: int = 128, K_init: float = 0.9, seed: int = 42):
        self.name = "BabyUniverseV3"
        self.iteration = 0
        self.state = {}
        np.random.seed(seed)
        self.N = N
        self.K_init = K_init
        self.state['grid'] = np.random.choice(
            [0, 1], size=(N, N), p=[1 - K_init, K_init]
        ).astype(np.float32)
        self.state['phi_history'] = []
        self.state['defect_counts'] = []
        self.state['grid_history'] = []
        self.state['R1_detected'] = None
        self.state['R2_detected'] = None
        self.state['R3_detected'] = None
        self.state['R4_detected'] = None
        self.state['elegance_history'] = []
        self.state['threshold'] = 0.8
        self.state['noise'] = 0.01
        self.state['prediction_errors_1'] = []
        self.state['prediction_errors_2'] = []
        self.state['prediction_errors_3'] = []
        self.state['prediction_errors_4'] = []

    def observe(self, external_input=None):
        grid = self.state['grid']
        avg_phi = float(np.mean(grid))
        defects = int(np.sum(grid == 0))
        return {'avg_phi': avg_phi, 'defects': defects, 'iteration': self.iteration}

    def control(self, metrics, memory, policy):
        threshold = policy.get('threshold', self.state['threshold'])
        noise = policy.get('noise', self.state['noise'])
        self.state['grid'] = compute_step_vectorized(
            self.state['grid'], threshold, noise
        )
        self.iteration += 1
        self.state['phi_history'].append(metrics['avg_phi'])
        self.state['defect_counts'].append(metrics['defects'])
        if self.iteration % 50 == 0:
            self.state['grid_history'].append(self.state['grid'].copy())
        return {'action': 'compute_step'}

    def evaluate(self, metrics):
        C = metrics['defects'] / (self.N * self.N)
        K = metrics['avg_phi']
        return C / (K + 1e-6)

    def mutate(self, memory, policy, elegance):
        if len(self.state['elegance_history']) > 100:
            recent = self.state['elegance_history'][-100:]
            if min(recent) >= max(recent) * 0.98:
                policy['threshold'] = max(0.7, min(0.9,
                    policy.get('threshold', 0.8) + np.random.uniform(-0.005, 0.005)))
                policy['noise'] = max(0.001, min(0.05,
                    policy.get('noise', 0.01) + np.random.uniform(-0.0005, 0.0005)))
        self.state['elegance_history'].append(elegance)
        return memory, policy

    def termination_condition(self, metrics):
        return metrics['avg_phi'] > 0.99 and metrics['defects'] < 20

    def detect_higher_recursion(self):
        half = self.N // 2
        Q0 = self.state['grid'][:half, :half].flatten()
        Q1 = self.state['grid'][:half, half:].flatten()
        Q2 = self.state['grid'][half:, :half].flatten()
        Q3 = self.state['grid'][half:, half:].flatten()

        pred_1 = float(np.mean(Q0))
        actual_1 = float(np.mean(Q1))
        err_1 = abs(pred_1 - actual_1)
        r1 = err_1 < 0.03

        pred_2 = float(np.mean(Q2))
        err_2 = abs(pred_2 - err_1)
        r2 = err_2 < 0.02

        pred_3 = float(np.mean(Q3))
        err_3 = abs(pred_3 - err_2)
        r3 = err_3 < 0.015

        diag_1 = np.diag(self.state['grid']).flatten()
        diag_2 = np.diag(np.fliplr(self.state['grid'])).flatten()
        pred_4 = float(np.mean(diag_1))
        err_4 = abs(pred_4 - err_3)
        r4 = err_4 < 0.01

        self.state['prediction_errors_1'].append(err_1)
        self.state['prediction_errors_2'].append(err_2)
        self.state['prediction_errors_3'].append(err_3)
        self.state['prediction_errors_4'].append(err_4)

        return r1, r2, r3, r4


def run_simulation_128(N=128, steps=2000, seed=42):
    universe = BabyUniverseV3(N=N, K_init=0.9, seed=seed)
    policy = {'threshold': 0.8, 'noise': 0.01}
    memory = {}

    print(f"Starting Baby Universe V3: {N}×{N}, K_init=0.9, {steps} steps")
    print("=" * 70)

    start_time = time.time()

    for step in range(steps):
        metrics = universe.observe()
        if universe.termination_condition(metrics):
            print(f"Convergence at step {step}.")
            break
        universe.control(metrics, memory, policy)
        elegance = universe.evaluate(metrics)
        memory, policy = universe.mutate(memory, policy, elegance)

        if step >= 200:
            r1, r2, r3, r4 = universe.detect_higher_recursion()
            if r1 and universe.state['R1_detected'] is None:
                universe.state['R1_detected'] = step
                print(f"  >> R=1 detected at step {step}")
            if r2 and universe.state['R2_detected'] is None:
                universe.state['R2_detected'] = step
                print(f"  >> R=2 detected at step {step}")
            if r3 and universe.state['R3_detected'] is None:
                universe.state['R3_detected'] = step
                print(f"  >> R=3 detected at step {step}")
            if r4 and universe.state['R4_detected'] is None:
                universe.state['R4_detected'] = step
                print(f"  >> R=4 detected at step {step}")

        if step % 100 == 0:
            print(f"Step {step:4d}: Φ={metrics['avg_phi']:.4f}, "
                  f"Defects={metrics['defects']:5d}, Elegance={elegance:.6f}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")
    print(f"Final Φ: {universe.state['phi_history'][-1]:.4f}")
    print(f"R1 detected: {universe.state['R1_detected']}")
    print(f"R2 detected: {universe.state['R2_detected']}")
    print(f"R3 detected: {universe.state['R3_detected']}")
    print(f"R4 detected: {universe.state['R4_detected']}")

    return universe


if __name__ == '__main__':
    run_simulation_128(N=128, steps=2000, seed=42)
