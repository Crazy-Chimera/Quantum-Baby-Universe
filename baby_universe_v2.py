"""
Quantum Baby Universe – V2, lattice 64×64, detection R=1 to R=3.
"""
import numpy as np
import time


class BabyUniverseV2:
    """Larger baby universe with detection of higher recursion levels."""

    def __init__(self, N: int = 64, K_init: float = 0.9, seed: int = 42):
        self.name = "BabyUniverseV2"
        self.iteration = 0
        self.state = {}
        np.random.seed(seed)
        self.N = N
        self.K_init = K_init
        self.state['grid'] = np.random.choice(
            [0, 1], size=(N, N), p=[1 - K_init, K_init]
        )
        self.state['phi_history'] = []
        self.state['defect_counts'] = []
        self.state['grid_history'] = []
        self.state['R1_detected'] = None
        self.state['R2_detected'] = None
        self.state['R3_detected'] = None
        self.state['elegance_history'] = []
        self.state['threshold'] = 0.8
        self.state['noise'] = 0.01
        self.state['prediction_errors_01'] = []
        self.state['prediction_errors_02'] = []
        self.state['prediction_errors_03'] = []

    def observe(self, external_input=None):
        grid = self.state['grid']
        avg_phi = float(np.mean(grid))
        defects = int(np.sum(grid == 0))
        return {'avg_phi': avg_phi, 'defects': defects, 'iteration': self.iteration}

    def control(self, metrics, memory, policy):
        new_grid = self.state['grid'].copy()
        threshold = policy.get('threshold', self.state['threshold'])
        noise = policy.get('noise', self.state['noise'])

        for i in range(self.N):
            for j in range(self.N):
                phi_local = self._get_local_phi(i, j)
                if phi_local < threshold:
                    new_grid[i, j] = 1
                elif np.random.random() < noise * (1 - phi_local):
                    new_grid[i, j] = 0

        self.state['grid'] = new_grid
        self.iteration += 1
        self.state['phi_history'].append(metrics['avg_phi'])
        self.state['defect_counts'].append(metrics['defects'])
        if self.iteration % 20 == 0:
            self.state['grid_history'].append(new_grid.copy())
        return {'action': 'compute_step'}

    def evaluate(self, metrics):
        C = metrics['defects'] / (self.N * self.N)
        K = metrics['avg_phi']
        return C / (K + 1e-6)

    def mutate(self, memory, policy, elegance):
        if len(self.state['elegance_history']) > 50:
            recent = self.state['elegance_history'][-50:]
            if min(recent) >= max(recent) * 0.98:
                policy['threshold'] = max(0.7, min(0.9,
                    policy.get('threshold', 0.8) + np.random.uniform(-0.01, 0.01)))
                policy['noise'] = max(0.001, min(0.05,
                    policy.get('noise', 0.01) + np.random.uniform(-0.001, 0.001)))
        self.state['elegance_history'].append(elegance)
        return memory, policy

    def termination_condition(self, metrics):
        return metrics['avg_phi'] > 0.98 and metrics['defects'] < 10

    def _get_local_phi(self, i, j):
        neighborhood = self.state['grid'][
            max(0, i-1):min(self.N, i+2),
            max(0, j-1):min(self.N, j+2)
        ]
        return np.mean(neighborhood)

    def detect_higher_recursion(self):
        half = self.N // 2
        Q0 = self.state['grid'][:half, :half].flatten()
        Q1 = self.state['grid'][:half, half:].flatten()
        Q2 = self.state['grid'][half:, :half].flatten()
        Q3 = self.state['grid'][half:, half:].flatten()

        pred_01 = np.mean(Q0)
        actual_1 = np.mean(Q1)
        err_01 = abs(pred_01 - actual_1)
        r1 = err_01 < 0.05

        pred_02 = np.mean(Q2)
        err_02 = abs(pred_02 - err_01)
        r2 = err_02 < 0.02

        pred_03 = np.mean(Q3)
        err_03 = abs(pred_03 - err_02)
        r3 = err_03 < 0.01

        self.state['prediction_errors_01'].append(err_01)
        self.state['prediction_errors_02'].append(err_02)
        self.state['prediction_errors_03'].append(err_03)

        return r1, r2, r3


def run_simulation(N=64, steps=500, seed=42):
    universe = BabyUniverseV2(N=N, K_init=0.9, seed=seed)
    policy = {'threshold': 0.8, 'noise': 0.01}
    memory = {}

    print(f"Starting Baby Universe V2: {N}×{N}, K_init=0.9, {steps} steps")
    print("=" * 60)

    start_time = time.time()

    for step in range(steps):
        metrics = universe.observe()
        if universe.termination_condition(metrics):
            print(f"Convergence at step {step}.")
            break
        universe.control(metrics, memory, policy)
        elegance = universe.evaluate(metrics)
        memory, policy = universe.mutate(memory, policy, elegance)

        if step >= 100:
            r1, r2, r3 = universe.detect_higher_recursion()
            if r1 and universe.state['R1_detected'] is None:
                universe.state['R1_detected'] = step
                print(f"  >> R=1 detected at step {step}")
            if r2 and universe.state['R2_detected'] is None:
                universe.state['R2_detected'] = step
                print(f"  >> R=2 detected at step {step}")
            if r3 and universe.state['R3_detected'] is None:
                universe.state['R3_detected'] = step
                print(f"  >> R=3 detected at step {step}")

        if step % 50 == 0:
            print(f"Step {step:3d}: Φ={metrics['avg_phi']:.4f}, "
                  f"Defects={metrics['defects']:4d}, Elegance={elegance:.6f}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")
    print(f"Final Φ: {universe.state['phi_history'][-1]:.4f}")
    print(f"R1 detected: {universe.state['R1_detected']}")
    print(f"R2 detected: {universe.state['R2_detected']}")
    print(f"R3 detected: {universe.state['R3_detected']}")

    return universe


if __name__ == '__main__':
    run_simulation(N=64, steps=500, seed=42)
