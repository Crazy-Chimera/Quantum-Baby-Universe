"""
Quantum Baby Universe – V1, basic loop 32×32.
"""
import numpy as np
import time


class BabyUniverseLoop:
    """Main simulation loop. Manages the informion lattice."""

    def __init__(self, N: int = 32, K_init: float = 0.9, seed: int = 42):
        self.name = "BabyUniverse"
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
        self.state['mutual_information'] = []
        self.state['R1_detected'] = None
        self.state['elegance_history'] = []
        self.state['threshold'] = 0.8
        self.state['noise'] = 0.01

    def observe(self, external_input=None):
        grid = self.state['grid']
        avg_phi = float(np.mean(grid))
        defects = int(np.sum(grid == 0))
        mi = self._compute_mutual_information(grid)
        return {
            'avg_phi': avg_phi,
            'defects': defects,
            'mutual_information': float(mi),
            'iteration': self.iteration,
        }

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
        self.state['mutual_information'].append(metrics['mutual_information'])

        if self.iteration % 10 == 0:
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
        return metrics['avg_phi'] > 0.98 and metrics['defects'] < 5

    def _get_local_phi(self, i, j):
        neighborhood = self.state['grid'][
            max(0, i-1):min(self.N, i+2),
            max(0, j-1):min(self.N, j+2)
        ]
        return np.mean(neighborhood)

    def _compute_mutual_information(self, grid):
        half = self.N // 2
        left = grid[:, :half].flatten()
        right = grid[:, half:].flatten()
        p_left_1 = np.mean(left)
        p_right_1 = np.mean(right)
        p_both_1 = np.mean(left * right)
        if p_left_1 == 0 or p_right_1 == 0 or p_both_1 == 0:
            return 0.0
        mi = p_both_1 * np.log(p_both_1 / (p_left_1 * p_right_1))
        return float(mi)

    def detect_R1(self):
        if len(self.state['phi_history']) < 50:
            return False, 0.0
        half = self.N // 2
        left_mean = np.mean(self.state['grid'][:, :half])
        right_mean = np.mean(self.state['grid'][:, half:])
        prediction_error = abs(left_mean - right_mean)
        detected = prediction_error < 0.1
        accuracy = 1.0 - prediction_error
        return detected, accuracy


def main():
    print("=" * 60)
    print("QUANTUM BABY UNIVERSE – LoopOS V1")
    print("=" * 60)

    universe = BabyUniverseLoop(N=32, K_init=0.9, seed=42)
    policy = {'threshold': 0.8, 'noise': 0.01}
    memory = {}

    print(f"Initial Φ: {np.mean(universe.state['grid']):.4f}")
    print(f"Initial defects: {np.sum(universe.state['grid'] == 0)}")
    print()

    start_time = time.time()

    for step in range(200):
        metrics = universe.observe()
        if universe.termination_condition(metrics):
            print(f"Convergence reached at step {step}.")
            break
        universe.control(metrics, memory, policy)
        elegance = universe.evaluate(metrics)
        memory, policy = universe.mutate(memory, policy, elegance)

        if step >= 50 and universe.state['R1_detected'] is None:
            r1_detected, accuracy = universe.detect_R1()
            if r1_detected:
                universe.state['R1_detected'] = step
                print(f"  >> R=1 detected at step {step}! (accuracy: {accuracy:.4f})")

        if step % 20 == 0:
            print(f"Step {step:3d}: Φ={metrics['avg_phi']:.4f}, "
                  f"Defects={metrics['defects']:4d}, MI={metrics['mutual_information']:.4f}")

    elapsed = time.time() - start_time

    print(f"\nSimulation completed in {elapsed:.1f} seconds.")
    print(f"Final Φ: {universe.state['phi_history'][-1]:.4f}")
    print(f"Final defects: {universe.state['defect_counts'][-1]}")
    print(f"R1 detected: {universe.state['R1_detected']}")
    if universe.state['elegance_history']:
        print(f"Elegance (last): {universe.state['elegance_history'][-1]:.6f}")


if __name__ == '__main__':
    main()
