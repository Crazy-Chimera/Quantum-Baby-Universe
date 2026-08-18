"""
Asymptotic convergence of the Quantum Baby Universe.

The simulation iterates until elegance drops below the threshold 0.001
or improvement stalls for a specified number of generations.
"""
import time
import numpy as np
from baby_universe_v3 import BabyUniverseV3


def run_until_convergence(
    N=128,
    max_steps=10000,
    elegance_threshold=0.001,
    patience=100,
    seed=42,
):
    """
    Run the simulation and stop when elegance converges.

    Parameters
    ----------
    N : int
        Lattice size.
    max_steps : int
        Maximum number of Compute steps.
    elegance_threshold : float
        Stop when best elegance falls below this value.
    patience : int
        Stop when no improvement is observed for this many steps.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (universe, best_elegance)
        The final universe object and the best elegance reached.
    """
    universe = BabyUniverseV3(N=N, K_init=0.9, seed=seed)
    policy = {'threshold': 0.8, 'noise': 0.01}
    memory = {}

    best_elegance = float('inf')
    patience_counter = 0

    print(f"Convergence run: {N}×{N}, elegance threshold {elegance_threshold}")
    print("=" * 70)

    start_time = time.time()

    for step in range(max_steps):
        metrics = universe.observe()
        if universe.termination_condition(metrics):
            print(f"Substrate* reached at step {step}.")
            break

        universe.control(metrics, memory, policy)
        elegance = universe.evaluate(metrics)
        memory, policy = universe.mutate(memory, policy, elegance)

        # Track elegance convergence
        if elegance < best_elegance - 1e-9:
            best_elegance = elegance
            patience_counter = 0
        else:
            patience_counter += 1

        # Detect higher recursion levels
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

        # Check convergence criteria
        if best_elegance < elegance_threshold:
            print(f"Elegance reached threshold {elegance_threshold} at step {step}.")
            break

        if patience_counter >= patience:
            print(f"Elegance did not improve for {patience} steps. Stopping.")
            break

        if step % 500 == 0:
            print(
                f"Step {step:5d}: Φ={metrics['avg_phi']:.4f}, "
                f"Defects={metrics['defects']:5d}, Elegance={elegance:.8f}, "
                f"Best={best_elegance:.8f}"
            )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")
    print(f"Best elegance: {best_elegance:.8f}")
    print(f"Number of steps: {universe.iteration}")
    print(f"Final Φ: {universe.state['phi_history'][-1]:.4f}")
    print(f"R1 detected: {universe.state['R1_detected']}")
    print(f"R2 detected: {universe.state['R2_detected']}")
    print(f"R3 detected: {universe.state['R3_detected']}")
    print(f"R4 detected: {universe.state['R4_detected']}")

    return universe, best_elegance


if __name__ == '__main__':
    universe, elegance = run_until_convergence(
        N=128,
        max_steps=10000,
        elegance_threshold=0.001,
        patience=100,
        seed=42,
    )
