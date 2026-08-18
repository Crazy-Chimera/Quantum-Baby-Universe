"""
Visualization tools for Quantum Baby Universe.

Provides functions to plot the lattice state, the mean Φ curve,
the elegance curve, and the prediction errors for recursion
levels R=1 to R=4.
"""
import matplotlib.pyplot as plt
import numpy as np


def visualize_universe(universe, filename='baby_universe.png'):
    """Plot the initial and final lattices, the mean Φ, and elegance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Quantum Baby Universe', fontsize=16)

    # Initial lattice
    ax1 = axes[0, 0]
    im1 = ax1.imshow(
        universe.state['grid_history'][0],
        cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest'
    )
    ax1.set_title(f'Initial Φ={universe.state["phi_history"][0]:.3f}')
    plt.colorbar(im1, ax=ax1, label='Φ')

    # Final lattice
    ax2 = axes[0, 1]
    im2 = ax2.imshow(
        universe.state['grid_history'][-1],
        cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest'
    )
    ax2.set_title(f'Final Φ={universe.state["phi_history"][-1]:.3f}')
    plt.colorbar(im2, ax=ax2, label='Φ')

    # Mean Φ over time
    ax3 = axes[1, 0]
    ax3.plot(universe.state['phi_history'], 'b-', linewidth=1)
    ax3.axhline(y=1.0, color='gold', linestyle='--', alpha=0.5, label='Substrate*')
    ax3.set_xlabel('Compute step')
    ax3.set_ylabel('Mean Φ')
    ax3.set_title('Convergence to Substrate*')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Elegance over time
    ax4 = axes[1, 1]
    ax4.plot(universe.state['elegance_history'], 'r-', linewidth=1)
    ax4.set_xlabel('Compute step')
    ax4.set_ylabel('Elegance C/K')
    ax4.set_title('Elegance minimisation')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def visualize_prediction_errors(universe, filename='prediction_errors.png'):
    """Plot prediction errors for R=1 through R=4 if available."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Prediction errors for R=1 to R=4', fontsize=16)

    # Support both V2 and V3 key naming conventions.
    error_sets = [
        (
            'R=1',
            universe.state.get(
                'prediction_errors_1',
                universe.state.get('prediction_errors_01', [])
            ),
            axes[0, 0]
        ),
        (
            'R=2',
            universe.state.get(
                'prediction_errors_2',
                universe.state.get('prediction_errors_02', [])
            ),
            axes[0, 1]
        ),
        (
            'R=3',
            universe.state.get(
                'prediction_errors_3',
                universe.state.get('prediction_errors_03', [])
            ),
            axes[1, 0]
        ),
        (
            'R=4',
            universe.state.get('prediction_errors_4', []),
            axes[1, 1]
        ),
    ]

    for title, err_list, ax in error_sets:
        if not err_list:
            ax.text(
                0.5, 0.5, 'No data',
                ha='center', va='center',
                transform=ax.transAxes
            )
            ax.set_title(title)
            continue
        ax.plot(err_list, 'b-', linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Prediction error')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


if __name__ == '__main__':
    from baby_universe_v3 import run_simulation_128
    universe = run_simulation_128(N=128, steps=2000, seed=42)
    visualize_universe(universe, 'baby_universe_v3.png')
    visualize_prediction_errors(universe, 'prediction_errors_v3.png')
