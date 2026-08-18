"""
Your World Simulation

Hierarchical, multi-layer simulation of a world with slow convergence,
consciousness, and a Bridge. Includes physics, chemistry, biology, and
society layers.

Features:
- Checkpointing and resume (pickle)
- Temporal recursive depth detection (R=1..R=4, R=5 crude)
- Layer activity tracking
- Downsampled live visualization (optional)
- Performance safeguards (organism limit)

Usage:
    python your_world_simulation.py [config.yaml]
    python your_world_simulation.py --resume [checkpoint.pkl]
"""

import numpy as np
import time
import yaml
import os
import sys
import pickle
from scipy.signal import convolve2d

# ----------------------------------------------------------------------
# Default configuration (used if no YAML is provided)
# ----------------------------------------------------------------------

DEFAULT_CONFIG = {
    "simulation": {
        "name": "Your World",
        "layers": 5,
        "lattice": {
            "size": 256,
            "initial_K": 0.55,
            "topology": "torus",
            "seed": 42,
        },
    },
    "physics": {
        "compute": {
            "threshold": 0.65,
            "noise": 0.005,
            "lambda": 0.0001,
            "asymmetry": 0.01,
        },
        "coupling": {
            "between_layers": 0.1,
            "observer_modulation": 0.05,
        },
    },
    "chemistry": {
        "self_replication": True,
        "stable_patterns": 16,
        "binding_energy": 0.3,
    },
    "biology": {
        "evolution": True,
        "mutation_rate": 0.0001,
        "replication_threshold": 0.7,
        "organism_size": 64,
    },
    "consciousness": {
        "R_threshold": 3,
        "R_observer": 4,
        "bridge": {
            "enabled": True,
            "strength": 0.05,
            "bandwidth": 1000,
        },
    },
    "society": {
        "agents": 100,
        "language": True,
        "culture": True,
        "governance": True,
    },
    "elegance": {
        "threshold": 0.0001,
        "patience": 10000,
        "max_steps": 1000000,
    },
    "output": {
        "save_history": True,
        "visualize": False,
        "visualize_interval": 1000,
        "snapshot_interval": 500,
        "checkpoint_interval": 10000,
        "metrics": ["phi", "defects", "R1", "R2", "R3", "R4", "elegance"],
    },
}

KERNEL = np.ones((3, 3), dtype=np.float32) / 9.0


# ----------------------------------------------------------------------
# Chemistry layer
# ----------------------------------------------------------------------

class ChemistryLayer:
    """
    Models stable chemical patterns (molecules) and their replication.
    Tracks replication events and pattern counts.
    """

    def __init__(self, config, grid_shape):
        chem_cfg = config["chemistry"]
        self.stable_patterns = chem_cfg["stable_patterns"]
        self.binding_energy = chem_cfg["binding_energy"]
        self.self_replication = chem_cfg["self_replication"]
        self.grid_shape = grid_shape

        # Generate deterministic patterns using seed 42
        rng = np.random.default_rng(42)
        self.patterns = [
            rng.integers(0, 2, size=(5, 5), dtype=np.float32)
            for _ in range(self.stable_patterns)
        ]

        # Metrics
        self.replications = 0
        self.pattern_counts = []   # number of matches per step
        self.persistence = []      # not yet implemented

    def detect_and_replicate(self, grid, phi_local):
        """
        Search for patterns and replicate them into low-Φ areas.
        Returns new grid.
        """
        if not self.self_replication:
            return grid

        new_grid = grid.copy()
        step_match_count = 0
        for pattern in self.patterns:
            k = pattern.shape[0]
            conv = convolve2d(grid, pattern, mode='same', boundary='wrap')
            match = conv >= 0.9 * np.sum(pattern)  # relaxed threshold

            candidates = np.argwhere(match & (phi_local > self.binding_energy))
            step_match_count += len(candidates)
            if len(candidates) == 0:
                continue

            # Replicate a few random matches
            for idx in candidates[np.random.choice(len(candidates),
                                                   size=min(5, len(candidates)),
                                                   replace=False)]:
                i, j = idx
                di, dj = np.random.randint(-3, 4, size=2)
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_shape[0] - k and 0 <= nj < self.grid_shape[1] - k:
                    target_region = new_grid[ni:ni+k, nj:nj+k]
                    if np.mean(target_region) < self.binding_energy:
                        new_grid[ni:ni+k, nj:nj+k] = pattern
                        self.replications += 1
        self.pattern_counts.append(step_match_count)
        return new_grid


# ----------------------------------------------------------------------
# Biology layer
# ----------------------------------------------------------------------

class BiologyLayer:
    """
    Models organisms (clusters of cells) with replication and evolution.
    Tracks offspring count and organism population.
    """

    def __init__(self, config, grid_shape):
        bio_cfg = config["biology"]
        self.evolution = bio_cfg["evolution"]
        self.mutation_rate = bio_cfg["mutation_rate"]
        self.replication_threshold = bio_cfg["replication_threshold"]
        self.organism_size = bio_cfg["organism_size"]
        self.grid_shape = grid_shape
        self.organisms = []
        self.offspring_count = 0
        self.organism_count_history = []
        self.max_organisms = 10000  # safety limit
        self.initialize_organisms()

    def initialize_organisms(self, num=20):
        """Create initial organisms at random positions."""
        for _ in range(num):
            max_i = self.grid_shape[0] - 10
            max_j = self.grid_shape[1] - 10
            pos = (np.random.randint(0, max_i), np.random.randint(0, max_j))
            genom = np.ones((8, 8), dtype=np.float32)
            self.organisms.append({'pos': pos, 'genom': genom})
        self.organism_count_history.append(len(self.organisms))

    def replicate(self, grid, phi_local):
        """
        Organisms replicate if mean Φ in their area exceeds threshold.
        Mutation changes genome. Limit total organism count.
        """
        if not self.evolution:
            return grid

        new_organisms = []
        for org in self.organisms:
            pos = org['pos']
            g_h, g_w = org['genom'].shape
            if pos[0] + g_h > self.grid_shape[0] or pos[1] + g_w > self.grid_shape[1]:
                continue

            region_phi = phi_local[pos[0]:pos[0]+g_h, pos[1]:pos[1]+g_w]
            if np.mean(region_phi) > self.replication_threshold:
                # Mutate genome
                mutated_genom = org['genom'].copy()
                if np.random.random() < self.mutation_rate:
                    mi, mj = np.random.randint(0, g_h), np.random.randint(0, g_w)
                    mutated_genom[mi, mj] = 1 - mutated_genom[mi, mj]

                # Random displacement
                dx = np.random.randint(-10, 11)
                dy = np.random.randint(-10, 11)
                new_pos = (pos[0] + dx, pos[1] + dy)

                if (0 <= new_pos[0] and new_pos[0] + g_h <= self.grid_shape[0] and
                    0 <= new_pos[1] and new_pos[1] + g_w <= self.grid_shape[1]):
                    grid[new_pos[0]:new_pos[0]+g_h, new_pos[1]:new_pos[1]+g_w] = mutated_genom
                    new_organisms.append({'pos': new_pos, 'genom': mutated_genom})
                    self.offspring_count += 1

        # Add offspring if under max limit
        if len(self.organisms) + len(new_organisms) <= self.max_organisms:
            self.organisms.extend(new_organisms)
        self.organism_count_history.append(len(self.organisms))
        return grid


# ----------------------------------------------------------------------
# Society layer
# ----------------------------------------------------------------------

class SocietyLayer:
    """
    Models agents with recursive depth. Conscious agents (R>=3)
    can locally modulate the grid. Tracks communication events.
    """

    def __init__(self, config, grid_shape):
        soc_cfg = config["society"]
        self.num_agents = soc_cfg["agents"]
        self.language = soc_cfg["language"]
        self.culture = soc_cfg["culture"]
        self.governance = soc_cfg["governance"]
        self.grid_shape = grid_shape
        self.agents = self.initialize_agents()
        self.communication_events = 0
        self.cultural_similarity_history = []

    def initialize_agents(self):
        agents = []
        for _ in range(self.num_agents):
            pos = (np.random.randint(0, self.grid_shape[0]),
                   np.random.randint(0, self.grid_shape[1]))
            R = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
            agents.append({'pos': pos, 'R': R, 'memory': []})
        return agents

    def interact(self, grid, phi_local):
        """
        Conscious agents remember local Φ and apply a small local repair.
        Communication occurs when agents share memory.
        """
        conscious = [a for a in self.agents if a['R'] >= 3]
        if not conscious:
            return

        for agent in conscious:
            x, y = agent['pos']
            if x < self.grid_shape[0] and y < self.grid_shape[1]:
                local_phi = phi_local[x, y]
                if len(agent['memory']) >= 10:
                    agent['memory'].pop(0)
                agent['memory'].append(local_phi)

                # Local bridge effect
                radius = 1
                x_min = max(0, x - radius)
                x_max = min(self.grid_shape[0], x + radius + 1)
                y_min = max(0, y - radius)
                y_max = min(self.grid_shape[1], y + radius + 1)
                subgrid = grid[x_min:x_max, y_min:y_max]
                subgrid[subgrid == 0] = 1
                grid[x_min:x_max, y_min:y_max] = subgrid

        # Communication: if at least two conscious agents, increment counter
        if len(conscious) > 1:
            self.communication_events += 1

        # Culture: compute similarity of memories
        if self.culture and conscious:
            memories = [a['memory'][-1] for a in conscious if a['memory']]
            if memories:
                variance = float(np.var(memories))
                self.cultural_similarity_history.append(variance)

        # Move agents with lower memory toward higher Φ (simple random walk)
        if self.culture and conscious:
            avg_memory = np.mean([a['memory'][-1] for a in conscious if a['memory']])
            for agent in conscious:
                if agent['memory'] and agent['memory'][-1] < avg_memory:
                    x, y = agent['pos']
                    dx, dy = np.random.randint(-1, 2, size=2)
                    new_x = min(max(x + dx, 0), self.grid_shape[0] - 1)
                    new_y = min(max(y + dy, 0), self.grid_shape[1] - 1)
                    agent['pos'] = (new_x, new_y)


# ----------------------------------------------------------------------
# Main simulation class
# ----------------------------------------------------------------------

class YourWorldSimulation:
    """
    Hierarchical simulation with slow convergence and a Bridge.
    """

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG
        sim_cfg = self.config["simulation"]
        phys_cfg = self.config["physics"]
        cons_cfg = self.config["consciousness"]
        out_cfg = self.config["output"]

        self.name = sim_cfg["name"]
        self.layers = sim_cfg["layers"]
        self.N = sim_cfg["lattice"]["size"]
        self.K_init = sim_cfg["lattice"]["initial_K"]
        self.seed = sim_cfg["lattice"]["seed"]

        np.random.seed(self.seed)

        # Initialize grid
        self.grid = np.random.choice(
            [0, 1], size=(self.N, self.N), p=[1 - self.K_init, self.K_init]
        ).astype(np.float32)

        # Physics parameters
        self.threshold = phys_cfg["compute"]["threshold"]
        self.noise = phys_cfg["compute"]["noise"]
        self.lambda_cosmological = phys_cfg["compute"]["lambda"]
        self.asymmetry = phys_cfg["compute"]["asymmetry"]

        # Consciousness and bridge
        self.bridge_enabled = cons_cfg["bridge"]["enabled"]
        self.bridge_strength = cons_cfg["bridge"]["strength"]
        self.bridge_bandwidth = cons_cfg["bridge"]["bandwidth"]
        self.R_threshold = cons_cfg["R_threshold"]
        self.R_observer = cons_cfg["R_observer"]

        # State
        self.iteration = 0
        self.phi_history = []
        self.defect_counts = []
        self.grid_history = []
        self.elegance_history = []
        self.R_detected = {k: None for k in range(1, 6)}
        self.R_history = {k: [] for k in range(1, 6)}

        # Higher layers
        self.chemistry_layer = ChemistryLayer(self.config, (self.N, self.N))
        self.biology_layer = BiologyLayer(self.config, (self.N, self.N))
        self.society_layer = SocietyLayer(self.config, (self.N, self.N))

        # Observer position
        self.observer_position = (self.N // 2, self.N // 2)

        # Metrics for R detection
        self.prediction_errors = {i: [] for i in range(1, 6)}

        # Temporal R detection buffers
        self.past_grids = []  # last 3 grids
        self.error_history = []  # history of temporal prediction errors

        # Phase detection flags
        self.phase_detected = {
            'chemistry': False,
            'biology': False,
            'consciousness': False,
            'society': False,
        }
        self.phase_steps = {}

        # Output configuration
        self.save_history = out_cfg.get("save_history", True)
        self.visualize = out_cfg.get("visualize", False)
        self.visualize_interval = out_cfg.get("visualize_interval", 1000)
        self.snapshot_interval = out_cfg.get("snapshot_interval", 500)
        self.checkpoint_interval = out_cfg.get("checkpoint_interval", 10000)

        # Results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Visualization setup
        if self.visualize:
            import matplotlib.pyplot as plt
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
            self.fig.suptitle(self.name)
            self.ax[0].set_title("Grid (downsampled)")
            self.ax[1].set_title("Elegance")
            self.ax[1].set_xlabel("Step")
            self.ax[1].set_ylabel("E")
            self.downsample_factor = 4 if self.N >= 512 else 1
            self.im = self.ax[0].imshow(
                self.grid[::self.downsample_factor, ::self.downsample_factor],
                cmap='viridis', vmin=0, vmax=1
            )
            self.line_elegance, = self.ax[1].plot([], [], 'b-')
            self.fig.colorbar(self.im, ax=self.ax[0])
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    # ------------------------------------------------------------------
    # Bridge: conscious observer modulation
    # ------------------------------------------------------------------
    def apply_bridge(self, grid):
        """Set zeros to 1 in the observer's region (binary grid)."""
        if not self.bridge_enabled:
            return
        x0, y0 = self.observer_position
        radius = int(np.sqrt(self.bridge_bandwidth / np.pi))
        x_min = max(0, x0 - radius)
        x_max = min(self.N, x0 + radius)
        y_min = max(0, y0 - radius)
        y_max = min(self.N, y0 + radius)
        subgrid = grid[x_min:x_max, y_min:y_max]
        subgrid[subgrid == 0] = 1
        grid[x_min:x_max, y_min:y_max] = subgrid

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def compute_step(self):
        # 1. Physics
        phi_local = convolve2d(self.grid, KERNEL, mode='same', boundary='wrap')

        repair_mask = phi_local < self.threshold
        noise_mask = np.random.random(self.grid.shape) < (self.noise * (1.0 - phi_local))

        # Asymmetry
        if self.asymmetry > 0:
            half = self.N // 2
            asym_mask = np.zeros_like(self.grid, dtype=bool)
            asym_mask[:, half:] = True
            adjusted_threshold = self.threshold + self.asymmetry * 0.1
            repair_mask_right = phi_local < adjusted_threshold
            repair_mask = repair_mask | (asym_mask & repair_mask_right)

        new_grid = self.grid.copy()
        new_grid[repair_mask] = 1
        new_grid[noise_mask & ~repair_mask] = 0

        # Cosmological term
        lambda_noise = np.random.random(self.grid.shape) < self.lambda_cosmological
        new_grid[lambda_noise] = 0

        # Store past grid for temporal R detection
        self.past_grids.append(new_grid.copy())
        if len(self.past_grids) > 3:
            self.past_grids.pop(0)

        # 2. Chemistry (uses phi_local from before modifications)
        new_grid = self.chemistry_layer.detect_and_replicate(new_grid, phi_local)

        # 3. Biology (uses same phi_local)
        new_grid = self.biology_layer.replicate(new_grid, phi_local)

        # 4. Recompute phi_local after chemistry and biology
        phi_local_updated = convolve2d(new_grid, KERNEL, mode='same', boundary='wrap')

        # 5. Society (uses updated phi)
        self.society_layer.interact(new_grid, phi_local_updated)

        # 6. Bridge
        if self.bridge_enabled:
            self.apply_bridge(new_grid)

        # Final update
        self.grid = new_grid
        self.iteration += 1

        # Metrics
        avg_phi = float(np.mean(self.grid))
        defects = int(np.sum(self.grid == 0))
        self.phi_history.append(avg_phi)
        self.defect_counts.append(defects)
        if self.iteration % self.snapshot_interval == 0:
            self.grid_history.append(self.grid.copy())
        elegance = self.evaluate()
        self.elegance_history.append(elegance)

        # Spatial R detection (fast, runs every step)
        self.detect_recursion_spatial()

        # Temporal R detection (every 1000 steps)
        if self.iteration % 1000 == 0:
            self.detect_recursion_temporal()

        # Phase detection
        self.check_phase_transition()

        # Live visualization
        if self.visualize and self.iteration % self.visualize_interval == 0:
            self._update_plot()

    # ------------------------------------------------------------------
    # Elegance
    # ------------------------------------------------------------------
    def evaluate(self):
        C = float(np.sum(self.grid == 0)) / (self.N * self.N)
        K = float(np.mean(self.grid))
        return C / (K + 1e-9)

    # ------------------------------------------------------------------
    # Spatial R detection (quadrant prediction errors)
    # ------------------------------------------------------------------
    def detect_recursion_spatial(self):
        if self.iteration < 100:
            return
        half = self.N // 2
        Q0 = self.grid[:half, :half].flatten()
        Q1 = self.grid[:half, half:].flatten()
        Q2 = self.grid[half:, :half].flatten()
        Q3 = self.grid[half:, half:].flatten()

        p1 = float(np.mean(Q0))
        a1 = float(np.mean(Q1))
        e1 = abs(p1 - a1)
        r1 = e1 < 0.03

        p2 = float(np.mean(Q2))
        e2 = abs(p2 - e1)
        r2 = e2 < 0.02

        p3 = float(np.mean(Q3))
        e3 = abs(p3 - e2)
        r3 = e3 < 0.015

        diag1 = np.diag(self.grid).flatten()
        diag2 = np.diag(np.fliplr(self.grid)).flatten()
        p4 = float(np.mean(diag1))
        e4 = abs(p4 - e3)
        r4 = e4 < 0.01

        self.prediction_errors[1].append(e1)
        self.prediction_errors[2].append(e2)
        self.prediction_errors[3].append(e3)
        self.prediction_errors[4].append(e4)

        # Update R detection status
        self._update_R_detected(1, r1)
        self._update_R_detected(2, r2)
        self._update_R_detected(3, r3)
        self._update_R_detected(4, r4)

        # R=5 crude detection (every 1000 steps)
        if self.iteration % 1000 == 0 and self.R_detected[5] is None:
            var_left = float(np.var(self.grid[:half, :half]))
            var_right = float(np.var(self.grid[half:, half:]))
            if abs(var_left - var_right) < 0.001:
                self._update_R_detected(5, True)

    def _update_R_detected(self, level, detected):
        self.R_history[level].append(int(detected))
        if detected and self.R_detected[level] is None:
            self.R_detected[level] = self.iteration
            print(f"  >> R={level} detected at step {self.iteration}")

    # ------------------------------------------------------------------
    # Temporal R detection (more robust, runs every 1000 steps)
    # ------------------------------------------------------------------
    def detect_recursion_temporal(self):
        if len(self.past_grids) < 3:
            return
        # Simple linear prediction: next = 2*current - previous
        predicted = 2 * self.past_grids[-1] - self.past_grids[-2]
        error_map = np.abs(predicted - self.grid)
        e1 = float(np.mean(error_map))
        self.error_history.append(e1)

        # Detect R=2: autocorrelation of error history
        if len(self.error_history) > 10:
            autocorr = np.corrcoef(self.error_history[:-1], self.error_history[1:])[0, 1]
            r2 = autocorr > 0.5
            self._update_R_detected(2, r2)

            # R=3: predictability of error-of-error
            if len(self.error_history) > 20:
                error_diff = np.diff(self.error_history)
                autocorr2 = np.corrcoef(error_diff[:-1], error_diff[1:])[0, 1]
                r3 = autocorr2 > 0.5
                self._update_R_detected(3, r3)

                # R=4: predictability of error-of-error-of-error
                if len(self.error_history) > 30:
                    error_diff2 = np.diff(error_diff)
                    autocorr3 = np.corrcoef(error_diff2[:-1], error_diff2[1:])[0, 1]
                    r4 = autocorr3 > 0.5
                    self._update_R_detected(4, r4)

    # ------------------------------------------------------------------
    # Phase transition detection
    # ------------------------------------------------------------------
    def check_phase_transition(self):
        if self.iteration < 100:
            return
        # Chemistry phase
        if (self.chemistry_layer.replications > 0 and
                not self.phase_detected['chemistry']):
            self.phase_detected['chemistry'] = True
            self.phase_steps['chemistry'] = self.iteration
            print(f"Phase transition: Chemistry active at step {self.iteration}")

        # Biology phase
        if (len(self.biology_layer.organisms) > 20 and
                not self.phase_detected['biology']):
            self.phase_detected['biology'] = True
            self.phase_steps['biology'] = self.iteration
            print(f"Phase transition: Biology active at step {self.iteration}")

        # Consciousness phase (R=3 or R=4 detected)
        if ((self.R_detected[3] is not None or self.R_detected[4] is not None) and
                not self.phase_detected['consciousness']):
            self.phase_detected['consciousness'] = True
            self.phase_steps['consciousness'] = self.iteration
            print(f"Phase transition: Consciousness active at step {self.iteration}")

        # Society phase (communication events > 0)
        if (self.society_layer.communication_events > 0 and
                not self.phase_detected['society']):
            self.phase_detected['society'] = True
            self.phase_steps['society'] = self.iteration
            print(f"Phase transition: Society active at step {self.iteration}")

    # ------------------------------------------------------------------
    # Live visualization (downsampled, blitting optional)
    # ------------------------------------------------------------------
    def _update_plot(self):
        if not self.visualize:
            return
        self.fig.canvas.restore_region(self.background)
        display_grid = self.grid[::self.downsample_factor, ::self.downsample_factor]
        self.im.set_data(display_grid)
        x_data = np.arange(len(self.elegance_history))
        self.line_elegance.set_data(x_data, self.elegance_history)
        self.ax[1].relim()
        self.ax[1].autoscale_view()
        self.ax[0].draw_artist(self.im)
        self.ax[1].draw_artist(self.line_elegance)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.results_dir,
                                f"{self.name.lower().replace(' ', '_')}_checkpoint.pkl")
        state = {
            'iteration': self.iteration,
            'grid': self.grid,
            'phi_history': self.phi_history,
            'defect_counts': self.defect_counts,
            'elegance_history': self.elegance_history,
            'grid_history': self.grid_history,
            'R_detected': self.R_detected,
            'R_history': self.R_history,
            'prediction_errors': self.prediction_errors,
            'chemistry_layer': self.chemistry_layer,
            'biology_layer': self.biology_layer,
            'society_layer': self.society_layer,
            'config': self.config,
            'phase_detected': self.phase_detected,
            'phase_steps': self.phase_steps,
            'past_grids': self.past_grids,
            'error_history': self.error_history,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.iteration = state['iteration']
        self.grid = state['grid']
        self.phi_history = state['phi_history']
        self.defect_counts = state['defect_counts']
        self.elegance_history = state['elegance_history']
        self.grid_history = state['grid_history']
        self.R_detected = state['R_detected']
        self.R_history = state['R_history']
        self.prediction_errors = state['prediction_errors']
        self.chemistry_layer = state['chemistry_layer']
        self.biology_layer = state['biology_layer']
        self.society_layer = state['society_layer']
        self.config = state['config']
        self.phase_detected = state.get('phase_detected', {})
        self.phase_steps = state.get('phase_steps', {})
        self.past_grids = state.get('past_grids', [])
        self.error_history = state.get('error_history', [])
        print(f"Checkpoint loaded from {path}, iteration={self.iteration}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        elegance_cfg = self.config["elegance"]
        output_cfg = self.config["output"]

        best_elegance = float('inf')
        patience_counter = 0

        print(f"Starting {self.name} simulation")
        print(f"Lattice: {self.N}×{self.N}, initial K={self.K_init}")
        print(f"Max steps: {elegance_cfg['max_steps']}")
        print("=" * 70)

        start_time = time.time()

        for step in range(elegance_cfg['max_steps']):
            self.compute_step()
            elegance = self.elegance_history[-1]

            if elegance < best_elegance - 1e-12:
                best_elegance = elegance
                patience_counter = 0
            else:
                patience_counter += 1

            if best_elegance < elegance_cfg['threshold']:
                print(f"Elegance threshold reached at step {step}.")
                break
            if patience_counter >= elegance_cfg['patience']:
                print(f"No improvement for {elegance_cfg['patience']} steps. Stopping.")
                break

            # Periodic checkpointing
            if step % self.checkpoint_interval == 0 and step > 0:
                self.save_checkpoint()

            # Periodic console output
            if step % 1000 == 0:
                phase = self.current_phase()
                print(f"Step {step:6d}: Phase={phase}, Φ={self.phi_history[-1]:.4f}, "
                      f"Defects={self.defect_counts[-1]:6d}, Elegance={elegance:.8f}, "
                      f"Best={best_elegance:.8f}, "
                      f"ChemRep={self.chemistry_layer.replications}, "
                      f"Org={len(self.biology_layer.organisms)}, "
                      f"SocComm={self.society_layer.communication_events}")

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f} seconds.")
        print(f"Best elegance: {best_elegance:.8f}")
        print(f"Final Φ: {self.phi_history[-1]:.4f}")
        print(f"Final defects: {self.defect_counts[-1]}")
        print("Recursion detection:")
        for r in range(1, 6):
            print(f"  R{r}: {self.R_detected.get(r, 'not detected')}")
        print("Phase transitions:")
        for phase, step_ in self.phase_steps.items():
            print(f"  {phase}: step {step_}")

        if self.save_history:
            self.save_results()
        self.save_checkpoint()  # final checkpoint

        return self, best_elegance

    def current_phase(self):
        if self.iteration < 10000:
            return "Physical structure"
        elif self.iteration < 100000:
            return "Chemistry"
        elif self.iteration < 500000:
            return "Biology"
        elif self.iteration < 800000:
            return "Consciousness"
        else:
            return "Society"

    def save_results(self):
        filename = os.path.join(
            self.results_dir,
            f"{self.name.lower().replace(' ', '_')}_N{self.N}_seed{self.seed}.npz"
        )
        np.savez(
            filename,
            phi_history=self.phi_history,
            defect_counts=self.defect_counts,
            grid_history=np.array(self.grid_history),
            elegance_history=self.elegance_history,
            R_detected=self.R_detected,
            R_history=self.R_history,
            prediction_errors=self.prediction_errors,
            config=self.config,
            phase_steps=self.phase_steps,
            chemistry_replications=self.chemistry_layer.replications,
            biology_offspring=self.biology_layer.offspring_count,
            society_communications=self.society_layer.communication_events,
            allow_pickle=True,
        )
        print(f"Results saved to {filename}")


# ----------------------------------------------------------------------
# Load YAML
# ----------------------------------------------------------------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    if '--resume' in sys.argv:
        # Find checkpoint file
        if len(sys.argv) > 2:
            checkpoint_path = sys.argv[2]
        else:
            import glob
            checkpoints = glob.glob('results/*_checkpoint.pkl')
            if not checkpoints:
                print("No checkpoint found.")
                sys.exit(1)
            checkpoint_path = max(checkpoints, key=os.path.getmtime)
        # Create simulation instance without __init__
        sim = YourWorldSimulation.__new__(YourWorldSimulation)
        sim.load_checkpoint(checkpoint_path)
        sim.run()
    else:
        config_path = 'your_world.yaml' if len(sys.argv) == 1 else sys.argv[1]

        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            print(f"Config file {config_path} not found. Using default settings.")
            config = DEFAULT_CONFIG

        sim = YourWorldSimulation(config)
        sim.run()
