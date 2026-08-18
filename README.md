# Quantum-Baby-Universe

This repository contains two related simulation projects built around emergent structure on informion lattices:

- **Quantum Baby Universe** (root-level files): a multi-version lattice simulation from V1 to V3 with recursion detection, convergence runs, visualization, and a fabric-integration performance model.
- **Your World** (`Your_World/`): a hierarchical 5-layer simulation (physics, chemistry, biology, consciousness, society) with checkpointing, recursion-depth detection, and post-run analytics.

---

## Repository Structure

### Root: Quantum Baby Universe

- `baby_universe_v1.py`  
  Basic 32×32 loop simulation (`BabyUniverseLoop`) with local repair/noise dynamics, elegance tracking, MI metric, and R=1 detection.

- `baby_universe_v2.py`  
  64×64 simulation (`BabyUniverseV2`) with higher recursion detection (R=1..R=3), expanded state tracking, and longer runs.

- `baby_universe_v3.py`  
  128×128 vectorized simulation (`BabyUniverseV3`) using `scipy.signal.convolve2d`, with recursion detection (R=1..R=4) and improved scaling.

- `fabric_integration.py`  
  Φ‑Fabric tile simulation model (`FabricAcceleratedUniverse`) with latency, estimated energy, and elegance reporting; includes CCP/STC tile metrics.

- `convergence.py`  
  Asymptotic run controller for V3 (`run_until_convergence`) with elegance threshold + patience stopping criteria.

- `visualization.py`  
  Plotting utilities for lattice snapshots, Φ history, elegance history, and prediction error series.

- `tests/test_baby_universe.py`  
  Pytest coverage for initialization, step execution, recursion detection outputs, vectorized compute constraints, and fabric metrics.

- `Makefile`  
  Convenience targets:
  - `make run-v1`
  - `make run-v2`
  - `make run-v3`
  - `make run-fabric`
  - `make converge`
  - `make visualize`
  - `make test`

- `.gitignore`  
  Ignores Python caches and generated artifacts (`*.png`, `*.npz`, etc.), keeps `results/.gitkeep`.

---

### `Your_World/`: Hierarchical Multi-Layer Simulation

- `Your_World/your_world_simulation.py`  
  Main engine (`YourWorldSimulation`) implementing:
  - physics compute operator (repair + noise + cosmological term + asymmetry),
  - chemistry pattern replication (`ChemistryLayer`),
  - biology replication/evolution (`BiologyLayer`),
  - society interactions (`SocietyLayer`),
  - bridge modulation,
  - spatial/temporal recursion-depth detection (R1..R5 crude),
  - checkpointing and resume,
  - optional live visualization.

- `Your_World/visualize.py`  
  Post-run visualization of saved `.npz` results:
  - Φ trajectory,
  - defect trajectory,
  - elegance trajectory (log scale),
  - snapshots,
  - recursion detection summary.

- `Your_World/your_world.yaml`  
  Default configuration for simulation, physics, chemistry, biology, consciousness, society, elegance stop criteria, and output intervals.

- `Your_World/requirements.txt`  
  Dependencies:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pytest`
  - `pyyaml`

---

## Theoretical Basis (shared direction)

Both simulation branches are grounded in the **Φ‑Elegance** idea:

- local and global field coherence is represented by Φ,
- defects/inconsistencies contribute to complexity,
- system evolution is monitored via an elegance ratio `E = C / K`,
- recursion depth (`R`) and observer/bridge effects are used as proxies for higher-order structure and self-modelling.

---

## Quick Start

### Quantum Baby Universe (root)

Install dependencies:

```bash
pip install numpy scipy matplotlib pytest pyyaml
```

Run:

```bash
make run-v1
make run-v2
make run-v3
make run-fabric
make converge
make test
```

### Your World

Run with default config:

```bash
python Your_World/your_world_simulation.py Your_World/your_world.yaml
```

Resume from checkpoint:

```bash
python Your_World/your_world_simulation.py --resume [checkpoint.pkl]
```

Visualize output:

```bash
python Your_World/visualize.py results/your_world_N256_seed42.npz
```

---

## Notes

- The repository currently includes executable source for both simulation tracks.
- `Your_World/` contains the structured, hierarchical extension and configuration-driven runtime.
- Root-level `tests/` currently covers the Quantum Baby Universe track; additional `Your_World/tests/` can be added in future iterations.
