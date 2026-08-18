# Quantum Baby Universe

A laboratory test of bootstrap cosmology. Simulation of the entangled field Φ
on a discrete lattice of informions, which spontaneously generates emergent time,
particles, and recursive self‑modelling R=1 to R=4.

## Installation

```bash
pip install numpy scipy matplotlib pytest
```

Running

```bash
make run-v1      # basic simulation 32×32
make run-v2      # lattice 64×64, detection R=1 to R=3
make run-v3      # lattice 128×128, detection R=1 to R=4, vectorised
make run-fabric  # Φ‑Fabric simulation
make test        # tests
```

Principle

Each cell of the lattice is an informion – a bit of the entangled field Φ.
The Compute operator applies a local rule:

· if the local Φ is below the threshold, the cell is repaired to 1,
· otherwise, with a small probability, it is randomly flipped to 0 (noise).

The system tends towards Substrate* (K=1, C=0) by minimising elegance C/K.

Versions

· V1: original monolithic simulation, now as a LoopObject.
· V2: scaling to 64×64, detection of higher recursion.
· V3: fully vectorised Compute step for 128×128, detection R=1 to R=4.
· Fabric: run on Φ‑Fabric CCP tile with latency and energy measurement.

Convergence

The file convergence.py controls asymptotic convergence: the simulation
iterates until elegance drops below the threshold 0.001 or improvement
stops after 100 generations.

Φ
