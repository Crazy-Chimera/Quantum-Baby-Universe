"""
Tests for Quantum Baby Universe.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from baby_universe_v1 import BabyUniverseLoop
from baby_universe_v2 import BabyUniverseV2
from baby_universe_v3 import BabyUniverseV3, compute_step_vectorized
from fabric_integration import FabricAcceleratedUniverse, CCP_TILE


def test_v1_initialization():
    universe = BabyUniverseLoop(N=32, K_init=0.9, seed=42)
    assert universe.state['grid'].shape == (32, 32)
    assert abs(np.mean(universe.state['grid']) - 0.9) < 0.05


def test_v1_compute_step():
    universe = BabyUniverseLoop(N=32, K_init=0.9, seed=42)
    metrics = universe.observe()
    universe.control(metrics, {}, {'threshold': 0.8, 'noise': 0.01})
    assert universe.iteration == 1
    assert len(universe.state['phi_history']) == 1


def test_v2_recursion_detection():
    universe = BabyUniverseV2(N=64, K_init=0.9, seed=42)
    r1, r2, r3 = universe.detect_higher_recursion()
    assert isinstance(r1, bool)
    assert isinstance(r2, bool)
    assert isinstance(r3, bool)


def test_v3_vectorized_step():
    grid = np.random.choice([0, 1], size=(128, 128), p=[0.1, 0.9]).astype(np.float32)
    new_grid = compute_step_vectorized(grid, 0.8, 0.01)
    assert new_grid.shape == (128, 128)
    assert np.all((new_grid == 0) | (new_grid == 1))


def test_v3_recursion_detection():
    universe = BabyUniverseV3(N=128, K_init=0.9, seed=42)
    r1, r2, r3, r4 = universe.detect_higher_recursion()
    assert isinstance(r1, bool)
    assert isinstance(r2, bool)
    assert isinstance(r3, bool)
    assert isinstance(r4, bool)


def test_fabric_compute():
    universe = FabricAcceleratedUniverse(N=32, seed=42, tile=CCP_TILE)
    result = universe.compute_step(0.8, 0.01)
    assert 'latency' in result
    assert 'energy' in result
    assert 'elegance' in result
    assert result['energy'] > 0
