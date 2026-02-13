"""
AUDIT TEST SUITE: MTI-EVO CORE HARDENING
=======================================
Covers deterministic behavior, shape consistency, eviction invariants,
and learning discrimination after curiosity-bias initialization.
"""

import hashlib
import pathlib
import sys

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mti_evo.core.config import MTIConfig
from mti_evo.core.lattice import HolographicLattice


def _build_lattice(**kwargs):
    cfg = MTIConfig(
        capacity_limit=kwargs.pop("capacity_limit", 32),
        grace_period=kwargs.pop("grace_period", 0),
        telemetry_enabled=kwargs.pop("telemetry_enabled", False),
        random_seed=kwargs.pop("random_seed", 1337),
        deterministic=kwargs.pop("deterministic", True),
        eviction_mode=kwargs.pop("eviction_mode", "deterministic_sample"),
        eviction_sample_size=kwargs.pop("eviction_sample_size", 50),
        stimulate_return_metrics=kwargs.pop("stimulate_return_metrics", False),
        **kwargs,
    )
    return HolographicLattice(config=cfg)


def _weights_checksum(lattice: HolographicLattice) -> str:
    digest = hashlib.sha256()
    for seed in sorted(lattice.active_tissue.keys()):
        neuron = lattice.active_tissue[seed]
        digest.update(np.asarray(neuron.weights, dtype=np.float64).tobytes())
        digest.update(np.asarray([float(neuron.bias)], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _run_fixed_script(seed=1337):
    lattice = _build_lattice(random_seed=seed, capacity_limit=128, grace_period=0)
    script = [
        ([11], np.array([1.0, 0.0, 1.0]), True),
        ([12], np.array([0.0, 1.0, 0.0]), True),
        ([11, 12], np.array([1.0, 0.0, 1.0]), True),
        ([13], np.array([0.2, 0.4, 0.6]), True),
        ([11], np.array([1.0, 0.0, 1.0]), False),
        ([12, 13], np.array([0.0, 1.0, 0.0]), True),
    ]
    for seeds, signal, learn in script:
        lattice.stimulate(seeds, signal, learn=learn)
    return lattice


def test_determinism_replay_identical_states():
    run_a = _run_fixed_script(seed=1337)
    run_b = _run_fixed_script(seed=1337)

    assert sorted(run_a.active_tissue.keys()) == sorted(run_b.active_tissue.keys())
    assert _weights_checksum(run_a) == _weights_checksum(run_b)

    for seed in sorted(run_a.active_tissue.keys()):
        neuron_a = run_a.active_tissue[seed]
        neuron_b = run_b.active_tissue[seed]
        assert np.array_equal(neuron_a.weights, neuron_b.weights)
        assert neuron_a.bias == neuron_b.bias


def test_adapt_consistent_for_1d_and_2d_inputs():
    seed = 404
    signal_1d = np.array([1.0, 0.0, 1.0])
    signal_2d = np.array([[1.0, 0.0, 1.0]])

    lattice_a = _build_lattice(random_seed=2026)
    lattice_b = _build_lattice(random_seed=2026)

    lattice_a.stimulate([seed], signal_1d, learn=True)
    lattice_b.stimulate([seed], signal_2d, learn=True)

    neuron_a = lattice_a.active_tissue[seed]
    neuron_b = lattice_b.active_tissue[seed]

    assert np.array_equal(neuron_a.weights, neuron_b.weights)
    assert np.array_equal(neuron_a.velocity, neuron_b.velocity)
    assert neuron_a.bias == neuron_b.bias


def test_discrimination_uses_margin_not_brittle_absolute_weight():
    lattice = _build_lattice(capacity_limit=10)
    pattern_a = np.array([1.0, 0.0, 1.0])
    pattern_b = np.array([-1.0, 1.0, -1.0])

    pre = lattice.stimulate([1000], pattern_a, learn=False)
    for _ in range(5):
        lattice.stimulate([1000], pattern_a, learn=True)

    post = lattice.stimulate([1000], pattern_a, learn=False)
    resp_b = lattice.stimulate([1000], pattern_b, learn=False)

    assert post > pre
    assert post - resp_b >= 0.4
    assert post >= 0.7


def test_shape_evolution_preserves_learning_and_state_vectors():
    lattice = _build_lattice(capacity_limit=10)
    seed = 77

    pattern_3 = np.array([1.0, 0.0, 1.0])
    pattern_5 = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

    lattice.stimulate([seed], pattern_3, learn=False)

    pre = lattice.stimulate([seed], pattern_5, learn=False)
    lattice.stimulate([seed], pattern_5, learn=True)
    post = lattice.stimulate([seed], pattern_5, learn=False)

    neuron = lattice.active_tissue[seed]
    assert len(neuron.weights) == 5
    assert len(neuron.velocity) == 5
    assert post > pre


def test_eviction_invariants_keep_strong_memories():
    capacity = 10
    lattice = _build_lattice(capacity_limit=capacity, grace_period=0, eviction_sample_size=capacity)

    for seed in range(1, 6):
        for _ in range(10):
            lattice.stimulate([seed], 0.9, learn=True)

    for seed in range(100, 200):
        lattice.stimulate([seed], 0.5, learn=True)

    survivors = set(lattice.active_tissue.keys())
    strong_seeds = set(range(1, 6))
    assert strong_seeds.issubset(survivors)

    strong_avg_weight = float(
        np.mean([np.mean(lattice.active_tissue[s].weights) for s in sorted(strong_seeds)])
    )
    noise_survivors = [s for s in survivors if s not in strong_seeds]
    noise_avg_weight = float(
        np.mean([np.mean(lattice.active_tissue[s].weights) for s in noise_survivors])
    )

    assert strong_avg_weight > noise_avg_weight
    assert len(lattice.active_tissue) == capacity


def test_eviction_modes_respect_capacity():
    for mode in ["full_scan", "sample", "deterministic_sample"]:
        lattice = _build_lattice(capacity_limit=8, grace_period=0, eviction_mode=mode, eviction_sample_size=8)
        for seed in range(20):
            lattice.stimulate([seed], np.array([0.25, 0.75]), learn=True)
        assert len(lattice.active_tissue) == 8


def test_stimulate_metrics_output_is_optional_and_backward_compatible():
    lattice_default = _build_lattice(stimulate_return_metrics=False)
    out_default = lattice_default.stimulate([1], np.array([0.9]), learn=True)
    assert isinstance(out_default, float)

    lattice_metrics = _build_lattice(stimulate_return_metrics=True)
    out_metrics = lattice_metrics.stimulate([1], np.array([0.9]), learn=True)
    assert isinstance(out_metrics, dict)
    assert "avg_resonance_pre" in out_metrics
    assert "avg_resonance_post" in out_metrics
    assert "active_count" in out_metrics
    assert "evictions" in out_metrics
    assert "latency_ms" in out_metrics


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
