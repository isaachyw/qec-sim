"""
Pytest test suite for the QC_claude DetectorSampler.

Tests verify:
    - Correct parsing of DETECTOR / OBSERVABLE_INCLUDE from Stim circuits
    - Shape/dtype/value-range of output arrays
    - Detection rates match Stim's compiled detector sampler (statistical)
    - Noiseless circuits produce zero detectors
    - REPEAT blocks are correctly unrolled
    - Unsupported measurement types (MX, MPP, …) are rejected
    - Functional interface (sample_detectors) works

Run from stim-playground/:
    uv run pytest QC_claude/tests/test_detector.py -n auto -v
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from QC_claude import DetectorSampler, DetectorSamplerResult, sample_detectors
from QC_claude.detector import _parse_detectors_and_observables, _evaluate_xor


# ── Helpers ──────────────────────────────────────────────────────────────────


def _simple_detector_circuit() -> stim.Circuit:
    """
    Minimal 2-qubit circuit with 1 detector and 1 observable.

        H 0
        CX 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]

    Noiseless Bell state: both measurements agree → detector always 0.
    Observable = qubit 1 measurement.
    """
    return stim.Circuit("""
        H 0
        CX 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)


# ── Parsing tests ────────────────────────────────────────────────────────────


def test_parse_simple_circuit():
    """Parse a simple circuit with 1 detector and 1 observable."""
    sc = _simple_detector_circuit()
    det_records, obs_records, total_meas = _parse_detectors_and_observables(sc)

    assert total_meas == 2
    assert len(det_records) == 1
    # DETECTOR rec[-1] rec[-2] → abs indices [1, 0] (meas_count=2 at DETECTOR)
    assert sorted(det_records[0]) == [0, 1]
    assert 0 in obs_records
    assert obs_records[0] == [1]  # rec[-1] → abs index 1


def test_parse_repetition_code():
    """Parse a generated repetition code with REPEAT blocks."""
    sc = stim.Circuit.generated(
        'repetition_code:memory',
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.01,
    )
    det_records, obs_records, total_meas = _parse_detectors_and_observables(sc)

    assert len(det_records) == sc.num_detectors
    assert len(obs_records) == sc.num_observables
    # All detector indices must be valid (in range [0, total_meas))
    for rec in det_records:
        for idx in rec:
            assert 0 <= idx < total_meas, f"Index {idx} out of range [0, {total_meas})"


def test_parse_multiple_observables():
    """Multiple OBSERVABLE_INCLUDE with different indices are tracked separately."""
    sc = stim.Circuit("""
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """)
    det_records, obs_records, total_meas = _parse_detectors_and_observables(sc)

    assert total_meas == 3
    assert len(det_records) == 0
    # obs 0: rec[-1] (abs=2) and rec[-3] (abs=0) — accumulated
    assert sorted(obs_records[0]) == [0, 2]
    # obs 1: rec[-2] (abs=1)
    assert obs_records[1] == [1]


def test_parse_unsupported_measurement_raises():
    """Circuits with MPP should raise ValueError (still unsupported)."""
    sc = stim.Circuit("""
        H 0
        CX 0 1
        MPP X0*Z1
        DETECTOR rec[-1]
    """)
    with pytest.raises(ValueError, match="MPP"):
        _parse_detectors_and_observables(sc)


@pytest.mark.parametrize("instr", ["MPP X0*Z1"])
def test_parse_unsupported_variants(instr: str):
    """Unsupported measurement instructions (MPP) are rejected."""
    sc = stim.Circuit(f"""
        H 0
        CX 0 1
        {instr}
    """)
    with pytest.raises(ValueError):
        _parse_detectors_and_observables(sc)


# ── XOR evaluation tests ────────────────────────────────────────────────────


def test_evaluate_xor_basic():
    """XOR of specified columns produces correct results."""
    meas = np.array([
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
    ], dtype=np.uint8)

    record_lists = [
        [0, 1],     # XOR col 0 and col 1
        [2, 3],     # XOR col 2 and col 3
        [0, 1, 2],  # XOR col 0, 1, 2
    ]
    result = _evaluate_xor(meas, record_lists)

    assert result.shape == (4, 3)
    assert result.dtype == np.uint8
    # Row 0: [0^1, 1^0, 0^1^1] = [1, 1, 0]
    np.testing.assert_array_equal(result[0], [1, 1, 0])
    # Row 1: [1^1, 0^1, 1^1^0] = [0, 1, 0]
    np.testing.assert_array_equal(result[1], [0, 1, 0])
    # Row 2: [0^0, 0^0, 0^0^0] = [0, 0, 0]
    np.testing.assert_array_equal(result[2], [0, 0, 0])
    # Row 3: [1^0, 1^1, 1^0^1] = [1, 0, 0]
    np.testing.assert_array_equal(result[3], [1, 0, 0])


def test_evaluate_xor_empty_record():
    """Empty record list → all zeros."""
    meas = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = _evaluate_xor(meas, [[]])
    assert result.shape == (2, 1)
    np.testing.assert_array_equal(result, [[0], [0]])


# ── DetectorSampler integration tests ────────────────────────────────────────


def test_detector_sampler_noiseless_bell():
    """
    Noiseless Bell state: detector = XOR of two measurements.
    Both measurements always agree → detector always 0.
    Observable = last measurement (either 0 or 1 with equal probability).
    """
    sc = _simple_detector_circuit()
    ds = DetectorSampler(sc)

    assert ds.n_detectors == 1
    assert ds.n_observables == 1
    assert abs(ds.one_norm - 1.0) < 1e-9

    result = ds.sample(n_samples=1_000, seed=0)

    assert isinstance(result, DetectorSamplerResult)
    assert result.detectors.shape == (1_000, 1)
    assert result.observables.shape == (1_000, 1)
    assert result.detectors.dtype == np.uint8
    assert result.observables.dtype == np.uint8
    assert result.weights.shape == (1_000,)

    # Noiseless: detector should be 0 for every sample
    assert np.all(result.detectors == 0), "Noiseless Bell should have 0 detectors firing"
    # All weights should be 1.0 (no RZ gates)
    assert np.allclose(result.weights, 1.0)


def test_detector_sampler_repetition_code():
    """
    Repetition code with depolarizing noise: detection rates should match
    Stim's compiled detector sampler within statistical tolerance.
    """
    sc = stim.Circuit.generated(
        'repetition_code:memory',
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.01,
    )
    ds = DetectorSampler(sc)
    result = ds.sample(n_samples=10_000, seed=42)

    # Shape checks
    assert result.detectors.shape == (10_000, sc.num_detectors)
    assert result.observables.shape == (10_000, sc.num_observables)
    assert result.n_detectors == sc.num_detectors
    assert result.n_observables == sc.num_observables
    assert np.allclose(result.weights, 1.0)
    assert abs(result.one_norm - 1.0) < 1e-9

    # Compare detection rates with Stim's reference (large sample for tighter bound)
    stim_det = sc.compile_detector_sampler().sample(shots=100_000)
    our_rate = result.detectors.astype(float).mean(axis=0)
    stim_rate = stim_det.astype(float).mean(axis=0)
    np.testing.assert_allclose(our_rate, stim_rate, atol=0.02)


def test_detector_sampler_surface_code():
    """
    Surface code circuit: validates shape and detector output on a
    more complex circuit with many detectors.
    """
    sc = stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        rounds=2,
        distance=3,
        after_clifford_depolarization=0.005,
    )
    ds = DetectorSampler(sc)
    result = ds.sample(n_samples=2_000, seed=99)

    assert result.detectors.shape == (2_000, sc.num_detectors)
    assert result.observables.shape == (2_000, sc.num_observables)
    assert np.all((result.detectors == 0) | (result.detectors == 1))
    assert np.all((result.observables == 0) | (result.observables == 1))
    assert np.allclose(result.weights, 1.0)


def test_detector_sampler_no_detectors():
    """Circuit with measurements but no detectors → 0-column detector array."""
    sc = stim.Circuit("""
        H 0
        M 0
    """)
    ds = DetectorSampler(sc)
    assert ds.n_detectors == 0
    assert ds.n_observables == 0

    result = ds.sample(n_samples=100, seed=0)
    assert result.detectors.shape == (100, 0)
    assert result.observables.shape == (100, 0)


def test_detector_sampler_mr_measurements():
    """
    Circuit with MR (measure-reset): 2 rounds of syndrome extraction.
    Detectors compare syndromes between rounds.
    """
    sc = stim.Circuit("""
        H 0
        CX 0 1
        MR 1
        DETECTOR rec[-1]
        H 0
        CX 0 1
        MR 1
        DETECTOR rec[-1] rec[-2]
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    ds = DetectorSampler(sc)
    assert ds.n_detectors == 2
    assert ds.n_observables == 1

    result = ds.sample(n_samples=500, seed=7)
    assert result.detectors.shape == (500, 2)
    assert result.observables.shape == (500, 1)


# ── Functional interface ─────────────────────────────────────────────────────


def test_sample_detectors_convenience():
    """sample_detectors() convenience function matches class-based usage."""
    sc = _simple_detector_circuit()
    result = sample_detectors(sc, n_samples=200, seed=0)

    assert isinstance(result, DetectorSamplerResult)
    assert result.n_samples == 200
    assert result.n_detectors == 1
    assert result.n_observables == 1
    # Noiseless Bell: detector always 0
    assert np.all(result.detectors == 0)
