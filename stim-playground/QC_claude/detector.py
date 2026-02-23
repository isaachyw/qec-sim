"""
Detector sampler: converts MCSampler measurement bitstrings into
Stim-style detector and observable results.

Given a stim.Circuit with DETECTOR and OBSERVABLE_INCLUDE instructions,
parses the measurement-record references (`rec[-k]`), runs MCSampler,
and post-processes the raw measurement bits into detector/observable XOR values.

This enables feeding QC_claude MC samples (including quasiprobability weights
from non-Clifford RZ gates) directly into a decoder (e.g. PyMatching).

Detector semantics:
    A DETECTOR fires (value = 1) when the XOR of its referenced measurement
    records is 1. The rec[-k] syntax is relative to the total measurement
    count at the point where the DETECTOR instruction appears in the circuit.

Observable semantics:
    OBSERVABLE_INCLUDE(n) accumulates measurement record references into
    logical observable n. Multiple OBSERVABLE_INCLUDE(n) with the same n
    are merged (their record lists are concatenated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from .circuit import Circuit
from .sampler import MCSampler


# ── Measurement instructions ─────────────────────────────────────────────────

_SUPPORTED_MEASUREMENT: frozenset[str] = frozenset({
    'M', 'MZ', 'MR', 'MRZ', 'MX', 'MY', 'MRX', 'MRY',
})

_UNSUPPORTED_MEASUREMENT: frozenset[str] = frozenset({
    'MPP', 'MPAD', 'HERALDED_ERASE', 'HERALDED_PAULI_CHANNEL_1',
})


# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class DetectorSamplerResult:
    """
    Output of DetectorSampler.sample().

    Attributes:
        detectors:      uint8 array of shape (n_samples, n_detectors).
                        Each entry is 0 or 1: the XOR of the measurement
                        records that define that detector.
        observables:    uint8 array of shape (n_samples, n_observables).
                        Each entry is 0 or 1: the XOR of the measurement
                        records that define that observable.
        weights:        float64 array of shape (n_samples,).
                        Real quasiprobability weight for each sample
                        (passthrough from SamplerResult).
        n_samples:      Number of Monte Carlo samples.
        n_detectors:    Number of detectors in the circuit.
        n_observables:  Number of observables in the circuit.
        one_norm:       Product of all 1-norms (passthrough from SamplerResult).
    """

    detectors: np.ndarray       # (n_samples, n_detectors), dtype uint8
    observables: np.ndarray     # (n_samples, n_observables), dtype uint8
    weights: np.ndarray         # (n_samples,), dtype float64
    n_samples: int
    n_detectors: int
    n_observables: int
    one_norm: float

    def __repr__(self) -> str:
        return (
            f"DetectorSamplerResult(n_samples={self.n_samples}, "
            f"n_detectors={self.n_detectors}, "
            f"n_observables={self.n_observables}, "
            f"one_norm={self.one_norm:.4f})"
        )


# ── Parsing helper ────────────────────────────────────────────────────────────


def _parse_detectors_and_observables(
    stim_circuit: stim.Circuit,
) -> tuple[list[list[int]], dict[int, list[int]], int]:
    """
    Parse a stim.Circuit to extract detector and observable definitions.

    Walks the flattened (REPEAT-unrolled) circuit, tracks the running
    measurement count, and resolves each ``rec[-k]`` target to an absolute
    measurement column index.

    Args:
        stim_circuit: The original Stim circuit (will be flattened internally).

    Returns:
        (detector_records, observable_records, total_measurements)

        - detector_records[i]: list of absolute measurement indices for detector i.
        - observable_records[n]: list of absolute measurement indices for
          observable n (accumulated across all OBSERVABLE_INCLUDE(n) instructions).
        - total_measurements: total number of measurement records in the circuit.

    Raises:
        ValueError: If unsupported measurement instructions (MX, MY, MPP, etc.)
                    are found — these would cause index misalignment with
                    MCSampler output.
    """
    flat = stim_circuit.flattened()

    meas_count = 0
    detector_records: list[list[int]] = []
    observable_records: dict[int, list[int]] = {}

    for instr in flat:
        name: str = instr.name

        # ── Supported measurements: count qubit targets ───────────────────
        if name in _SUPPORTED_MEASUREMENT:
            targets = instr.targets_copy()
            meas_count += sum(1 for t in targets if t.is_qubit_target)

        # ── Unsupported measurements: error out ──────────────────────────
        elif name in _UNSUPPORTED_MEASUREMENT:
            raise ValueError(
                f"DetectorSampler does not support '{name}' instructions. "
                f"Only M, MZ, MR, MRZ are supported (matching MCSampler)."
            )

        # ── DETECTOR: resolve rec[-k] to absolute indices ─────────────────
        elif name == 'DETECTOR':
            abs_indices = []
            for t in instr.targets_copy():
                if t.is_measurement_record_target:
                    # t.value is negative (e.g. -1, -2)
                    abs_indices.append(meas_count + t.value)
            detector_records.append(abs_indices)

        # ── OBSERVABLE_INCLUDE: accumulate into dict ──────────────────────
        elif name == 'OBSERVABLE_INCLUDE':
            obs_idx = int(instr.gate_args_copy()[0])
            abs_indices = []
            for t in instr.targets_copy():
                if t.is_measurement_record_target:
                    abs_indices.append(meas_count + t.value)
            if obs_idx not in observable_records:
                observable_records[obs_idx] = []
            observable_records[obs_idx].extend(abs_indices)

        # ── Everything else (Cliffords, noise, TICK, …): skip ─────────────

    return detector_records, observable_records, meas_count


# ── XOR evaluation ────────────────────────────────────────────────────────────


def _evaluate_xor(
    measurements: np.ndarray,
    record_lists: list[list[int]],
) -> np.ndarray:
    """
    For each record list, XOR the corresponding measurement columns.

    Args:
        measurements: (n_samples, n_measurements) uint8 array.
        record_lists: list of K lists, where record_lists[i] contains
                      the absolute measurement indices for output column i.

    Returns:
        (n_samples, K) uint8 array where ``out[:, i] = XOR of
        measurements[:, j] for j in record_lists[i]``.
    """
    n_samples = measurements.shape[0]
    K = len(record_lists)
    out = np.zeros((n_samples, K), dtype=np.uint8)
    for i, indices in enumerate(record_lists):
        for j in indices:
            out[:, i] ^= measurements[:, j]
    return out


# ── DetectorSampler ───────────────────────────────────────────────────────────


class DetectorSampler:
    """
    Converts MCSampler measurement results into Stim-style detector
    and observable results.

    Given a stim.Circuit with DETECTOR and OBSERVABLE_INCLUDE instructions,
    this class:
        1. Parses the circuit to build detector → measurement record mappings.
        2. Builds an MCSampler from the circuit (annotations are silently skipped).
        3. On sample(), runs the MCSampler and post-processes the raw
           measurement bitstrings into detector/observable values via XOR.

    This is the right primitive for feeding QC_claude MC samples into a
    decoder (e.g. PyMatching, MWPM):
        1. ``result = DetectorSampler(sc).sample(n_samples)``
        2. ``result.detectors`` is the syndrome matrix (N × D).
        3. ``result.observables`` is the logical observable matrix (N × L).
        4. ``result.weights`` carries the quasiprobability weights.

    Example::

        import stim
        from QC_claude import DetectorSampler

        sc = stim.Circuit.generated(
            'repetition_code:memory',
            rounds=3, distance=3,
            after_clifford_depolarization=0.01,
        )
        result = DetectorSampler(sc).sample(n_samples=10_000, seed=42)
        # result.detectors.shape   == (10_000, num_detectors)
        # result.observables.shape == (10_000, num_observables)
        # result.weights — all 1.0 for Clifford+noise circuits
    """

    def __init__(self, stim_circuit: stim.Circuit) -> None:
        """
        Args:
            stim_circuit: A stim.Circuit with DETECTOR/OBSERVABLE_INCLUDE
                          instructions and supported measurement types
                          (M, MZ, MR, MRZ only).

        Raises:
            ValueError: If unsupported measurement types are present.
            ValueError: If the circuit has no measurements.
        """
        self._stim_circuit = stim_circuit

        # Parse detector/observable mappings from the original Stim circuit.
        (
            self._detector_records,
            self._observable_records,
            self._total_measurements,
        ) = _parse_detectors_and_observables(stim_circuit)

        self._n_detectors = len(self._detector_records)

        # Observable indices may be sparse (e.g. 0, 2, 5).
        # We allocate max_index + 1 columns in the output array.
        if self._observable_records:
            self._n_observables = max(self._observable_records.keys()) + 1
        else:
            self._n_observables = 0

        # Build the QC_claude Circuit and MCSampler.
        # Circuit.from_stim silently skips DETECTOR/OBSERVABLE_INCLUDE.
        qc_circuit = Circuit.from_stim(stim_circuit)
        self._sampler = MCSampler(qc_circuit)

        # Sanity check: measurement count must be consistent between
        # our parser and the Circuit built by from_stim.
        if qc_circuit.n_measurements != self._total_measurements:
            raise ValueError(
                f"Measurement count mismatch: detector parser found "
                f"{self._total_measurements} measurements but Circuit.from_stim "
                f"produced {qc_circuit.n_measurements}. "
                f"This may indicate an unsupported measurement type was "
                f"silently skipped by from_stim."
            )

    @property
    def n_detectors(self) -> int:
        """Number of DETECTOR instructions in the circuit."""
        return self._n_detectors

    @property
    def n_observables(self) -> int:
        """Number of distinct observable indices (max OBSERVABLE_INCLUDE index + 1)."""
        return self._n_observables

    @property
    def one_norm(self) -> float:
        """Product of 1-norms (passthrough from MCSampler)."""
        return self._sampler.one_norm

    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
        n_workers: int | None = None,
    ) -> DetectorSamplerResult:
        """
        Draw n_samples and compute detector/observable values.

        Args:
            n_samples:  Number of Monte Carlo samples.
            seed:       Optional integer seed for reproducibility.
            n_workers:  Parallel workers (passed through to MCSampler.sample).

        Returns:
            DetectorSamplerResult with detectors, observables, and weights.
        """
        # Step 1: Run MCSampler to get raw measurement bitstrings.
        sampler_result = self._sampler.sample(
            n_samples=n_samples, seed=seed, n_workers=n_workers,
        )

        # Step 2: Compute detector values via XOR.
        if self._n_detectors > 0:
            detectors = _evaluate_xor(
                sampler_result.measurements,
                self._detector_records,
            )
        else:
            detectors = np.zeros((n_samples, 0), dtype=np.uint8)

        # Step 3: Compute observable values via XOR.
        if self._n_observables > 0:
            obs_record_lists = [
                self._observable_records.get(i, [])
                for i in range(self._n_observables)
            ]
            observables = _evaluate_xor(
                sampler_result.measurements,
                obs_record_lists,
            )
        else:
            observables = np.zeros((n_samples, 0), dtype=np.uint8)

        return DetectorSamplerResult(
            detectors=detectors,
            observables=observables,
            weights=sampler_result.weights,
            n_samples=n_samples,
            n_detectors=self._n_detectors,
            n_observables=self._n_observables,
            one_norm=sampler_result.one_norm,
        )


# ── Functional interface ──────────────────────────────────────────────────────


def sample_detectors(
    stim_circuit: stim.Circuit,
    n_samples: int = 10_000,
    seed: int | None = None,
    n_workers: int | None = None,
) -> DetectorSamplerResult:
    """
    Convenience wrapper: ``DetectorSampler(stim_circuit).sample(...)``.

    Args:
        stim_circuit: Stim circuit with DETECTOR/OBSERVABLE_INCLUDE.
        n_samples:    Number of Monte Carlo samples.
        seed:         Optional integer seed.
        n_workers:    Parallel workers (None=serial, -1=all CPUs).

    Returns:
        DetectorSamplerResult with detectors, observables, and weights.
    """
    return DetectorSampler(stim_circuit).sample(
        n_samples=n_samples, seed=seed, n_workers=n_workers,
    )
