"""
Monte Carlo sampler for quantum circuits with measurements.

Unlike MCEstimator (which computes <O> by Rao-Blackwellising over an observable),
MCSampler returns raw measurement bitstrings and quasiprobability weights.

This is the right primitive for memory experiments:
    1. Run MCSampler.sample(n_samples) → SamplerResult.
    2. Feed SamplerResult.measurements (shape N×M) into a decoder.
    3. Weight each sample by SamplerResult.weights to compute the weighted
       logical error rate.

Output shapes:
    measurements : (N, M)  uint8    — 0/1 bit per measurement per sample,
                                      ordered by MeasureOp appearance in the circuit.
    weights      : (N,)    float64  — real quasiprobability weight per sample.
                                      Pure Clifford/noise circuits: all weights = 1.0.
                                      RZ gates: weights may be negative (quasi-probability).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import stim

from .circuit import (
    Circuit,
    CliffordOp,
    RZOp,
    PauliNoiseOp,
    TwoQubitPauliNoiseOp,
    DampOp,
    MeasureOp,
    TickOp,
)
from .decompositions import (
    GateDecomposition,
    rz_decomposition,
    pauli_noise_decomp_1q,
    pauli_noise_decomp_2q,
    damp_decomposition,
)


# ── Stim instruction string helpers ──────────────────────────────────────────

# Map our canonical Clifford names → stim instruction names
_CLIFFORD_TO_STIM: dict[str, str] = {"Sdg": "S_DAG"}


def _gate_term_to_stim_str(gate: str, qubits: tuple[int, ...]) -> str:
    """Convert a decomposition gate term (e.g. 'X', 'RESET', 'XZ') to a stim instruction string."""
    if len(qubits) == 1:
        q = qubits[0]
        if gate == "I":
            return ""
        elif gate == "RESET":
            return f"R {q}"
        else:  # X, Y, Z, S
            return f"{gate} {q}"
    else:
        # 2Q Pauli term: e.g. "XZ" on (q0, q1) → "X q0\nZ q1"
        parts = []
        for pauli, q in zip(gate, qubits):
            if pauli != "I":
                parts.append(f"{pauli} {q}")
        return "\n".join(parts)


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class SamplerResult:
    """
    Output of MCSampler.sample().

    Attributes:
        measurements:   uint8 array of shape (n_samples, n_measurements).
                        Each row is one sample; each column is one measurement bit,
                        ordered by the appearance of MeasureOp in the circuit.
        weights:        float64 array of shape (n_samples,).
                        Real quasiprobability weight for each sample.
                        For circuits with only Clifford gates and noise (no RZ):
                        all weights equal 1.0.
        n_samples:      Number of Monte Carlo samples.
        n_measurements: Total number of measurement bits per sample.
        one_norm:       Product of all 1-norms from RZ decompositions.
                        Equal to 1.0 for Clifford+noise only circuits.
    """

    measurements: np.ndarray  # (n_samples, n_measurements), dtype uint8
    weights: np.ndarray  # (n_samples,), dtype float64
    n_samples: int
    n_measurements: int
    one_norm: float

    def __repr__(self) -> str:
        return (
            f"SamplerResult(n_samples={self.n_samples}, "
            f"n_measurements={self.n_measurements}, "
            f"one_norm={self.one_norm:.4f}, "
            f"measurements.shape={self.measurements.shape}, "
            f"weights.shape={self.weights.shape})"
        )


# ── Module-level worker (must be at module level for pickle) ──────────────────


def _proc_worker(
    sampler: "MCSampler", child_seed: np.random.SeedSequence, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Worker function for ProcessPoolExecutor.

    Must be defined at module level so multiprocessing can pickle it by name.
    child_seed is a spawned SeedSequence (unique per worker via spawn_key).
    Uses batch sampling for efficiency.
    """
    rng = np.random.default_rng(child_seed)
    return sampler._batch_sample(n, rng)


# ── MCSampler ─────────────────────────────────────────────────────────────────


class MCSampler:
    """
    Monte Carlo sampler for circuits containing measurements.

    Each call to sample() runs n_samples independent trajectories through the
    circuit. For each trajectory:
        - Clifford gates are applied exactly.
        - RZ(theta) gates are replaced by a sampled Clifford term from the
          3-term quasiprobability decomposition; the ratio coeff/prob is
          accumulated into the sample weight.
        - Pauli noise ops are replaced by a sampled Pauli term (all weights
          remain 1.0 because noise 1-norms are 1).
        - MeasureOps call sim.measure_many() and collect bit outcomes.

    Example::

        sc = stim.Circuit(\"\"\"
            H 0
            CX 0 1
            DEPOLARIZE1(0.01) 0 1
            MR 0 1
        \"\"\")
        c = Circuit.from_stim(sc)
        sampler = MCSampler(c)
        result = sampler.sample(n_samples=10_000, seed=0)
        # result.measurements: shape (10_000, 2)
        # result.weights: shape (10_000,)  — all 1.0 (no RZ)
    """

    def __init__(self, circuit: Circuit) -> None:
        if circuit.n_measurements == 0:
            raise ValueError(
                "MCSampler requires a circuit with at least one MeasureOp. "
                "Use circuit.measure(*qubits) or circuit.measure_reset(*qubits) "
                "to add measurements, or Circuit.from_stim() with M/MR instructions."
            )

        self._circuit = circuit
        self._n_measurements = circuit.n_measurements

        # Build op plan: list of (op, decomps | None)
        #   CliffordOp / MeasureOp → None  (applied directly)
        #   TickOp                 → skipped
        #   RZOp / PauliNoiseOp / DampOp → list of per-qubit GateDecompositions
        #   TwoQubitPauliNoiseOp → single-element list[GateDecomposition]
        self._op_plan: list[tuple[object, list[GateDecomposition] | None]] = []
        for op in circuit.ops:
            if isinstance(op, (CliffordOp, MeasureOp)):
                self._op_plan.append((op, None))
            elif isinstance(op, RZOp):
                self._op_plan.append(
                    (op, [rz_decomposition(q, op.theta) for q in op.qubits])
                )
            elif isinstance(op, PauliNoiseOp):
                self._op_plan.append(
                    (
                        op,
                        [
                            pauli_noise_decomp_1q(q, op.px, op.py, op.pz)
                            for q in op.qubits
                        ],
                    )
                )
            elif isinstance(op, TwoQubitPauliNoiseOp):
                self._op_plan.append(
                    (op, [pauli_noise_decomp_2q(op.q0, op.q1, op.probs)])
                )
            elif isinstance(op, DampOp):
                self._op_plan.append(
                    (op, [damp_decomposition(q, op.t, op.T1, op.T2) for q in op.qubits])
                )
            elif isinstance(op, TickOp):
                pass  # tick markers have no simulation effect
            else:
                raise TypeError(f"Unknown op type: {type(op)}")

        # ── Pre-compute flat decomposition arrays for batch sampling ──────
        # Flatten all GateDecompositions into parallel arrays so that we can
        # draw ALL random numbers in one rng.random() call and vectorize the
        # weight computation across samples.
        #
        # _flat_qubits[j]:  target qubit tuple for decomposition j
        # _flat_gates[j]:   list of gate labels (one per term)
        # _flat_cdfs[j]:    CDF array for importance sampling
        # _flat_coeffs[j]:  complex coefficient array (for weight = coeff/prob)
        # _flat_probs[j]:   sampling probability array
        self._flat_qubits: list[tuple[int, ...]] = []
        self._flat_gates: list[list[str]] = []
        self._flat_cdfs: list[np.ndarray] = []
        self._flat_coeffs: list[np.ndarray] = []
        self._flat_probs: list[np.ndarray] = []

        for _, decomps in self._op_plan:
            if decomps is not None:
                for decomp in decomps:
                    abs_c = np.array([abs(t.coefficient) for t in decomp.terms])
                    norm = abs_c.sum()
                    probs = abs_c / norm
                    self._flat_qubits.append(decomp.qubits)
                    self._flat_gates.append([t.gate for t in decomp.terms])
                    self._flat_cdfs.append(np.cumsum(probs))
                    self._flat_coeffs.append(
                        np.array([t.coefficient for t in decomp.terms])
                    )
                    self._flat_probs.append(probs)

        self._n_flat = len(self._flat_qubits)

        # ── Pre-build stim instruction strings for Circuit-per-sample ─────
        # _circuit_template is a flat list matching the op_plan order.
        # Fixed ops (Clifford/Measure) → str instruction line.
        # Noise/RZ decomps → list[str] (one instruction line per term).
        # In the per-sample loop we select the right term string, join all
        # lines, parse one stim.Circuit, and call sim.do_circuit() once —
        # collapsing ~N individual pybind11 crossings into ~3.
        self._circuit_template: list[str | list[str]] = []
        for op, decomps in self._op_plan:
            if isinstance(op, CliffordOp):
                if op.name == "I":
                    continue
                name = _CLIFFORD_TO_STIM.get(op.name, op.name)
                targets = " ".join(str(t) for t in op.targets)
                self._circuit_template.append(f"{name} {targets}")
            elif isinstance(op, MeasureOp):
                basis_suffix = "" if op.basis == "Z" else op.basis
                mn = ("MR" if op.reset else "M") + basis_suffix
                targets = " ".join(str(q) for q in op.qubits)
                if op.flip_probability > 0:
                    self._circuit_template.append(
                        f"{mn}({op.flip_probability}) {targets}"
                    )
                else:
                    self._circuit_template.append(f"{mn} {targets}")
            elif decomps is not None:
                for decomp in decomps:
                    term_strs = [
                        _gate_term_to_stim_str(term.gate, decomp.qubits)
                        for term in decomp.terms
                    ]
                    self._circuit_template.append(term_strs)

    @property
    def one_norm(self) -> float:
        """Product of 1-norms for all decompositions (noise ops contribute 1.0)."""
        result = 1.0
        for _, decomps in self._op_plan:
            if decomps is not None:
                for decomp in decomps:
                    result *= decomp.one_norm
        return result

    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
        n_workers: int | None = None,
    ) -> SamplerResult:
        """
        Draw n_samples measurement trajectories.

        Args:
            n_samples:  Number of Monte Carlo samples.
            seed:       Optional integer seed for reproducibility.
            n_workers:  Number of parallel worker processes.
                        None or 1 → serial (default).
                        -1 → one process per logical CPU (os.cpu_count()).
                        N → N processes.

        Returns:
            SamplerResult with measurements (N×M) and weights (N,).
        """
        import os

        if n_workers == -1:
            n_workers = os.cpu_count()

        if n_workers is None or n_workers <= 1:
            # ── Serial path ───────────────────────────────────────────────────
            rng = np.random.default_rng(seed)
            measurements, weights = self._batch_sample(n_samples, rng)
        else:
            # ── Parallel path (ProcessPoolExecutor) ───────────────────────────
            base, rem = divmod(n_samples, n_workers)
            chunk_sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]
            child_seeds = np.random.SeedSequence(seed).spawn(n_workers)

            chunks_m: list[np.ndarray] = []
            chunks_w: list[np.ndarray] = []
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(_proc_worker, self, s, n)
                    for s, n in zip(child_seeds, chunk_sizes)
                ]
                for f in futures:
                    m, w = f.result()
                    chunks_m.append(m)
                    chunks_w.append(w)
            measurements = np.concatenate(chunks_m, axis=0)
            weights = np.concatenate(chunks_w, axis=0)

        return SamplerResult(
            measurements=measurements,
            weights=weights,
            n_samples=n_samples,
            n_measurements=self._n_measurements,
            one_norm=self.one_norm,
        )

    # ── Batch sampling core ──────────────────────────────────────────────────

    def _batch_sample(
        self, n_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw n_samples trajectories using batch-vectorized noise sampling.

        Instead of calling GateDecomposition.sample(rng) per-decomp per-sample
        (slow: ~22 us Python overhead each), we:
          1. Draw ALL random numbers in one rng.random() call.
          2. Convert to term indices via searchsorted on pre-computed CDFs.
          3. Vectorize the weight computation across all samples.
          4. Per-sample loop only does cheap gate lookups + Stim C++ calls.

        Returns:
            (measurements, weights) — measurements is (N, M) uint8,
            weights is (N,) float64.
        """
        N = self._n_flat
        stim_seeds = rng.integers(
            0, np.iinfo(np.int64).max, size=n_samples, dtype=np.int64
        )

        # ── Step 1: batch-draw term indices for all decomps × all samples ─────
        # term_indices[i, j] = which Clifford term was chosen for sample i,
        #                      flat decomposition j.
        if N > 0:
            term_indices = np.empty((n_samples, N), dtype=np.int8)
            for j in range(N):
                u = rng.random(n_samples)
                term_indices[:, j] = np.searchsorted(self._flat_cdfs[j], u)

            # ── Step 2: vectorized weight computation ─────────────────────────
            # weight[i] = prod_j ( coeff[j][idx] / prob[j][idx] )
            weights = np.ones(n_samples, dtype=np.complex128)
            for j in range(N):
                idx_j = term_indices[:, j]
                weights *= self._flat_coeffs[j][idx_j] / self._flat_probs[j][idx_j]
            weights_real = weights.real
        else:
            term_indices = np.empty((n_samples, 0), dtype=np.int8)
            weights_real = np.ones(n_samples, dtype=np.float64)

        # ── Step 3: per-sample simulation via stim.Circuit batching ──────────
        # Instead of ~N individual sim.h()/sim.x()/… pybind11 calls per sample,
        # we build one stim.Circuit string per sample and call sim.do_circuit()
        # once — collapsing ~N crossings into ~3 (Circuit parse + do_circuit +
        # current_measurement_record).
        measurements = np.empty((n_samples, self._n_measurements), dtype=np.uint8)
        template = self._circuit_template

        for i in range(n_samples):
            sim = stim.TableauSimulator(seed=int(stim_seeds[i]))
            lines: list[str] = []
            flat_idx = 0

            for entry in template:
                if isinstance(entry, list):
                    line = entry[term_indices[i, flat_idx]]
                    if line:
                        lines.append(line)
                    flat_idx += 1
                else:
                    lines.append(entry)

            sim.do_circuit(stim.Circuit("\n".join(lines)))
            measurements[i, :] = sim.current_measurement_record()

        return measurements, weights_real


# ── Functional interface ───────────────────────────────────────────────────────


def sample(
    circuit: Circuit,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> SamplerResult:
    """
    Convenience wrapper: MCSampler(circuit).sample(n_samples, seed).

    Args:
        circuit:   Circuit with at least one MeasureOp.
        n_samples: Number of Monte Carlo samples.
        seed:      Optional integer seed.

    Returns:
        SamplerResult with measurements (N×M) and weights (N,).
    """
    return MCSampler(circuit).sample(n_samples=n_samples, seed=seed)
