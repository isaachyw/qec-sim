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
from .estimator import _apply_gate_term


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
) -> tuple[list, list]:
    """
    Worker function for ProcessPoolExecutor.

    Must be defined at module level so multiprocessing can pickle it by name.
    child_seed is a spawned SeedSequence (unique per worker via spawn_key).
    """
    rng = np.random.default_rng(child_seed)
    bits_list: list[list[int]] = []
    weights_list: list[float] = []
    for _ in range(n):
        b, w = sampler._single_sample(rng)
        bits_list.append(b)
        weights_list.append(w.real)
    return bits_list, weights_list


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

        Parallelism notes:
            Each sample is fully independent so the loop is embarrassingly
            parallel. Uses ProcessPoolExecutor (separate processes) rather
            than threads because Stim's pybind11 bindings hold the GIL for
            individual gate calls, making threads ineffective. Each worker
            process receives an independent RNG stream derived from `seed`
            via numpy SeedSequence, preserving reproducibility.
            The sampler object is pickled once per worker at startup; with
            Linux fork-based multiprocessing the cost is negligible.

        Returns:
            SamplerResult with measurements (N×M) and weights (N,).
        """
        import os

        if n_workers == -1:
            n_workers = os.cpu_count()

        if n_workers is None or n_workers <= 1:
            # ── Serial path ───────────────────────────────────────────────────
            rng = np.random.default_rng(seed)
            all_bits: list[list[int]] = []
            all_weights: list[float] = []
            for _ in range(n_samples):
                bits, weight = self._single_sample(rng)
                all_bits.append(bits)
                all_weights.append(weight.real)
        else:
            # ── Parallel path (ProcessPoolExecutor) ───────────────────────────
            # Distribute samples as evenly as possible across workers.
            base, rem = divmod(n_samples, n_workers)
            chunk_sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]

            # Each worker gets an independent, reproducible RNG child stream.
            # SeedSequence objects are picklable and carry unique spawn_key state.
            child_seeds = np.random.SeedSequence(seed).spawn(n_workers)

            all_bits = []
            all_weights = []
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(_proc_worker, self, s, n)
                    for s, n in zip(child_seeds, chunk_sizes)
                ]
                for f in futures:
                    b, w = f.result()
                    all_bits.extend(b)
                    all_weights.extend(w)

        return SamplerResult(
            measurements=np.array(all_bits, dtype=np.uint8),
            weights=np.array(all_weights, dtype=np.float64),
            n_samples=n_samples,
            n_measurements=self._n_measurements,
            one_norm=self.one_norm,
        )

    def _single_sample(self, rng: np.random.Generator) -> tuple[list[int], complex]:
        """Run one trajectory; return (bit_list, complex_weight)."""
        sim = stim.TableauSimulator()
        # stim circuit num qubits is bigger than the real, might cause tableau track extra qubits
        # sim.set_num_qubits(self._circuit.n_qubits)
        weight: complex = 1.0 + 0j
        bits: list[int] = []

        for op, decomps in self._op_plan:
            if isinstance(op, MeasureOp):
                result = op.apply(sim)  # bool
                bits.append(int(result))
            elif isinstance(op, CliffordOp):
                op.apply(sim)
            else:
                # Each qubit in a SIMD op is sampled independently.
                for decomp in decomps:  # type: ignore[union-attr]
                    coeff, prob, gate = decomp.sample(rng)
                    _apply_gate_term(sim, gate, decomp.qubits)
                    weight *= coeff / prob

        return bits, weight


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
