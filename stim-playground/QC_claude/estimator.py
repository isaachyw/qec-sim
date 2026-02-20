"""
Monte Carlo estimator for quantum circuits using stabilizer channel decompositions.

Implements the algorithm from algo.md:

    Inputs:
        rho = sum_i q0[i] * sigma[i]          (initial state decomposition)
        chi[k] = sum_i qk[i] * S[k][i]        (gate channel decompositions)
        phi = sum_i q_obs[i] * sigma[i]        (observable decomposition)

    Algorithm (per sample):
        1. Sample index i_k from |q[k]|/norm for each stochastic level k.
        2. Start from stabilizer state sigma[i_0] = |0...0>.
        3. For k = 1..K: rho_star = apply_channel(S[k][i_k], rho_star).
        4. Compute weight w = prod_k q[k][i_k] / prod_k p[k][i_k].
        5. Compute f = Tr(phi * rho_star)  [exact, using Stim].
        6. Accumulate F += w * f / N.

Implementation notes:
    - Initial state: |0...0>  (pure stabilizer state; trivial decomposition)
    - RZ gates:  decomposed as a*I + b*Z + c*S  (quasi-probability; 1-norm > 1)
    - Noise ops: decomposed as probability distribution over Paulis (1-norm = 1)
    - Observable: computed exactly via Stim's peek_observable_expectation
      rather than sampled (Rao-Blackwellisation; reduces variance at no extra cost)
    - The Stim TableauSimulator is the stabilizer simulation backend.
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
from .observable import PauliObservable


# ── Gate-term application helper ──────────────────────────────────────────────


def _apply_gate_term(
    sim: stim.TableauSimulator, gate: str, qubits: tuple[int, ...]
) -> None:
    """
    Apply a sampled Clifford gate term to the simulator.

    For single-qubit decompositions (RZ and 1q noise):
        gate ∈ {'I', 'X', 'Y', 'Z', 'S'}

    For two-qubit noise decompositions:
        gate is a 2-character Pauli string, e.g. 'IX', 'ZY', 'II'
    """
    if len(qubits) == 1:
        q = qubits[0]
        match gate:
            case "I":
                pass
            case "X":
                sim.x(q)
            case "Y":
                sim.y(q)
            case "Z":
                sim.z(q)
            case "S":
                sim.s(q)
            case "RESET":
                sim.reset(q)
    else:
        # Two-qubit Pauli: apply each character to its qubit
        for pauli, q in zip(gate, qubits):
            match pauli:
                case "I":
                    pass
                case "X":
                    sim.x(q)
                case "Y":
                    sim.y(q)
                case "Z":
                    sim.z(q)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class EstimationResult:
    """
    Result of a Monte Carlo expectation value estimation.

    Attributes:
        value:       Estimated expectation value  E[phi].
        std_error:   Monte Carlo standard error (std / sqrt(N)).
        n_samples:   Number of samples used.
        one_norm:    Product of 1-norms of all RZ gate decompositions (gamma).
                     Variance of the estimator scales as gamma^2 / N.
                     Noise channels contribute 1-norm = 1 (no overhead).
        raw_samples: Array of weighted per-sample estimates (shape: [n_samples]).
    """

    value: float
    std_error: float
    n_samples: int
    one_norm: float
    raw_samples: np.ndarray

    def __repr__(self) -> str:
        return (
            f"EstimationResult("
            f"value={self.value:.6f} ± {self.std_error:.6f}, "
            f"n_samples={self.n_samples}, "
            f"one_norm={self.one_norm:.4f})"
        )


# ── Module-level worker (must be at module level for pickle) ──────────────────


def _estimator_proc_worker(
    estimator: "MCEstimator", child_seed: np.random.SeedSequence, n: int
) -> list[float]:
    """
    Worker function for ProcessPoolExecutor.

    Must be at module level so multiprocessing can pickle it by name.
    child_seed is a spawned SeedSequence (unique per worker via spawn_key).
    Returns the real parts of n weighted sample values.
    """
    rng = np.random.default_rng(child_seed)
    return [estimator._single_sample(rng).real for _ in range(n)]


# ── Core estimator ────────────────────────────────────────────────────────────


class MCEstimator:
    """
    Monte Carlo estimator for quantum expectation values using stabilizer
    channel decompositions (quasiprobability sampling).

    Supports:
        - Clifford gates (applied deterministically)
        - RZ(theta) non-Clifford gates (quasi-probability decomposition)
        - Single-qubit Pauli noise: DEPOLARIZE1, PAULI_CHANNEL_1, X/Y/Z_ERROR
        - Two-qubit Pauli noise:    DEPOLARIZE2, PAULI_CHANNEL_2
        - Circuits imported from Stim via Circuit.from_stim()

    Example:
        >>> import numpy as np, stim
        >>> from QC_claude import Circuit, PauliObservable, MCEstimator
        >>>
        >>> sc = stim.Circuit("H 0\\nCX 0 1\\nDEPOLARIZE1(0.01) 0 1")
        >>> c = Circuit.from_stim(sc)
        >>> c.rz(0, np.pi / 4)
        >>>
        >>> obs = PauliObservable(n_qubits=2, terms=[('ZZ', 1.0)])
        >>> result = MCEstimator(c, obs).estimate(n_samples=50_000, seed=0)
    """

    def __init__(self, circuit: Circuit, observable: PauliObservable) -> None:
        if circuit.n_qubits != observable.n_qubits:
            raise ValueError(
                f"Circuit has {circuit.n_qubits} qubits but observable has "
                f"{observable.n_qubits} qubits."
            )
        self.circuit = circuit
        self.observable = observable

        # Build a unified op plan: list of (op, decompositions_or_None).
        #   CliffordOp → None            (applied deterministically)
        #   TickOp     → skipped
        #   RZOp / PauliNoiseOp / DampOp → list of per-qubit GateDecompositions
        #       (one decomp per qubit; each qubit sampled independently)
        #   TwoQubitPauliNoiseOp → single-element list[GateDecomposition]
        # Weight = product of (coeff/prob) over all decomps across all ops.
        # 1-norm  = product of individual decomp.one_norm values.
        self._op_plan: list[tuple[object, list[GateDecomposition] | None]] = []
        for op in circuit.ops:
            if isinstance(op, CliffordOp):
                self._op_plan.append((op, None))
            elif isinstance(op, RZOp):
                self._op_plan.append(
                    (op, [rz_decomposition(q, op.theta) for q in op.qubits])
                )
            elif isinstance(op, PauliNoiseOp):
                self._op_plan.append(
                    (op, [pauli_noise_decomp_1q(q, op.px, op.py, op.pz)
                           for q in op.qubits])
                )
            elif isinstance(op, TwoQubitPauliNoiseOp):
                self._op_plan.append(
                    (op, [pauli_noise_decomp_2q(op.q0, op.q1, op.probs)])
                )
            elif isinstance(op, DampOp):
                self._op_plan.append(
                    (op, [damp_decomposition(q, op.t, op.T1, op.T2)
                           for q in op.qubits])
                )
            elif isinstance(op, TickOp):
                pass  # tick markers have no simulation effect
            elif isinstance(op, MeasureOp):
                raise ValueError(
                    "MCEstimator does not support circuits with MeasureOp. "
                    "Use MCSampler for circuits with measurements."
                )

    @property
    def one_norm(self) -> float:
        """
        Product of the 1-norms of all stochastic channel decompositions.

        RZ gates contribute gamma > 1; noise ops contribute exactly 1.
        """
        product = 1.0
        for _, decomps in self._op_plan:
            if decomps is not None:
                for decomp in decomps:
                    product *= decomp.one_norm
        return product

    # ── Main estimation method ────────────────────────────────────────────────

    def estimate(
        self,
        n_samples: int = 10_000,
        seed: int | None = None,
        n_workers: int | None = None,
    ) -> EstimationResult:
        """
        Run the Monte Carlo estimator.

        Args:
            n_samples:  Number of Monte Carlo samples.
            seed:       Random seed for reproducibility.
            n_workers:  Number of parallel worker processes.
                        None or 1 → serial (default).
                        -1 → one process per logical CPU (os.cpu_count()).
                        N → N processes.

        Returns:
            EstimationResult with value, uncertainty, and diagnostics.
        """
        import os

        if n_workers == -1:
            n_workers = os.cpu_count()
        if n_workers is None or n_workers <= 1:
            # ── Serial path ───────────────────────────────────────────────────
            rng = np.random.default_rng(seed)
            raw: list[float] = [self._single_sample(rng).real for _ in range(n_samples)]
        else:
            # ── Parallel path (ProcessPoolExecutor) ───────────────────────────
            base, rem = divmod(n_samples, n_workers)
            chunk_sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]
            child_seeds = np.random.SeedSequence(seed).spawn(n_workers)
            raw = []
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(_estimator_proc_worker, self, s, n)
                    for s, n in zip(child_seeds, chunk_sizes)
                ]
                for f in futures:
                    raw.extend(f.result())

        real_samples = np.array(raw, dtype=np.float64)
        value = float(np.mean(real_samples))
        std_error = float(np.std(real_samples, ddof=1) / np.sqrt(n_samples))

        return EstimationResult(
            value=value,
            std_error=std_error,
            n_samples=n_samples,
            one_norm=self.one_norm,
            raw_samples=real_samples,
        )

    def _single_sample(self, rng: np.random.Generator) -> complex:
        """
        Execute one Monte Carlo sample:
            1. Initialise Stim simulator in |0...0>.
            2. Walk the op plan:
               - (op, None):   CliffordOp → apply deterministically.
               - (op, decomp): Sample a channel term, apply its Clifford gate,
                               accumulate quasi-probability weight.
            3. Compute Tr(phi * rho_star) exactly with Stim.
            4. Return weight * Tr(phi * rho_star).

        For noise ops (1-norm = 1): coeff == prob, so weight contribution = 1.
        For RZ ops (1-norm > 1):    coeff may differ in sign, giving
                                    non-unit (possibly negative) weight.
        """
        sim = stim.TableauSimulator()
        weight: complex = 1.0 + 0j

        for op, decomps in self._op_plan:
            if decomps is None:
                op.apply(sim)  # type: ignore[union-attr]
            else:
                # Each qubit in a SIMD op is sampled independently.
                for decomp in decomps:
                    coeff, prob, gate = decomp.sample(rng)
                    _apply_gate_term(sim, gate, decomp.qubits)
                    weight *= coeff / prob

        f = self.observable.expectation(sim)
        return weight * f


# ── Functional interface ──────────────────────────────────────────────────────


def estimate(
    circuit: Circuit,
    observable: PauliObservable,
    n_samples: int = 10_000,
    seed: int | None = None,
    n_workers: int | None = None,
) -> EstimationResult:
    """
    Functional interface: estimate E[observable] for `circuit` from |0...0>.

    Equivalent to ``MCEstimator(circuit, observable).estimate(n_samples, seed, n_workers)``.

    Args:
        circuit:    Circuit with Clifford, RZ, and/or noise ops.
        observable: Pauli observable to measure.
        n_samples:  Number of Monte Carlo samples.
        seed:       Random seed for reproducibility.
        n_workers:  Number of parallel worker processes (None=serial, -1=all CPUs).

    Returns:
        EstimationResult with value, uncertainty, and diagnostics.
    """
    return MCEstimator(circuit, observable).estimate(
        n_samples=n_samples, seed=seed, n_workers=n_workers
    )
