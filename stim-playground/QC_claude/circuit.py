"""
Circuit representation supporting Clifford gates, non-Clifford RZ gates,
Pauli noise channels, T1/T2 damping, and Z-basis measurements.

- CliffordOp:          Clifford gate applied deterministically to the simulator.
                       Supports SIMD: one op covers any number of targets
                       (e.g. H 0 1 2 → sim.h(0, 1, 2)).
- RZOp:                Non-Clifford RZ(theta) gate; decomposed via quasi-probability
                       sampling. Supports SIMD: one op covers multiple qubits,
                       each qubit independently sampled. 1-norm = single^n_qubits.
- PauliNoiseOp:        Single-qubit Pauli noise; supports SIMD (same px/py/pz on
                       all listed qubits, each sampled independently). 1-norm = 1.
- TwoQubitPauliNoiseOp Two-qubit Pauli noise on one qubit pair. 1-norm = 1.
- DampOp:              Combined T1/T2 amplitude+phase damping; 3-term stabilizer
                       decomposition. Supports SIMD (same T1/T2/t on all listed
                       qubits, each sampled independently).
- MeasureOp:           Pauli-basis measurement via measure_observable(); supports
                       Z/X/Y basis, flip_probability for noisy readout, and reset.
                       Each MeasureOp produces exactly one measurement record.

Use Circuit.from_stim() to import an existing stim.Circuit (Clifford + noise +
measurement instructions). Multi-target Stim instructions are parsed as a single
SIMD op (except measurements: one MeasureOp per qubit target).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Union

import stim


# ── Pauli-2 label ordering (matches Stim's PAULI_CHANNEL_2 convention) ────────
_PAULI2_LABELS: tuple[str, ...] = (
    'IX', 'IY', 'IZ',
    'XI', 'XX', 'XY', 'XZ',
    'YI', 'YX', 'YY', 'YZ',
    'ZI', 'ZX', 'ZY', 'ZZ',
)

# ── Stim gate name → our canonical gate name ──────────────────────────────────
_STIM_1Q_CLIFFORD: dict[str, str] = {
    'H': 'H', 'H_YZ': 'H_YZ', 'H_XY': 'H_XY',
    'I': 'I',
    'S': 'S', 'SQRT_Z': 'S',
    'S_DAG': 'Sdg', 'SQRT_Z_DAG': 'Sdg',
    'X': 'X', 'Y': 'Y', 'Z': 'Z',
    'SQRT_X': 'SQRT_X', 'SX': 'SQRT_X',
    'SQRT_X_DAG': 'SQRT_X_DAG', 'SX_DAG': 'SQRT_X_DAG',
    'SQRT_Y': 'SQRT_Y', 'SQRT_Y_DAG': 'SQRT_Y_DAG',
    'R': 'R', 'RX': 'RX', 'RY': 'RY',  # reset gates
}

_STIM_2Q_CLIFFORD: dict[str, str] = {
    'CX': 'CX', 'CNOT': 'CX', 'ZCX': 'CX',
    'CY': 'CY',
    'CZ': 'CZ', 'ZCZ': 'CZ',
    'SWAP': 'SWAP',
    'ISWAP': 'ISWAP',
    'ISWAP_DAG': 'ISWAP_DAG',
}

# Instructions to silently skip during from_stim parsing
_STIM_SKIP: frozenset[str] = frozenset({
    'QUBIT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS',
    'MPP',
    'HERALDED_ERASE', 'HERALDED_PAULI_CHANNEL_1', 'MPAD',
})

# Measurement instruction → (basis, reset) mapping
_MEAS_MAP: dict[str, tuple[str, bool]] = {
    'M': ('Z', False), 'MZ': ('Z', False),
    'MR': ('Z', True),  'MRZ': ('Z', True),
    'MX': ('X', False), 'MRX': ('X', True),
    'MY': ('Y', False), 'MRY': ('Y', True),
}


# ── Gate op definitions ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class CliffordOp:
    """
    A Clifford gate applied to one or more qubits (SIMD-style).

    For single-qubit gates (H, S, Z, …):
        targets holds all the qubits; applied with one Stim call, e.g. sim.h(0, 1, 2).
    For two-qubit gates (CX, CZ, SWAP, …):
        targets holds interleaved pairs (c0, t0, c1, t1, …);
        applied with one Stim call, e.g. sim.cx(0, 1, 2, 3).
    """
    name: str
    targets: tuple[int, ...]

    def apply(self, sim: stim.TableauSimulator) -> None:
        """Apply this gate to a Stim TableauSimulator."""
        match self.name:
            case 'H':
                sim.h(*self.targets)
            case 'H_YZ':
                sim.h_yz(*self.targets)
            case 'H_XY':
                sim.h_xy(*self.targets)
            case 'S':
                sim.s(*self.targets)
            case 'Sdg' | 'S_DAG':
                sim.s_dag(*self.targets)
            case 'SQRT_X' | 'SX':
                sim.sqrt_x(*self.targets)
            case 'SQRT_X_DAG' | 'SX_DAG':
                sim.sqrt_x_dag(*self.targets)
            case 'SQRT_Y':
                sim.sqrt_y(*self.targets)
            case 'SQRT_Y_DAG':
                sim.sqrt_y_dag(*self.targets)
            case 'X':
                sim.x(*self.targets)
            case 'Y':
                sim.y(*self.targets)
            case 'Z':
                sim.z(*self.targets)
            case 'CX' | 'CNOT':
                sim.cx(*self.targets)
            case 'CZ':
                sim.cz(*self.targets)
            case 'CY':
                sim.cy(*self.targets)
            case 'SWAP':
                sim.swap(*self.targets)
            case 'ISWAP':
                sim.iswap(*self.targets)
            case 'ISWAP_DAG':
                sim.iswap_dag(*self.targets)
            case 'R':
                sim.reset(*self.targets)
            case 'RX':
                sim.reset_x(*self.targets)
            case 'RY':
                sim.reset_y(*self.targets)
            case 'I':
                pass
            case _:
                raise ValueError(f"Unknown Clifford gate: '{self.name}'")


@dataclass(frozen=True)
class RZOp:
    """
    Non-Clifford RZ(theta) gate; quasiprobability-decomposed during sampling.

    Supports SIMD: one op can target multiple qubits with the same angle theta.
    Each qubit is sampled independently from the same {I, Z, S} decomposition.
    The total 1-norm = single_qubit_1_norm ** len(qubits).
    """
    qubits: tuple[int, ...]
    theta: float  # rotation angle in radians


@dataclass(frozen=True)
class PauliNoiseOp:
    """
    Single-qubit Pauli noise channel (e.g. DEPOLARIZE1, PAULI_CHANNEL_1).

        chi(rho) = (1-px-py-pz)*I + px*X(rho) + py*Y(rho) + pz*Z(rho)

    Supports SIMD: one op can list multiple qubits (all with the same px/py/pz).
    Each qubit is sampled independently. Total 1-norm = 1 (exact probability).
    """
    qubits: tuple[int, ...]
    px: float   # X-error probability
    py: float   # Y-error probability
    pz: float   # Z-error probability

    @property
    def p_identity(self) -> float:
        return 1.0 - self.px - self.py - self.pz


@dataclass(frozen=True)
class TwoQubitPauliNoiseOp:
    """
    Two-qubit Pauli noise channel (e.g. DEPOLARIZE2, PAULI_CHANNEL_2).

    probs: 15-tuple of non-II Pauli error probabilities in _PAULI2_LABELS order:
        (p_IX, p_IY, p_IZ, p_XI, p_XX, p_XY, p_XZ, p_YI, p_YX, p_YY, p_YZ,
         p_ZI, p_ZX, p_ZY, p_ZZ)
    """
    q0: int
    q1: int
    probs: tuple[float, ...]  # length 15

    @property
    def p_identity(self) -> float:
        return 1.0 - sum(self.probs)


@dataclass(frozen=True)
class TickOp:
    """
    A tick (time-boundary marker).  No effect during simulation.
    Displayed as 'TICK' when the circuit is printed; parsed from Stim TICK instructions.
    """


@dataclass(frozen=True)
class DampOp:
    """
    Combined amplitude + phase damping channel (T1/T2 relaxation).

    Supports SIMD: one op can target multiple qubits with the same t/T1/T2.
    Each qubit is sampled independently. The total 1-norm is per-qubit 1-norm
    raised to the power of len(qubits).

    Parameters:
        qubits: Target qubit indices.
        t:      Evolution time (same units as T1, T2).
        T1:     Amplitude relaxation time.  float('inf') → no amplitude damping.
        T2:     Total coherence time (T2*). float('inf') → no dephasing at all.
                Must satisfy T2 ≤ 2·T1 (physical constraint: Tφ must be positive).

    Internally computes the pure dephasing time via:
        1/Tφ = 1/T2 - 1/(2·T1)

    Decomposition (3 Clifford terms per qubit):
        c̃₀ · Identity  +  c̃₁ · Z  +  c̃₂ · Reset|0⟩
    When T1 ≥ T2:       all c̃ᵢ ≥ 0 (exact probability, 1-norm = 1.0).
    When T1 < T2 ≤ 2T1: c̃₁ < 0   (quasiprobability regime, 1-norm > 1.0).
    """
    qubits: tuple[int, ...]
    t: float
    T1: float
    T2: float


@dataclass(frozen=True)
class MeasureOp:
    """
    Pauli-basis measurement via stim.TableauSimulator.measure_observable().

    Measures the Pauli observable defined by `basis` on all `qubits` and produces
    exactly ONE measurement result (the joint parity for multi-qubit observables,
    or a single-qubit result when len(qubits)==1).

    For single-qubit Stim instructions (M, MX, MR, MRX, …), from_stim creates
    one MeasureOp per qubit target so that each produces one measurement record.

    If `reset=True`, qubits are reset to the +1 eigenstate of the measurement
    basis after measurement (|0⟩ for Z, |+⟩ for X, |+i⟩ for Y).

    `flip_probability` models classical readout noise: the result is flipped
    with this probability, handled natively by measure_observable().
    """
    qubits: tuple[int, ...]
    basis: str = 'Z'              # 'Z', 'X', or 'Y'
    reset: bool = False           # True → measure then reset
    flip_probability: float = 0.0 # classical bit-flip probability

    def apply(self, sim: stim.TableauSimulator) -> bool:
        """Measure the Pauli observable and return a single bool. Optionally reset."""
        max_q = max(self.qubits)
        ps = stim.PauliString(max_q + 1)
        for q in self.qubits:
            ps[q] = self.basis
        result = sim.measure_observable(ps, flip_probability=self.flip_probability)
        if self.reset:
            sim.reset(*self.qubits)
            if self.basis == 'X':
                sim.h(*self.qubits)
            elif self.basis == 'Y':
                sim.h(*self.qubits)
                sim.s(*self.qubits)
        return result


# Union type for any circuit operation
Op = Union[CliffordOp, RZOp, PauliNoiseOp, TwoQubitPauliNoiseOp, DampOp, MeasureOp, TickOp]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_qubits(q: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalise a single qubit int or qubit tuple to a tuple."""
    return (q,) if isinstance(q, int) else tuple(q)


# ── Circuit class ─────────────────────────────────────────────────────────────

class Circuit:
    """
    A quantum circuit consisting of Clifford gates, non-Clifford RZ gates,
    and Pauli noise channels.

    All circuits start from |0...0>. Clifford gates are applied exactly;
    RZ and noise ops are handled via (quasi-)probability sampling in MCEstimator.

    Gates support SIMD-style multi-qubit targeting:

        # Single-qubit Clifford applied to several qubits at once:
        c.h(0, 1, 2)                     # H on qubits 0, 1, 2

        # Two-qubit Clifford applied to multiple pairs at once:
        c.cx(0, 1, 2, 3)                 # CNOT on (0→1) and (2→3)

        # RZ on multiple qubits (same angle, independent sampling):
        c.rz((0, 1, 2), np.pi / 4)

        # Noise on multiple qubits (same params, independent sampling):
        c.depolarize1((0, 1, 2), 0.01)

        # Damping on multiple qubits (same T1/T2/t, independent sampling):
        c.damp((0, 1), t=0.01, T1=100.0, T2=80.0)

    Build manually:
        >>> c = Circuit(n_qubits=2)
        >>> c.h(0, 1).cx(0, 1).rz(0, np.pi/4).depolarize1((0, 1), 0.01)

    Import from Stim:
        >>> sc = stim.Circuit("H 0 1\\nCX 0 1\\nDEPOLARIZE1(0.01) 0 1")
        >>> c = Circuit.from_stim(sc)   # H→1 CliffordOp, DEPOL→1 PauliNoiseOp
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self._ops: list[Op] = []

    # ── Clifford gate builders (SIMD: *qubits) ────────────────────────────────

    def h(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('H', qubits))
        return self

    def s(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('S', qubits))
        return self

    def sdg(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('Sdg', qubits))
        return self

    def x(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('X', qubits))
        return self

    def y(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('Y', qubits))
        return self

    def z(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('Z', qubits))
        return self

    def cx(self, *qubits: int) -> 'Circuit':
        """CNOT on interleaved pairs: cx(c0, t0, c1, t1, …)."""
        self._ops.append(CliffordOp('CX', qubits))
        return self

    def cz(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('CZ', qubits))
        return self

    def cy(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('CY', qubits))
        return self

    def swap(self, *qubits: int) -> 'Circuit':
        self._ops.append(CliffordOp('SWAP', qubits))
        return self

    def reset(self, *qubits: int) -> 'Circuit':
        """Reset qubits to |0> (Z basis)."""
        self._ops.append(CliffordOp('R', qubits))
        return self

    # ── Measurement builders (one MeasureOp per qubit) ─────────────────────────

    def measure(self, *qubits: int) -> 'Circuit':
        """Z-basis measurement (M instruction). One MeasureOp per qubit."""
        for q in qubits:
            self._ops.append(MeasureOp(qubits=(q,), basis='Z'))
        return self

    def measure_reset(self, *qubits: int) -> 'Circuit':
        """Z-basis measurement + reset to |0> (MR instruction)."""
        for q in qubits:
            self._ops.append(MeasureOp(qubits=(q,), basis='Z', reset=True))
        return self

    def measure_x(self, *qubits: int) -> 'Circuit':
        """X-basis measurement (MX instruction)."""
        for q in qubits:
            self._ops.append(MeasureOp(qubits=(q,), basis='X'))
        return self

    def measure_reset_x(self, *qubits: int) -> 'Circuit':
        """X-basis measurement + reset to |+> (MRX instruction)."""
        for q in qubits:
            self._ops.append(MeasureOp(qubits=(q,), basis='X', reset=True))
        return self

    def measure_y(self, *qubits: int) -> 'Circuit':
        """Y-basis measurement (MY instruction)."""
        for q in qubits:
            self._ops.append(MeasureOp(qubits=(q,), basis='Y'))
        return self

    def measure_reset_y(self, *qubits: int) -> 'Circuit':
        """Y-basis measurement + reset to |+i> (MRY instruction)."""
        for q in qubits:
            self._ops.append(MeasureOp(qubits=(q,), basis='Y', reset=True))
        return self

    def tick(self) -> 'Circuit':
        """Insert a tick (time-boundary marker). No effect on simulation."""
        self._ops.append(TickOp())
        return self

    # ── Non-Clifford gate ──────────────────────────────────────────────────────

    def rz(self, qubits: int | tuple[int, ...], theta: float) -> 'Circuit':
        """
        Non-Clifford RZ(theta) rotation gate (SIMD-capable).

        Args:
            qubits: Single qubit int, or tuple of qubit indices (same theta applied
                    to each qubit independently; 1-norm = single^n_qubits).
            theta:  Rotation angle in radians.

        Decomposed as: chi[RZ] = a*I + b*Z + c*S  (see decompositions.py).
        """
        self._ops.append(RZOp(qubits=_to_qubits(qubits), theta=theta))
        return self

    # ── Noise builders ─────────────────────────────────────────────────────────

    def pauli_channel_1(
        self, qubits: int | tuple[int, ...], px: float, py: float, pz: float
    ) -> 'Circuit':
        """Single-qubit Pauli noise (SIMD): (1-px-py-pz)*I + px*X + py*Y + pz*Z."""
        self._ops.append(PauliNoiseOp(_to_qubits(qubits), px, py, pz))
        return self

    def depolarize1(self, qubits: int | tuple[int, ...], p: float) -> 'Circuit':
        """Single-qubit depolarizing noise (SIMD): uniform Pauli error with probability p."""
        return self.pauli_channel_1(qubits, p / 3, p / 3, p / 3)

    def x_error(self, qubits: int | tuple[int, ...], p: float) -> 'Circuit':
        """Bit-flip (X) error with probability p (SIMD)."""
        return self.pauli_channel_1(qubits, p, 0.0, 0.0)

    def y_error(self, qubits: int | tuple[int, ...], p: float) -> 'Circuit':
        """Y-error with probability p (SIMD)."""
        return self.pauli_channel_1(qubits, 0.0, p, 0.0)

    def z_error(self, qubits: int | tuple[int, ...], p: float) -> 'Circuit':
        """Phase-flip (Z) error with probability p (SIMD)."""
        return self.pauli_channel_1(qubits, 0.0, 0.0, p)

    def pauli_channel_2(self, q0: int, q1: int, probs: tuple[float, ...]) -> 'Circuit':
        """
        Two-qubit Pauli noise channel.

        Args:
            q0, q1: Target qubit pair.
            probs:  15-tuple of non-II Pauli error probabilities in the order
                    (IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ).
                    This matches Stim's PAULI_CHANNEL_2 parameter order.
        """
        if len(probs) != 15:
            raise ValueError(f"Expected 15 probabilities, got {len(probs)}")
        self._ops.append(TwoQubitPauliNoiseOp(q0, q1, tuple(probs)))
        return self

    def depolarize2(self, q0: int, q1: int, p: float) -> 'Circuit':
        """Two-qubit depolarizing noise: uniform Pauli error (on all 15 non-II Paulis)."""
        return self.pauli_channel_2(q0, q1, tuple([p / 15] * 15))

    def damp(
        self,
        qubits: int | tuple[int, ...],
        t: float,
        T1: float,
        T2: float,
    ) -> 'Circuit':
        """
        Combined T1/T2 amplitude+phase damping channel (SIMD-capable).

        Args:
            qubits: Single qubit int, or tuple of qubit indices (same T1/T2/t
                    applied to each qubit independently).
            t:      Evolution time (same units as T1, T2).
            T1:     Amplitude relaxation time.  float('inf') → no amplitude damping.
            T2:     Total coherence time (T2*). float('inf') → no dephasing at all.
                    Must satisfy T2 ≤ 2·T1.
        """
        self._ops.append(DampOp(qubits=_to_qubits(qubits), t=t, T1=T1, T2=T2))
        return self

    # ── Import from Stim ───────────────────────────────────────────────────────

    @classmethod
    def from_stim(cls, stim_circuit: stim.Circuit, n_qubits: int | None = None) -> 'Circuit':
        """
        Build a Circuit from an existing stim.Circuit.

        Multi-target Stim instructions are parsed as a single SIMD op:
            H 0 1 2          → one CliffordOp('H', (0, 1, 2))
            CX 0 1 2 3       → one CliffordOp('CX', (0, 1, 2, 3))
            DEPOLARIZE1(p) 0 1 → one PauliNoiseOp((0, 1), p/3, p/3, p/3)

        Supported instructions:
            Clifford:     H, S, S_DAG, X, Y, Z, CX, CY, CZ, SWAP, ISWAP, R, RX, RY,
                          SQRT_X, SQRT_X_DAG, SQRT_Y, SQRT_Y_DAG, H_YZ, H_XY, …
            Noise:        PAULI_CHANNEL_1, PAULI_CHANNEL_2
                          DEPOLARIZE1, DEPOLARIZE2
                          X_ERROR, Y_ERROR, Z_ERROR
            Measurement:  M, MZ, MX, MY → MeasureOp(reset=False)
                          MR, MRZ, MRX, MRY → MeasureOp(reset=True)
                          Each qubit target → one MeasureOp (one measurement record).
                          Noisy variants (e.g. MR(0.005)) set flip_probability.
            REPEAT block: unrolled into a flat gate sequence.

        Silently skipped (annotations, unsupported measurement types):
            MPP, QUBIT_COORDS, DETECTOR, OBSERVABLE_INCLUDE, …

        Unknown gates raise a warning and are skipped.

        Args:
            stim_circuit: Source Stim circuit.
            n_qubits:     Override qubit count (default: stim_circuit.num_qubits).

        Returns:
            Circuit with Clifford and noise ops. Add RZ gates manually afterward
            if needed.
        """
        if n_qubits is None:
            n_qubits = stim_circuit.num_qubits or 1
        circuit = cls(n_qubits)
        circuit._extend_from_stim(stim_circuit)
        return circuit

    def _extend_from_stim(self, sc: stim.Circuit) -> None:
        """Recursively parse a stim.Circuit and append ops to self._ops."""
        for instr in sc:
            # ── REPEAT blocks: unroll ─────────────────────────────────────────
            if isinstance(instr, stim.CircuitRepeatBlock):
                for _ in range(instr.repeat_count):
                    self._extend_from_stim(instr.body_copy())
                continue

            name: str = instr.name
            # Qubit targets only (filter out measurement-record targets)
            targets = [t for t in instr.targets_copy() if t.is_qubit_target]
            args: list[float] = instr.gate_args_copy()

            # ── Single-qubit Clifford (SIMD: one op for all targets) ──────────
            if name in _STIM_1Q_CLIFFORD:
                gate = _STIM_1Q_CLIFFORD[name]
                self._ops.append(CliffordOp(gate, tuple(t.value for t in targets)))

            # ── Two-qubit Clifford (SIMD: one op with all interleaved pairs) ──
            elif name in _STIM_2Q_CLIFFORD:
                gate = _STIM_2Q_CLIFFORD[name]
                self._ops.append(CliffordOp(gate, tuple(t.value for t in targets)))

            # ── Single-qubit noise (SIMD: one op for all targets) ─────────────
            elif name == 'PAULI_CHANNEL_1':
                px, py, pz = args
                self._ops.append(
                    PauliNoiseOp(tuple(t.value for t in targets), px, py, pz)
                )

            elif name == 'DEPOLARIZE1':
                p = args[0]
                self._ops.append(
                    PauliNoiseOp(tuple(t.value for t in targets), p / 3, p / 3, p / 3)
                )

            elif name in ('X_ERROR', 'Y_ERROR', 'Z_ERROR'):
                p = args[0]
                px = p if name == 'X_ERROR' else 0.0
                py = p if name == 'Y_ERROR' else 0.0
                pz = p if name == 'Z_ERROR' else 0.0
                self._ops.append(
                    PauliNoiseOp(tuple(t.value for t in targets), px, py, pz)
                )

            # ── Two-qubit noise (one op per pair) ─────────────────────────────
            elif name == 'PAULI_CHANNEL_2':
                probs = tuple(args)  # 15 values
                for i in range(0, len(targets), 2):
                    self._ops.append(
                        TwoQubitPauliNoiseOp(
                            targets[i].value, targets[i + 1].value, probs
                        )
                    )

            elif name == 'DEPOLARIZE2':
                p = args[0]
                probs = tuple([p / 15] * 15)
                for i in range(0, len(targets), 2):
                    self._ops.append(
                        TwoQubitPauliNoiseOp(
                            targets[i].value, targets[i + 1].value, probs
                        )
                    )

            # ── Measurements (one MeasureOp per qubit target) ─────────────────
            elif name in _MEAS_MAP:
                basis, reset = _MEAS_MAP[name]
                flip_prob = args[0] if args else 0.0
                for t in targets:
                    self._ops.append(MeasureOp(
                        qubits=(t.value,), basis=basis,
                        reset=reset, flip_probability=flip_prob,
                    ))

            # ── Tick ──────────────────────────────────────────────────────────
            elif name == 'TICK':
                self._ops.append(TickOp())

            # ── Silently skip ─────────────────────────────────────────────────
            elif name in _STIM_SKIP:
                pass

            else:
                warnings.warn(
                    f"Circuit.from_stim: unsupported instruction '{name}' skipped.",
                    stacklevel=3,
                )

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def ops(self) -> list[Op]:
        return list(self._ops)

    @property
    def n_clifford_gates(self) -> int:
        return sum(1 for op in self._ops if isinstance(op, CliffordOp))

    @property
    def n_rz_gates(self) -> int:
        return sum(1 for op in self._ops if isinstance(op, RZOp))

    @property
    def n_noise_ops(self) -> int:
        return sum(
            1 for op in self._ops
            if isinstance(op, (PauliNoiseOp, TwoQubitPauliNoiseOp))
        )

    @property
    def n_measurements(self) -> int:
        """Total number of measurement records (one per MeasureOp)."""
        return sum(1 for op in self._ops if isinstance(op, MeasureOp))

    def __len__(self) -> int:
        return len(self._ops)

    @staticmethod
    def _op_to_str(op: 'Op') -> str:
        """Format a single op as a Stim-like instruction string."""
        if isinstance(op, TickOp):
            return 'TICK'
        if isinstance(op, CliffordOp):
            targets = ' '.join(str(t) for t in op.targets)
            name = 'S_DAG' if op.name == 'Sdg' else op.name
            return f'{name} {targets}'
        if isinstance(op, RZOp):
            qubits_str = ' '.join(str(q) for q in op.qubits)
            return f'RZ({op.theta:.6g}) {qubits_str}'
        if isinstance(op, PauliNoiseOp):
            qubits_str = ' '.join(str(q) for q in op.qubits)
            px, py, pz = op.px, op.py, op.pz
            if abs(px - py) < 1e-12 and abs(py - pz) < 1e-12:
                return f'DEPOLARIZE1({px * 3:.6g}) {qubits_str}'
            return f'PAULI_CHANNEL_1({px:.6g},{py:.6g},{pz:.6g}) {qubits_str}'
        if isinstance(op, TwoQubitPauliNoiseOp):
            p0 = op.probs[0]
            if all(abs(p - p0) < 1e-12 for p in op.probs):
                return f'DEPOLARIZE2({p0 * 15:.6g}) {op.q0} {op.q1}'
            probs = ','.join(f'{p:.6g}' for p in op.probs)
            return f'PAULI_CHANNEL_2({probs}) {op.q0} {op.q1}'
        if isinstance(op, DampOp):
            qubits_str = ' '.join(str(q) for q in op.qubits)
            T1 = 'inf' if op.T1 == float('inf') else f'{op.T1:.6g}'
            T2 = 'inf' if op.T2 == float('inf') else f'{op.T2:.6g}'
            return f'DAMP(t={op.t:.6g},T1={T1},T2={T2}) {qubits_str}'
        if isinstance(op, MeasureOp):
            targets = ' '.join(str(q) for q in op.qubits)
            basis_suffix = '' if op.basis == 'Z' else op.basis
            name = ('MR' if op.reset else 'M') + basis_suffix
            if op.flip_probability > 0:
                return f'{name}({op.flip_probability:.6g}) {targets}'
            return f'{name} {targets}'
        return f'UNKNOWN({type(op).__name__})'

    def __str__(self) -> str:
        header = (
            f'Circuit(n_qubits={self.n_qubits}, '
            f'clifford={self.n_clifford_gates}, '
            f'rz={self.n_rz_gates}, '
            f'noise={self.n_noise_ops}, '
            f'measurements={self.n_measurements})'
        )
        body = '\n'.join(self._op_to_str(op) for op in self._ops)
        return f'{header}\n{body}' if body else header

    def __repr__(self) -> str:
        return str(self)
