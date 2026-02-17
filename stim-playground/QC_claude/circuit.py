"""
Circuit representation supporting Clifford gates, non-Clifford RZ gates,
Pauli noise channels, and Z-basis measurements.

- CliffordOp:          Clifford gate applied deterministically to the simulator.
- RZOp:                Non-Clifford RZ(theta) gate; decomposed via quasi-probability sampling.
- PauliNoiseOp:        Single-qubit Pauli noise; sampled with true probability (1-norm = 1).
- TwoQubitPauliNoiseOp Two-qubit Pauli noise; sampled with true probability (1-norm = 1).
- MeasureOp:           Z-basis measurement (optionally with reset); used by MCSampler.

Use Circuit.from_stim() to import an existing stim.Circuit (Clifford + noise + M/MR instructions).
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
    'TICK', 'QUBIT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS',
    'MX', 'MY', 'MPP', 'MRX', 'MRY',
    'HERALDED_ERASE', 'HERALDED_PAULI_CHANNEL_1', 'MPAD',
})


# ── Gate op definitions ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class CliffordOp:
    """A Clifford gate applied to one or two qubits."""
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
    """Non-Clifford RZ(theta) gate; quasiprobability-decomposed during sampling."""
    qubit: int
    theta: float  # rotation angle in radians


@dataclass(frozen=True)
class PauliNoiseOp:
    """
    Single-qubit Pauli noise channel (e.g. DEPOLARIZE1, PAULI_CHANNEL_1).

        chi(rho) = (1-px-py-pz)*I + px*X(rho) + py*Y(rho) + pz*Z(rho)

    All coefficients are non-negative (true probabilities); 1-norm = 1.
    """
    qubit: int
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
class MeasureOp:
    """
    Z-basis measurement on one or more qubits.

    Produces one bit per qubit in `qubits`, appended to the sample in circuit order.
    If `reset=True` (i.e. MR instruction), each qubit is reset to |0> immediately
    after measurement (used for mid-circuit stabilizer readouts in memory experiments).
    """
    qubits: tuple[int, ...]
    reset: bool = False  # True → MR: measure then reset to |0>

    def apply(self, sim: stim.TableauSimulator) -> list[bool]:
        """Measure qubits and return bit results (True = 1). Optionally reset afterward."""
        results = list(sim.measure_many(*self.qubits))
        if self.reset:
            sim.reset(*self.qubits)
        return results


# Union type for any circuit operation
Op = Union[CliffordOp, RZOp, PauliNoiseOp, TwoQubitPauliNoiseOp, MeasureOp]


# ── Circuit class ─────────────────────────────────────────────────────────────

class Circuit:
    """
    A quantum circuit consisting of Clifford gates, non-Clifford RZ gates,
    and Pauli noise channels.

    All circuits start from |0...0>. Clifford gates are applied exactly;
    RZ and noise ops are handled via (quasi-)probability sampling in MCEstimator.

    Build manually:
        >>> c = Circuit(n_qubits=2)
        >>> c.h(0).cx(0, 1).rz(0, np.pi/4).depolarize1(0, 0.01)

    Import from Stim:
        >>> sc = stim.Circuit("H 0\\nCX 0 1\\nDEPOLARIZE1(0.01) 0 1")
        >>> c = Circuit.from_stim(sc)
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self._ops: list[Op] = []

    # ── Clifford gate builders ─────────────────────────────────────────────────

    def h(self, qubit: int) -> 'Circuit':
        self._ops.append(CliffordOp('H', (qubit,)))
        return self

    def s(self, qubit: int) -> 'Circuit':
        self._ops.append(CliffordOp('S', (qubit,)))
        return self

    def sdg(self, qubit: int) -> 'Circuit':
        self._ops.append(CliffordOp('Sdg', (qubit,)))
        return self

    def x(self, qubit: int) -> 'Circuit':
        self._ops.append(CliffordOp('X', (qubit,)))
        return self

    def y(self, qubit: int) -> 'Circuit':
        self._ops.append(CliffordOp('Y', (qubit,)))
        return self

    def z(self, qubit: int) -> 'Circuit':
        self._ops.append(CliffordOp('Z', (qubit,)))
        return self

    def cx(self, control: int, target: int) -> 'Circuit':
        self._ops.append(CliffordOp('CX', (control, target)))
        return self

    def cz(self, q0: int, q1: int) -> 'Circuit':
        self._ops.append(CliffordOp('CZ', (q0, q1)))
        return self

    def cy(self, q0: int, q1: int) -> 'Circuit':
        self._ops.append(CliffordOp('CY', (q0, q1)))
        return self

    def swap(self, q0: int, q1: int) -> 'Circuit':
        self._ops.append(CliffordOp('SWAP', (q0, q1)))
        return self

    def reset(self, qubit: int) -> 'Circuit':
        """Reset qubit to |0> (Z basis)."""
        self._ops.append(CliffordOp('R', (qubit,)))
        return self

    # ── Measurement builders ───────────────────────────────────────────────────

    def measure(self, *qubits: int) -> 'Circuit':
        """Z-basis measurement on one or more qubits (M instruction)."""
        self._ops.append(MeasureOp(qubits=qubits, reset=False))
        return self

    def measure_reset(self, *qubits: int) -> 'Circuit':
        """Z-basis measurement + reset to |0> (MR instruction)."""
        self._ops.append(MeasureOp(qubits=qubits, reset=True))
        return self

    # ── Non-Clifford gate ──────────────────────────────────────────────────────

    def rz(self, qubit: int, theta: float) -> 'Circuit':
        """
        Non-Clifford RZ(theta) rotation gate.
        Decomposed as: chi[RZ] = a*I + b*Z + c*S  (see decompositions.py).
        """
        self._ops.append(RZOp(qubit=qubit, theta=theta))
        return self

    # ── Noise builders ─────────────────────────────────────────────────────────

    def pauli_channel_1(self, qubit: int, px: float, py: float, pz: float) -> 'Circuit':
        """Single-qubit Pauli noise: (1-px-py-pz)*I + px*X + py*Y + pz*Z."""
        self._ops.append(PauliNoiseOp(qubit, px, py, pz))
        return self

    def depolarize1(self, qubit: int, p: float) -> 'Circuit':
        """Single-qubit depolarizing noise: uniform Pauli error with probability p."""
        return self.pauli_channel_1(qubit, p / 3, p / 3, p / 3)

    def x_error(self, qubit: int, p: float) -> 'Circuit':
        """Bit-flip (X) error with probability p."""
        return self.pauli_channel_1(qubit, p, 0.0, 0.0)

    def y_error(self, qubit: int, p: float) -> 'Circuit':
        """Y-error with probability p."""
        return self.pauli_channel_1(qubit, 0.0, p, 0.0)

    def z_error(self, qubit: int, p: float) -> 'Circuit':
        """Phase-flip (Z) error with probability p."""
        return self.pauli_channel_1(qubit, 0.0, 0.0, p)

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

    # ── Import from Stim ───────────────────────────────────────────────────────

    @classmethod
    def from_stim(cls, stim_circuit: stim.Circuit, n_qubits: int | None = None) -> 'Circuit':
        """
        Build a Circuit from an existing stim.Circuit.

        Supported instructions:
            Clifford:     H, S, S_DAG, X, Y, Z, CX, CY, CZ, SWAP, ISWAP, R, RX, RY,
                          SQRT_X, SQRT_X_DAG, SQRT_Y, SQRT_Y_DAG, H_YZ, H_XY, …
            Noise:        PAULI_CHANNEL_1, PAULI_CHANNEL_2
                          DEPOLARIZE1, DEPOLARIZE2
                          X_ERROR, Y_ERROR, Z_ERROR
            Measurement:  M, MZ  → MeasureOp(reset=False)
                          MR, MRZ → MeasureOp(reset=True)
            REPEAT block: unrolled into a flat gate sequence.

        Silently skipped (annotations, unsupported measurement bases):
            MX, MY, MPP, MRX, MRY, TICK, QUBIT_COORDS, DETECTOR, …

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

            # ── Single-qubit Clifford ─────────────────────────────────────────
            if name in _STIM_1Q_CLIFFORD:
                gate = _STIM_1Q_CLIFFORD[name]
                for t in targets:
                    self._ops.append(CliffordOp(gate, (t.value,)))

            # ── Two-qubit Clifford ────────────────────────────────────────────
            elif name in _STIM_2Q_CLIFFORD:
                gate = _STIM_2Q_CLIFFORD[name]
                for i in range(0, len(targets), 2):
                    self._ops.append(
                        CliffordOp(gate, (targets[i].value, targets[i + 1].value))
                    )

            # ── Single-qubit noise ────────────────────────────────────────────
            elif name == 'PAULI_CHANNEL_1':
                px, py, pz = args
                for t in targets:
                    self._ops.append(PauliNoiseOp(t.value, px, py, pz))

            elif name == 'DEPOLARIZE1':
                p = args[0]
                for t in targets:
                    self._ops.append(PauliNoiseOp(t.value, p / 3, p / 3, p / 3))

            elif name in ('X_ERROR', 'Y_ERROR', 'Z_ERROR'):
                p = args[0]
                for t in targets:
                    px = p if name == 'X_ERROR' else 0.0
                    py = p if name == 'Y_ERROR' else 0.0
                    pz = p if name == 'Z_ERROR' else 0.0
                    self._ops.append(PauliNoiseOp(t.value, px, py, pz))

            # ── Two-qubit noise ───────────────────────────────────────────────
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

            # ── Measurements ──────────────────────────────────────────────────
            elif name in ('M', 'MZ'):
                qubits = tuple(t.value for t in targets)
                self._ops.append(MeasureOp(qubits=qubits, reset=False))

            elif name in ('MR', 'MRZ'):
                qubits = tuple(t.value for t in targets)
                self._ops.append(MeasureOp(qubits=qubits, reset=True))

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
        """Total number of measurement bits (sum of qubits across all MeasureOps)."""
        return sum(len(op.qubits) for op in self._ops if isinstance(op, MeasureOp))

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        return (
            f"Circuit(n_qubits={self.n_qubits}, "
            f"clifford={self.n_clifford_gates}, "
            f"rz={self.n_rz_gates}, "
            f"noise={self.n_noise_ops}, "
            f"measurements={self.n_measurements})"
        )
