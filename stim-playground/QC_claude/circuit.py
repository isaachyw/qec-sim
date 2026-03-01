"""
Circuit representation supporting Clifford gates, non-Clifford RZ gates,
Pauli noise channels, T1/T2 damping, and Z-basis measurements.

Tries to import the C++ implementation (_circuit) built via pybind11 for
faster data structures. Falls back to pure Python if the .so is not found.
"""

from __future__ import annotations

from typing import Union

try:
    from ._circuit import (  # type: ignore[import-not-found]
        CliffordOp,
        RZOp,
        PauliNoiseOp,
        TwoQubitPauliNoiseOp,
        TickOp,
        DampOp,
        MeasureOp,
        Circuit,
    )

    _CPP_BACKEND = True
except ImportError:
    _CPP_BACKEND = False

    import warnings
    from dataclasses import dataclass

    import stim

    # ── Pauli-2 label ordering ────────────────────────────────────────────────
    _PAULI2_LABELS: tuple[str, ...] = (
        "IX",
        "IY",
        "IZ",
        "XI",
        "XX",
        "XY",
        "XZ",
        "YI",
        "YX",
        "YY",
        "YZ",
        "ZI",
        "ZX",
        "ZY",
        "ZZ",
    )

    _STIM_1Q_CLIFFORD: dict[str, str] = {
        "H": "H",
        "H_YZ": "H_YZ",
        "H_XY": "H_XY",
        "I": "I",
        "S": "S",
        "SQRT_Z": "S",
        "S_DAG": "Sdg",
        "SQRT_Z_DAG": "Sdg",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        "SQRT_X": "SQRT_X",
        "SX": "SQRT_X",
        "SQRT_X_DAG": "SQRT_X_DAG",
        "SX_DAG": "SQRT_X_DAG",
        "SQRT_Y": "SQRT_Y",
        "SQRT_Y_DAG": "SQRT_Y_DAG",
        "R": "R",
        "RX": "RX",
        "RY": "RY",
    }

    _STIM_2Q_CLIFFORD: dict[str, str] = {
        "CX": "CX",
        "CNOT": "CX",
        "ZCX": "CX",
        "CY": "CY",
        "CZ": "CZ",
        "ZCZ": "CZ",
        "SWAP": "SWAP",
        "ISWAP": "ISWAP",
        "ISWAP_DAG": "ISWAP_DAG",
    }

    _STIM_SKIP: frozenset[str] = frozenset(
        {
            "QUBIT_COORDS",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "SHIFT_COORDS",
            "MPP",
            "HERALDED_ERASE",
            "HERALDED_PAULI_CHANNEL_1",
            "MPAD",
        }
    )

    _MEAS_MAP: dict[str, tuple[str, bool]] = {
        "M": ("Z", False),
        "MZ": ("Z", False),
        "MR": ("Z", True),
        "MRZ": ("Z", True),
        "MX": ("X", False),
        "MRX": ("X", True),
        "MY": ("Y", False),
        "MRY": ("Y", True),
    }

    @dataclass(frozen=True)
    class CliffordOp:
        name: str
        targets: tuple[int, ...]

        def apply(self, sim: stim.TableauSimulator) -> None:
            match self.name:
                case "H":
                    sim.h(*self.targets)
                case "H_YZ":
                    sim.h_yz(*self.targets)
                case "H_XY":
                    sim.h_xy(*self.targets)
                case "S":
                    sim.s(*self.targets)
                case "Sdg" | "S_DAG":
                    sim.s_dag(*self.targets)
                case "SQRT_X" | "SX":
                    sim.sqrt_x(*self.targets)
                case "SQRT_X_DAG" | "SX_DAG":
                    sim.sqrt_x_dag(*self.targets)
                case "SQRT_Y":
                    sim.sqrt_y(*self.targets)
                case "SQRT_Y_DAG":
                    sim.sqrt_y_dag(*self.targets)
                case "X":
                    sim.x(*self.targets)
                case "Y":
                    sim.y(*self.targets)
                case "Z":
                    sim.z(*self.targets)
                case "CX" | "CNOT":
                    sim.cx(*self.targets)
                case "CZ":
                    sim.cz(*self.targets)
                case "CY":
                    sim.cy(*self.targets)
                case "SWAP":
                    sim.swap(*self.targets)
                case "ISWAP":
                    sim.iswap(*self.targets)
                case "ISWAP_DAG":
                    sim.iswap_dag(*self.targets)
                case "R":
                    sim.reset(*self.targets)
                case "RX":
                    sim.reset_x(*self.targets)
                case "RY":
                    sim.reset_y(*self.targets)
                case "I":
                    pass
                case _:
                    raise ValueError(f"Unknown Clifford gate: '{self.name}'")

    @dataclass(frozen=True)
    class RZOp:
        qubits: tuple[int, ...]
        theta: float

    @dataclass(frozen=True)
    class PauliNoiseOp:
        qubits: tuple[int, ...]
        px: float
        py: float
        pz: float

        @property
        def p_identity(self) -> float:
            return 1.0 - self.px - self.py - self.pz

    @dataclass(frozen=True)
    class TwoQubitPauliNoiseOp:
        q0: int
        q1: int
        probs: tuple[float, ...]

        @property
        def p_identity(self) -> float:
            return 1.0 - sum(self.probs)

    @dataclass(frozen=True)
    class TickOp:
        pass

    @dataclass(frozen=True)
    class DampOp:
        qubits: tuple[int, ...]
        t: float
        T1: float
        T2: float

    @dataclass(frozen=True)
    class MeasureOp:
        qubits: tuple[int, ...]
        basis: str = "Z"
        reset: bool = False
        flip_probability: float = 0.0

        def apply(self, sim: stim.TableauSimulator) -> bool:
            max_q = max(self.qubits)
            ps = stim.PauliString(max_q + 1)
            for q in self.qubits:
                ps[q] = self.basis
            result = sim.measure_observable(ps, flip_probability=self.flip_probability)
            if self.reset:
                if self.basis == "Z":
                    sim.reset(*self.qubits)
                elif self.basis == "X":
                    sim.reset_x(*self.qubits)
                elif self.basis == "Y":
                    sim.reset_y(*self.qubits)
            return result

    def _to_qubits(q: int | tuple[int, ...]) -> tuple[int, ...]:
        return (q,) if isinstance(q, int) else tuple(q)

    class Circuit:
        def __init__(self, n_qubits: int) -> None:
            if n_qubits < 1:
                raise ValueError("n_qubits must be >= 1")
            self.n_qubits = n_qubits
            self._ops: list[Op] = []

        def h(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("H", qubits))
            return self

        def s(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("S", qubits))
            return self

        def sdg(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("Sdg", qubits))
            return self

        def x(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("X", qubits))
            return self

        def y(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("Y", qubits))
            return self

        def z(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("Z", qubits))
            return self

        def cx(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("CX", qubits))
            return self

        def cz(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("CZ", qubits))
            return self

        def cy(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("CY", qubits))
            return self

        def swap(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("SWAP", qubits))
            return self

        def reset(self, *qubits: int) -> "Circuit":
            self._ops.append(CliffordOp("R", qubits))
            return self

        def measure(self, *qubits: int) -> "Circuit":
            for q in qubits:
                self._ops.append(MeasureOp(qubits=(q,), basis="Z"))
            return self

        def measure_reset(self, *qubits: int) -> "Circuit":
            for q in qubits:
                self._ops.append(MeasureOp(qubits=(q,), basis="Z", reset=True))
            return self

        def measure_x(self, *qubits: int) -> "Circuit":
            for q in qubits:
                self._ops.append(MeasureOp(qubits=(q,), basis="X"))
            return self

        def measure_reset_x(self, *qubits: int) -> "Circuit":
            for q in qubits:
                self._ops.append(MeasureOp(qubits=(q,), basis="X", reset=True))
            return self

        def measure_y(self, *qubits: int) -> "Circuit":
            for q in qubits:
                self._ops.append(MeasureOp(qubits=(q,), basis="Y"))
            return self

        def measure_reset_y(self, *qubits: int) -> "Circuit":
            for q in qubits:
                self._ops.append(MeasureOp(qubits=(q,), basis="Y", reset=True))
            return self

        def tick(self) -> "Circuit":
            self._ops.append(TickOp())
            return self

        def rz(self, qubits: int | tuple[int, ...], theta: float) -> "Circuit":
            self._ops.append(RZOp(qubits=_to_qubits(qubits), theta=theta))
            return self

        def pauli_channel_1(
            self, qubits: int | tuple[int, ...], px: float, py: float, pz: float
        ) -> "Circuit":
            self._ops.append(PauliNoiseOp(_to_qubits(qubits), px, py, pz))
            return self

        def depolarize1(self, qubits: int | tuple[int, ...], p: float) -> "Circuit":
            return self.pauli_channel_1(qubits, p / 3, p / 3, p / 3)

        def x_error(self, qubits: int | tuple[int, ...], p: float) -> "Circuit":
            return self.pauli_channel_1(qubits, p, 0.0, 0.0)

        def y_error(self, qubits: int | tuple[int, ...], p: float) -> "Circuit":
            return self.pauli_channel_1(qubits, 0.0, p, 0.0)

        def z_error(self, qubits: int | tuple[int, ...], p: float) -> "Circuit":
            return self.pauli_channel_1(qubits, 0.0, 0.0, p)

        def pauli_channel_2(
            self, q0: int, q1: int, probs: tuple[float, ...]
        ) -> "Circuit":
            if len(probs) != 15:
                raise ValueError(f"Expected 15 probabilities, got {len(probs)}")
            self._ops.append(TwoQubitPauliNoiseOp(q0, q1, tuple(probs)))
            return self

        def depolarize2(self, q0: int, q1: int, p: float) -> "Circuit":
            return self.pauli_channel_2(q0, q1, tuple([p / 15] * 15))

        def damp(
            self, qubits: int | tuple[int, ...], t: float, T1: float, T2: float
        ) -> "Circuit":
            self._ops.append(DampOp(qubits=_to_qubits(qubits), t=t, T1=T1, T2=T2))
            return self

        def add_idle_exact_before_measurement(
            self, t: float, T1: float, T2: float
        ) -> "Circuit":
            new_circuit = Circuit(self.n_qubits)
            all_qubits = tuple(range(self.n_qubits))
            prev_was_measure = False
            for op in self._ops:
                if isinstance(op, MeasureOp):
                    if not prev_was_measure:
                        new_circuit._ops.append(
                            DampOp(qubits=all_qubits, t=t, T1=T1, T2=T2)
                        )
                    prev_was_measure = True
                else:
                    prev_was_measure = False
                new_circuit._ops.append(op)
            return new_circuit

        @classmethod
        def from_stim(
            cls, stim_circuit: stim.Circuit, n_qubits: int | None = None
        ) -> "Circuit":
            if n_qubits is None:
                n_qubits = stim_circuit.num_qubits or 1
            circuit = cls(n_qubits)
            circuit._extend_from_stim(stim_circuit)
            return circuit

        def _extend_from_stim(self, sc: stim.Circuit) -> None:
            for instr in sc:
                if isinstance(instr, stim.CircuitRepeatBlock):
                    for _ in range(instr.repeat_count):
                        self._extend_from_stim(instr.body_copy())
                    continue
                name: str = instr.name
                targets = [t for t in instr.targets_copy() if t.is_qubit_target]
                args: list[float] = instr.gate_args_copy()

                if name in _STIM_1Q_CLIFFORD:
                    self._ops.append(
                        CliffordOp(
                            _STIM_1Q_CLIFFORD[name], tuple(t.value for t in targets)
                        )
                    )
                elif name in _STIM_2Q_CLIFFORD:
                    self._ops.append(
                        CliffordOp(
                            _STIM_2Q_CLIFFORD[name], tuple(t.value for t in targets)
                        )
                    )
                elif name == "PAULI_CHANNEL_1":
                    self._ops.append(
                        PauliNoiseOp(
                            tuple(t.value for t in targets), args[0], args[1], args[2]
                        )
                    )
                elif name == "DEPOLARIZE1":
                    p = args[0]
                    self._ops.append(
                        PauliNoiseOp(
                            tuple(t.value for t in targets), p / 3, p / 3, p / 3
                        )
                    )
                elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                    p = args[0]
                    self._ops.append(
                        PauliNoiseOp(
                            tuple(t.value for t in targets),
                            p if name == "X_ERROR" else 0.0,
                            p if name == "Y_ERROR" else 0.0,
                            p if name == "Z_ERROR" else 0.0,
                        )
                    )
                elif name == "PAULI_CHANNEL_2":
                    probs = tuple(args)
                    for i in range(0, len(targets), 2):
                        self._ops.append(
                            TwoQubitPauliNoiseOp(
                                targets[i].value, targets[i + 1].value, probs
                            )
                        )
                elif name == "DEPOLARIZE2":
                    probs = tuple([args[0] / 15] * 15)
                    for i in range(0, len(targets), 2):
                        self._ops.append(
                            TwoQubitPauliNoiseOp(
                                targets[i].value, targets[i + 1].value, probs
                            )
                        )
                elif name in _MEAS_MAP:
                    basis, reset = _MEAS_MAP[name]
                    flip_prob = args[0] if args else 0.0
                    for t in targets:
                        self._ops.append(
                            MeasureOp(
                                qubits=(t.value,),
                                basis=basis,
                                reset=reset,
                                flip_probability=flip_prob,
                            )
                        )
                elif name == "TICK":
                    self._ops.append(TickOp())
                elif name in _STIM_SKIP:
                    pass
                else:
                    warnings.warn(
                        f"Circuit.from_stim: unsupported instruction '{name}' skipped.",
                        stacklevel=3,
                    )

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
                1
                for op in self._ops
                if isinstance(op, (PauliNoiseOp, TwoQubitPauliNoiseOp))
            )

        @property
        def n_measurements(self) -> int:
            return sum(1 for op in self._ops if isinstance(op, MeasureOp))

        def __len__(self) -> int:
            return len(self._ops)

        @staticmethod
        def _op_to_str(op: "Op") -> str:
            if isinstance(op, TickOp):
                return "TICK"
            if isinstance(op, CliffordOp):
                targets = " ".join(str(t) for t in op.targets)
                name = "S_DAG" if op.name == "Sdg" else op.name
                return f"{name} {targets}"
            if isinstance(op, RZOp):
                return f"RZ({op.theta:.6g}) {' '.join(str(q) for q in op.qubits)}"
            if isinstance(op, PauliNoiseOp):
                qs = " ".join(str(q) for q in op.qubits)
                if abs(op.px - op.py) < 1e-12 and abs(op.py - op.pz) < 1e-12:
                    return f"DEPOLARIZE1({op.px * 3:.6g}) {qs}"
                return f"PAULI_CHANNEL_1({op.px:.6g},{op.py:.6g},{op.pz:.6g}) {qs}"
            if isinstance(op, TwoQubitPauliNoiseOp):
                p0 = op.probs[0]
                if all(abs(p - p0) < 1e-12 for p in op.probs):
                    return f"DEPOLARIZE2({p0 * 15:.6g}) {op.q0} {op.q1}"
                return f"PAULI_CHANNEL_2({','.join(f'{p:.6g}' for p in op.probs)}) {op.q0} {op.q1}"
            if isinstance(op, DampOp):
                qs = " ".join(str(q) for q in op.qubits)
                t1 = "inf" if op.T1 == float("inf") else f"{op.T1:.6g}"
                t2 = "inf" if op.T2 == float("inf") else f"{op.T2:.6g}"
                return f"DAMP(t={op.t:.6g},T1={t1},T2={t2}) {qs}"
            if isinstance(op, MeasureOp):
                tgts = " ".join(str(q) for q in op.qubits)
                bsuf = "" if op.basis == "Z" else op.basis
                n = ("MR" if op.reset else "M") + bsuf
                if op.flip_probability > 0:
                    return f"{n}({op.flip_probability:.6g}) {tgts}"
                return f"{n} {tgts}"
            return f"UNKNOWN({type(op).__name__})"

        def __str__(self) -> str:
            header = (
                f"Circuit(n_qubits={self.n_qubits}, clifford={self.n_clifford_gates}, "
                f"rz={self.n_rz_gates}, noise={self.n_noise_ops}, measurements={self.n_measurements})"
            )
            body = "\n".join(self._op_to_str(op) for op in self._ops)
            return f"{header}\n{body}" if body else header

        def __repr__(self) -> str:
            return str(self)


Op = Union[
    CliffordOp, RZOp, PauliNoiseOp, TwoQubitPauliNoiseOp, DampOp, MeasureOp, TickOp
]
