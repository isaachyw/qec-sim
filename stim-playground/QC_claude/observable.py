"""
Hermitian observable as a linear combination of Pauli strings.

The observable is decomposed as:
    phi = sum_i q_obs[i] * P_i

where P_i are n-qubit Pauli strings (e.g. 'ZII', 'IXZ') and q_obs[i] are
real or complex coefficients.

For a stabilizer state, each Pauli expectation <P_i> is exactly +1, -1, or 0,
and can be computed analytically by Stim without sampling.
"""

from __future__ import annotations

import numpy as np
import stim


class PauliObservable:
    """
    A Hermitian observable expressed as a weighted sum of Pauli strings.

        phi = sum_i coefficients[i] * PauliString[i]

    Pauli strings use the convention:
        - Characters: 'I', 'X', 'Y', 'Z' (one per qubit, qubit 0 is leftmost)
        - Example: 'ZIX' means Z on qubit 0, I on qubit 1, X on qubit 2.

    Args:
        n_qubits: Number of qubits in the system.
        terms: List of (pauli_string, coefficient) pairs.

    Example:
        >>> obs = PauliObservable(2, [('ZI', 1.0), ('IZ', 1.0)])  # Z0 + Z1
        >>> obs = PauliObservable.single_z(n_qubits=3, qubit=0)   # Z on qubit 0
    """

    def __init__(self, n_qubits: int, terms: list[tuple[str, complex]]) -> None:
        self.n_qubits = n_qubits
        # Normalise: store as list[(pauli_string, coeff)]
        self._terms: list[tuple[stim.PauliString, complex]] = []
        for pauli_str, coeff in terms:
            ps = self._parse(pauli_str)
            self._terms.append((ps, complex(coeff)))

    def _parse(self, pauli_str: str) -> stim.PauliString:
        """Parse a Pauli string, padding with I to match n_qubits."""
        if len(pauli_str) > self.n_qubits:
            raise ValueError(
                f"Pauli string '{pauli_str}' has length {len(pauli_str)} "
                f"but circuit has {self.n_qubits} qubits."
            )
        padded = pauli_str.ljust(self.n_qubits, "I")
        return stim.PauliString(padded)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def single_pauli(
        cls, n_qubits: int, qubit: int, pauli: str, coefficient: complex = 1.0
    ) -> "PauliObservable":
        """
        Observable consisting of a single Pauli operator on one qubit.

        Example: single_pauli(3, 1, 'Z') → IZI
        """
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"pauli must be 'X', 'Y', or 'Z', got '{pauli}'")
        s = "I" * qubit + pauli + "I" * (n_qubits - qubit - 1)
        return cls(n_qubits, [(s, coefficient)])

    @classmethod
    def single_z(cls, n_qubits: int, qubit: int) -> "PauliObservable":
        """Z operator on a single qubit: observable = Z_qubit."""
        return cls.single_pauli(n_qubits, qubit, "Z")

    @classmethod
    def single_x(cls, n_qubits: int, qubit: int) -> "PauliObservable":
        """X operator on a single qubit."""
        return cls.single_pauli(n_qubits, qubit, "X")

    @classmethod
    def single_y(cls, n_qubits: int, qubit: int) -> "PauliObservable":
        """Y operator on a single qubit."""
        return cls.single_pauli(n_qubits, qubit, "Y")

    @classmethod
    def pauli_sum(
        cls, n_qubits: int, terms: list[tuple[str, complex]]
    ) -> "PauliObservable":
        """
        General Pauli sum.

        Args:
            n_qubits: System size.
            terms: [(pauli_string, coefficient), ...].

        Example:
            >>> obs = PauliObservable.pauli_sum(2, [('ZI', 0.5), ('IZ', 0.5)])
        """
        return cls(n_qubits, terms)

    # ── Expectation value evaluation ──────────────────────────────────────────

    def expectation(self, sim: stim.TableauSimulator) -> complex:
        """
        Compute the exact expectation value of this observable on the current
        stabilizer state held by `sim`.

        For each Pauli term P_i, stim.peek_observable_expectation returns
        +1 if P_i is in the stabilizer, -1 if -P_i is a stabilizer, else 0.

        Returns:
            sum_i coeff_i * <P_i>  (should be real for Hermitian observables)
        """
        total = 0.0 + 0j
        for ps, coeff in self._terms:
            ev = sim.peek_observable_expectation(ps)
            total += coeff * ev
        return total

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def terms(self) -> list[tuple[stim.PauliString, complex]]:
        return list(self._terms)

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    def one_norm(self) -> float:
        """Sum of |coefficient| values."""
        return sum(abs(c) for _, c in self._terms)

    def __repr__(self) -> str:
        parts = [f"{c:+.3f}*{ps}" for ps, c in self._terms]
        return f"PauliObservable({' '.join(parts)})"
