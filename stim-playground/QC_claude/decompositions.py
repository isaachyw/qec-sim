"""
Quasiprobability decompositions of non-Clifford gates and noise channels into
stabilizer channels.

── RZ(theta) channel decomposition ─────────────────────────────────────────────

    chi[RZ_theta](rho) = a * I(rho)  +  b * Z(rho)  +  c * S(rho)

    where
        I(rho) = rho              (identity channel)
        Z(rho) = Z rho Z†         (Pauli-Z channel)
        S(rho) = S rho S†         (phase gate channel)

    and coefficients:
        a = (1 + cos(theta) - sin(theta)) / 2
        b = (1 - cos(theta) - sin(theta)) / 2
        c = sin(theta)

    Verification: diagonal preserved (a+b+c = 1), off-diagonal picks up
    correct phase factor a-b-ic = e^{-i*theta}.

    Special cases:
        theta = 0    → a=1, b=0, c=0  (identity)
        theta = pi/2 → a=0, b=0, c=1  (pure S; RZ(pi/2) ~ S)
        theta = pi   → a=0, b=1, c=0  (pure Z; RZ(pi) ~ Z)
        theta = pi/4 → 1-norm = sqrt(2)  (T gate)

── Noise channel decompositions ─────────────────────────────────────────────────

    Single-qubit Pauli noise:
        chi(rho) = (1-px-py-pz)*I + px*X + py*Y + pz*Z
    All coefficients >= 0 (true probabilities), so 1-norm = 1 (no MC overhead).

    Two-qubit Pauli noise:
        chi(rho) = (1-sum_p)*II + sum_{P in _PAULI2_LABELS} p_P * P
    where P ∈ {IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ}.

    Coefficients store in _PAULI2_LABELS order (the standard Stim PAULI_CHANNEL_2 order).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# Two-qubit Pauli label order (matches Stim's PAULI_CHANNEL_2 parameter order)
_PAULI2_LABELS: tuple[str, ...] = (
    'IX', 'IY', 'IZ',
    'XI', 'XX', 'XY', 'XZ',
    'YI', 'YX', 'YY', 'YZ',
    'ZI', 'ZX', 'ZY', 'ZZ',
)


@dataclass(frozen=True)
class CliffordTerm:
    """A single stabilizer (Clifford) channel term in a decomposition."""
    coefficient: complex
    gate: str  # 1q: 'I','X','Y','Z','S'  |  2q: 'II','IX','XY', etc.

    @property
    def abs_coeff(self) -> float:
        return abs(self.coefficient)


@dataclass
class GateDecomposition:
    """
    A quasiprobability (or probability) decomposition into stabilizer channels.

        channel = sum_i terms[i].coefficient * stabilizer_channel_i

    For RZ gates the coefficients may be negative (quasi-probability).
    For noise channels all coefficients are >= 0 (true probability); 1-norm = 1.

    Attributes:
        qubits: Target qubit indices (length 1 for single-qubit, 2 for two-qubit).
        terms:  List of CliffordTerms.
    """
    qubits: tuple[int, ...]
    terms: list[CliffordTerm]

    @property
    def one_norm(self) -> float:
        """Sum of |coefficient| values (gamma factor; variance scales as gamma^2)."""
        return sum(t.abs_coeff for t in self.terms)

    def probabilities(self) -> list[float]:
        """Importance-sampling probabilities proportional to |coefficient|."""
        norms = [t.abs_coeff for t in self.terms]
        total = sum(norms)
        return [n / total for n in norms]

    def sample(self, rng: np.random.Generator) -> tuple[complex, float, str]:
        """
        Sample one Clifford term for Monte Carlo.

        Returns:
            (coefficient, sampling_probability, gate_label)
        """
        probs = self.probabilities()
        idx = int(rng.choice(len(self.terms), p=probs))
        term = self.terms[idx]
        return term.coefficient, probs[idx], term.gate


# ── RZ decomposition ──────────────────────────────────────────────────────────

def rz_decomposition(qubit: int, theta: float) -> GateDecomposition:
    """
    Quasiprobability decomposition of the RZ(theta) channel.

        chi[RZ_theta] = a * I_channel + b * Z_channel + c * S_channel

    Args:
        qubit: Target qubit index.
        theta: Rotation angle in radians.

    Returns:
        GateDecomposition with three terms (I, Z, S); b or c may be negative.
    """
    a = (1.0 + np.cos(theta) - np.sin(theta)) / 2.0  # I coefficient
    b = (1.0 - np.cos(theta) - np.sin(theta)) / 2.0  # Z coefficient
    c = float(np.sin(theta))                           # S coefficient
    return GateDecomposition(
        qubits=(qubit,),
        terms=[
            CliffordTerm(coefficient=complex(a), gate='I'),
            CliffordTerm(coefficient=complex(b), gate='Z'),
            CliffordTerm(coefficient=complex(c), gate='S'),
        ],
    )


# ── Noise decompositions ──────────────────────────────────────────────────────

def pauli_noise_decomp_1q(qubit: int, px: float, py: float, pz: float) -> GateDecomposition:
    """
    Decomposition for a single-qubit Pauli noise channel.

        chi(rho) = (1-px-py-pz)*I + px*X + py*Y + pz*Z

    All coefficients are non-negative; 1-norm = 1 (no variance overhead).

    Args:
        qubit: Target qubit.
        px, py, pz: Single-qubit Pauli error probabilities.
    """
    pi = 1.0 - px - py - pz
    return GateDecomposition(
        qubits=(qubit,),
        terms=[
            CliffordTerm(coefficient=complex(pi), gate='I'),
            CliffordTerm(coefficient=complex(px), gate='X'),
            CliffordTerm(coefficient=complex(py), gate='Y'),
            CliffordTerm(coefficient=complex(pz), gate='Z'),
        ],
    )


def pauli_noise_decomp_2q(
    q0: int, q1: int, probs: tuple[float, ...]
) -> GateDecomposition:
    """
    Decomposition for a two-qubit Pauli noise channel.

    The 15 non-II Pauli probabilities are in _PAULI2_LABELS order
    (matching Stim's PAULI_CHANNEL_2 convention):
        IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ

    Args:
        q0, q1: Target qubit pair.
        probs:  15-tuple of Pauli error probabilities (for all non-II Paulis).
    """
    if len(probs) != 15:
        raise ValueError(f"Expected 15 probabilities, got {len(probs)}")
    pi = 1.0 - sum(probs)
    terms = [CliffordTerm(coefficient=complex(pi), gate='II')]
    for label, p in zip(_PAULI2_LABELS, probs):
        terms.append(CliffordTerm(coefficient=complex(p), gate=label))
    return GateDecomposition(qubits=(q0, q1), terms=terms)
