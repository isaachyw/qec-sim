"""
Pytest test suite for the QC_claude Monte Carlo stabilizer estimator.

Each test verifies that the MC estimator agrees with an analytically exact
value to within 4 standard errors (z < 4).  Seeds are fixed so failures are
deterministic, not flaky.

Run from stim-playground/:
    uv run pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from QC_claude import Circuit, PauliObservable, estimate, MCSampler

# ── Numpy gate matrices ───────────────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_H  = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.diag([1.0, -1.0]).astype(complex)
_S  = np.diag([1.0, 1j]).astype(complex)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def _RZ(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def _expect(state: np.ndarray, obs: np.ndarray) -> float:
    return float(np.real(state.conj() @ obs @ state))


def _check(result, exact: float, threshold: float = 4.0) -> None:
    """Assert that the MC estimate agrees with exact within `threshold` σ."""
    z = abs(result.value - exact) / (result.std_error + 1e-15)
    assert z < threshold, (
        f"z={z:.2f} ≥ {threshold}: mc={result.value:.6f} ± {result.std_error:.6f}, "
        f"exact={exact:.6f}, one_norm={result.one_norm:.4f}"
    )


# ── RZ decomposition tests ────────────────────────────────────────────────────


@pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2, np.pi / 8, 3 * np.pi / 4])
def test_hrz_h(theta: float):
    """H-RZ(theta)-H → <Z> = cos(theta)."""
    exact = float(np.cos(theta))
    c = Circuit(n_qubits=1)
    c.h(0).rz(0, theta).h(0)
    result = estimate(c, PauliObservable.single_z(1, 0), n_samples=50_000, seed=1)
    _check(result, exact)


def test_sdg_rz_h():
    """H-Sdg-RZ(pi/3)-H → <Z>, exact via numpy."""
    theta = np.pi / 3
    state = _H @ _RZ(theta) @ _S.conj().T @ _H @ np.array([1.0, 0.0])
    exact = _expect(state, _Z)
    c = Circuit(n_qubits=1)
    c.h(0).sdg(0).rz(0, theta).h(0)
    result = estimate(c, PauliObservable.single_z(1, 0), n_samples=50_000, seed=1)
    _check(result, exact)


def test_two_rz():
    """H-RZ(t1)-H-RZ(t2)-H → <Z>, exact via numpy."""
    t1, t2 = np.pi / 4, np.pi / 3
    state = _H @ _RZ(t2) @ _H @ _RZ(t1) @ _H @ np.array([1.0, 0.0])
    exact = _expect(state, _Z)
    c = Circuit(n_qubits=1)
    c.h(0).rz(0, t1).h(0).rz(0, t2).h(0)
    result = estimate(c, PauliObservable.single_z(1, 0), n_samples=50_000, seed=2)
    _check(result, exact)


def test_bell_rz():
    """Bell state + RZ on q0 + H → <ZI>, exact via numpy."""
    theta = np.pi / 4
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    state = np.kron(_H, _I2) @ state
    state = _CNOT @ state
    state = np.kron(_RZ(theta), _I2) @ state
    state = np.kron(_H, _I2) @ state
    exact = _expect(state, np.kron(_Z, _I2))

    c = Circuit(n_qubits=2)
    c.h(0).cx(0, 1).rz(0, theta).h(0)
    result = estimate(c, PauliObservable.single_z(2, 0), n_samples=50_000, seed=3)
    _check(result, exact)


def test_multi_term_observable():
    """H-RZ(pi/4)-H → <(Z+X)/sqrt(2)>."""
    theta = np.pi / 4
    state = _H @ _RZ(theta) @ _H @ np.array([1.0, 0.0])
    exact = _expect(state, (_Z + _X) / np.sqrt(2))

    c = Circuit(n_qubits=1)
    c.h(0).rz(0, theta).h(0)
    obs = PauliObservable(1, [("Z", 1.0 / np.sqrt(2)), ("X", 1.0 / np.sqrt(2))])
    result = estimate(c, obs, n_samples=50_000, seed=4)
    _check(result, exact)


@pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 9, endpoint=False).tolist())
def test_scan_angles(theta: float):
    """H-RZ(theta)-H → <Z> = cos(theta) for theta in [0, 2pi)."""
    c = Circuit(n_qubits=1)
    c.h(0).rz(0, theta).h(0)
    result = estimate(c, PauliObservable.single_z(1, 0), n_samples=30_000, seed=6)
    _check(result, float(np.cos(theta)), threshold=5.0)


# ── Clifford-only test ────────────────────────────────────────────────────────


def test_clifford_only():
    """Bell state |Phi+>: <ZZ> = 1, one_norm = 1."""
    c = Circuit(n_qubits=2)
    c.h(0).cx(0, 1)
    result = estimate(c, PauliObservable(2, [("ZZ", 1.0)]), n_samples=10_000, seed=5)
    _check(result, 1.0)
    assert abs(result.one_norm - 1.0) < 1e-9


# ── Pauli noise tests ─────────────────────────────────────────────────────────


def test_depolarize1():
    """Bell + DEPOLARIZE1(p) on both qubits → <ZZ> = (1-4p/3)^2."""
    p = 0.1
    exact = (1.0 - 4 * p / 3) ** 2
    c = Circuit(n_qubits=2)
    c.h(0).cx(0, 1).depolarize1(0, p).depolarize1(1, p)
    result = estimate(c, PauliObservable(2, [("ZZ", 1.0)]), n_samples=100_000, seed=7)
    _check(result, exact)
    assert abs(result.one_norm - 1.0) < 1e-9  # noise 1-norm = 1


def test_from_stim_depolarize():
    """from_stim: Bell + DEPOLARIZE1 → <ZZ> = (1-4p/3)^2."""
    p = 0.05
    exact = (1.0 - 4 * p / 3) ** 2
    sc = stim.Circuit(f"H 0\nCX 0 1\nDEPOLARIZE1({p}) 0 1")
    c = Circuit.from_stim(sc)
    result = estimate(c, PauliObservable(2, [("ZZ", 1.0)]), n_samples=100_000, seed=8)
    _check(result, exact)


def test_from_stim_rz():
    """from_stim: H-DEPOL(p)-RZ(theta)-H → <Z>, exact via density matrix."""
    theta, p = np.pi / 4, 0.02
    rho = np.array([[1.0, 0.0], [0.0, 0.0]])
    rho = _H @ rho @ _H
    rho = (1 - 4 * p / 3) * rho + (4 * p / 3) * (_I2 / 2)
    RZm = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
    rho = RZm @ rho @ RZm.conj().T
    rho = _H @ rho @ _H
    exact = float(np.real(np.trace(_Z @ rho)))

    sc = stim.Circuit(f"H 0\nDEPOLARIZE1({p}) 0")
    c = Circuit.from_stim(sc)
    c.rz(0, theta).h(0)
    result = estimate(c, PauliObservable.single_z(1, 0), n_samples=100_000, seed=9)
    _check(result, exact)


def test_depolarize2():
    """Bell + DEPOLARIZE2(p) → <ZZ>, exact via density matrix."""
    p = 0.05
    paulis = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}
    labels = ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
              "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ")

    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    rho = np.kron(_H, _I2) @ rho @ np.kron(_H, _I2)
    rho = _CNOT @ rho @ _CNOT

    rho_new = (1 - p) * rho
    for label in labels:
        P = np.kron(paulis[label[0]], paulis[label[1]])
        rho_new += (p / 15) * P @ rho @ P.conj().T
    exact = float(np.real(np.trace(np.kron(_Z, _Z) @ rho_new)))

    sc = stim.Circuit(f"H 0\nCX 0 1\nDEPOLARIZE2({p}) 0 1")
    c = Circuit.from_stim(sc)
    result = estimate(c, PauliObservable(2, [("ZZ", 1.0)]), n_samples=100_000, seed=10)
    _check(result, exact)


# ── MCSampler tests ───────────────────────────────────────────────────────────


def test_sampler_memory_experiment():
    """
    MCSampler on 3-qubit repetition code: 2 syndrome rounds + data readout.
    Validates shapes, dtypes, weight unity (no RZ → all weights = 1), and one_norm.
    """
    p = 0.05
    sc = stim.Circuit(f"""
H 3 4
CX 3 0
CX 3 1
CX 4 1
CX 4 2
H 3 4
DEPOLARIZE1({p}) 0 1 2 3 4
MR 3 4
H 3 4
CX 3 0
CX 3 1
CX 4 1
CX 4 2
H 3 4
DEPOLARIZE1({p}) 0 1 2 3 4
MR 3 4
M 0 1 2
""")
    c = Circuit.from_stim(sc)
    result = MCSampler(c).sample(n_samples=2_000, seed=11)

    assert result.measurements.shape == (2_000, 7)
    assert result.weights.shape == (2_000,)
    assert result.measurements.dtype == np.uint8
    assert np.all((result.measurements == 0) | (result.measurements == 1))
    assert np.allclose(result.weights, 1.0), "All weights should be 1.0 for Clifford+noise"
    assert abs(result.one_norm - 1.0) < 1e-9


def test_sampler_requires_measurements():
    """MCSampler rejects circuits with no measurements."""
    c = Circuit(n_qubits=1)
    c.h(0)
    with pytest.raises(ValueError, match="MeasureOp"):
        MCSampler(c)


# ── DampOp tests ──────────────────────────────────────────────────────────────


def test_damp_pure_dephasing():
    """
    Pure dephasing: T1=inf, T2 finite.
    Prepare |+>, apply DampOp, measure <X>.
    Expected: <X> = exp(-t/T2)  (Tφ = T2 when T1=inf).
    one_norm should be 1.0 (T1=inf → T1 ≥ T2 threshold trivially satisfied).
    """
    t, T2 = 1.0, 2.0
    exact = float(np.exp(-t / T2))

    c = Circuit(n_qubits=1)
    c.h(0).damp(0, t=t, T1=float("inf"), T2=T2)
    result = estimate(c, PauliObservable(1, [("X", 1.0)]), n_samples=50_000, seed=12)
    _check(result, exact)
    assert abs(result.one_norm - 1.0) < 1e-9


def test_damp_pure_amplitude_decay():
    """
    Pure amplitude decay: T2=inf, T1 finite.
    Prepare |+>, apply DampOp, measure <X>.
    Expected: <X> = exp(-t/(2*T1))  [= sqrt(1-gamma), off-diagonal decay].
    """
    t, T1 = 1.0, 1.0
    exact = float(np.exp(-t / (2.0 * T1)))

    c = Circuit(n_qubits=1)
    c.h(0).damp(0, t=t, T1=T1, T2=float("inf"))
    result = estimate(c, PauliObservable(1, [("X", 1.0)]), n_samples=50_000, seed=13)
    _check(result, exact)


def test_damp_exact_regime():
    """
    T1 >= T2: exact probability regime, 1-norm = 1.0.
    Use T1=2.0, T2=1.0 (well inside T1 > T2).
    """
    t, T1, T2 = 1.0, 2.0, 1.0
    # Compute exact <Z> after DampOp on |0>:
    # starting from |0>, the amplitude decay can excite |1>? No — amplitude damping
    # only decays |1>->|0>, not excite. With pure state |0>, gamma*Reset = |0>
    # and Z|0> = |0> terms all give <Z>=+1.
    # Actually with |0> as input: all three terms (I, Z, Reset) leave state as |0>.
    # So <Z> = 1 regardless of parameters (for |0> input).
    # Better to use |+> → <Z> = 0 always, or X|0>=|1> input:
    # After DampOp on |1>:  <Z>_final = (c0+c1)*(-1) + gamma*(+1) = -(1-gamma)+gamma = 2gamma-1
    import math
    inv_t_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
    t_phi = 1.0 / inv_t_phi
    gamma = 1.0 - math.exp(-t / T1)
    lam = 1.0 - math.exp(-t / t_phi)
    # <X> after DampOp on |+>: = sqrt(1-gamma) * exp(-t/t_phi) ... actually:
    # full formula: <X>_final = (c0_tot - c1_tot) where
    # c0_tot = (1-p)*c0 + p*c1, c1_tot = p*c0 + (1-p)*c1
    # c0_tot - c1_tot = (1-2p)*(c0-c1) = (1-lam)*sqrt(1-gamma)
    exact = float((1.0 - lam) * math.sqrt(1.0 - gamma))

    c = Circuit(n_qubits=1)
    c.h(0).damp(0, t=t, T1=T1, T2=T2)
    result = estimate(c, PauliObservable(1, [("X", 1.0)]), n_samples=50_000, seed=14)
    _check(result, exact)
    assert result.one_norm <= 1.0 + 1e-9, f"one_norm={result.one_norm:.6f} should be ≤ 1 in exact regime"


def test_damp_quasiprobability_regime():
    """
    T1 < T2 <= 2*T1: quasiprobability regime, 1-norm > 1.0.
    Use T1=1.0, T2=1.5 (so T1 < T2 < 2*T1).
    """
    import math
    t, T1, T2 = 1.0, 1.0, 1.5
    inv_t_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
    t_phi = 1.0 / inv_t_phi
    gamma = 1.0 - math.exp(-t / T1)
    lam = 1.0 - math.exp(-t / t_phi)
    exact = float((1.0 - lam) * math.sqrt(1.0 - gamma))

    c = Circuit(n_qubits=1)
    c.h(0).damp(0, t=t, T1=T1, T2=T2)
    result = estimate(c, PauliObservable(1, [("X", 1.0)]), n_samples=100_000, seed=15)
    _check(result, exact)
    assert result.one_norm > 1.0, f"one_norm={result.one_norm:.6f} should be > 1 in quasiprobability regime"


def test_damp_invalid_t2():
    """T2 >= 2*T1 raises ValueError."""
    c = Circuit(n_qubits=1)
    c.damp(0, t=1.0, T1=1.0, T2=3.0)  # T2 > 2*T1: invalid
    with pytest.raises(ValueError, match="T2"):
        estimate(c, PauliObservable(1, [("Z", 1.0)]), n_samples=10)


def test_estimator_rejects_measurements():
    """MCEstimator raises ValueError when circuit contains MeasureOp."""
    c = Circuit(n_qubits=1)
    c.h(0).measure(0)
    with pytest.raises(ValueError, match="MeasureOp"):
        estimate(c, PauliObservable.single_z(1, 0), n_samples=10)
