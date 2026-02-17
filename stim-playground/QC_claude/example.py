"""
Validation examples for the Monte Carlo stabilizer estimator.

Each example computes an expectation value via:
  1. Exact numpy state-vector simulation (ground truth).
  2. MCEstimator (Monte Carlo with stabilizer channel decomposition).

All circuits start from |0...0>.

Run with:
    cd stim-playground
    uv run python QC_claude/example.py
"""

from __future__ import annotations

import sys
import os
import numpy as np

# ── Make the package importable when run as a script ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stim
from QC_claude import Circuit, PauliObservable, estimate, MCSampler


# ── Exact numpy utilities ─────────────────────────────────────────────────────

def _exact_expectation(state: np.ndarray, obs: np.ndarray) -> float:
    """<psi|obs|psi> for a state vector and matrix observable."""
    return float(np.real(state.conj() @ obs @ state))


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1., -1.]).astype(complex)
_S = np.diag([1., 1j]).astype(complex)


def _RZ(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


# ── Formatting ────────────────────────────────────────────────────────────────

def print_result(name: str, exact: float, mc_result) -> None:
    z = abs(mc_result.value - exact) / (mc_result.std_error + 1e-15)
    status = "PASS" if z < 4 else "FAIL"
    print(f"[{status}] {name}")
    print(f"       exact={exact:+.6f}  mc={mc_result.value:+.6f} ± {mc_result.std_error:.6f}"
          f"  z={z:.1f}  γ={mc_result.one_norm:.4f}")
    print()


# ── Examples ──────────────────────────────────────────────────────────────────

def example_hrz_h(theta: float = np.pi / 4, n_samples: int = 50_000, seed: int = 0):
    """
    Circuit: |0> -H- RZ(theta) -H-  measure <Z>.

    H RZ(theta) H = RX(theta) so <Z> = cos(theta) on |0>.
    """
    # Exact
    state = _H @ _RZ(theta) @ _H @ np.array([1., 0.])
    exact = _exact_expectation(state, _Z)

    c = Circuit(n_qubits=1)
    c.h(0).rz(0, theta).h(0)
    obs = PauliObservable.single_z(n_qubits=1, qubit=0)
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(f"H-RZ({theta:.4f})-H → <Z>  (expect cos={np.cos(theta):.4f})", exact, result)
    return result


def example_sdg_rz_h(theta: float = np.pi / 3, n_samples: int = 50_000, seed: int = 1):
    """
    Circuit: |0> -H- S† -RZ(theta)- H-  measure <Z>.

    Prepares a less trivial initial condition for the estimator.
    """
    state = _H @ _RZ(theta) @ _S.conj().T @ _H @ np.array([1., 0.])
    exact = _exact_expectation(state, _Z)

    c = Circuit(n_qubits=1)
    c.h(0).sdg(0).rz(0, theta).h(0)
    obs = PauliObservable.single_z(n_qubits=1, qubit=0)
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(f"H-Sdg-RZ({theta:.4f})-H → <Z>", exact, result)
    return result


def example_two_rz(
    theta1: float = np.pi / 4,
    theta2: float = np.pi / 3,
    n_samples: int = 50_000,
    seed: int = 2,
):
    """
    Circuit: |0> -H- RZ(t1) -H- RZ(t2) -H-  measure <Z>.

    Two non-Clifford gates; 1-norm = gamma(t1) * gamma(t2).
    """
    state = _H @ _RZ(theta2) @ _H @ _RZ(theta1) @ _H @ np.array([1., 0.])
    exact = _exact_expectation(state, _Z)

    c = Circuit(n_qubits=1)
    c.h(0).rz(0, theta1).h(0).rz(0, theta2).h(0)
    obs = PauliObservable.single_z(n_qubits=1, qubit=0)
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(
        f"H-RZ({theta1:.3f})-H-RZ({theta2:.3f})-H → <Z>  (expect sin(t1)*sin(t2)={np.sin(theta1)*np.sin(theta2):.4f})",
        exact, result,
    )
    return result


def example_bell_rz(theta: float = np.pi / 4, n_samples: int = 50_000, seed: int = 3):
    """
    Circuit: |00> -H(0)- CX(0,1)- RZ(0, theta)- H(0)-  measure <ZI>.

    Bell state preparation followed by a non-Clifford rotation on qubit 0.
    """
    I2 = np.eye(2, dtype=complex)
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

    state = np.array([1., 0., 0., 0.], dtype=complex)
    state = np.kron(_H, I2) @ state
    state = CX @ state
    state = np.kron(_RZ(theta), I2) @ state
    state = np.kron(_H, I2) @ state
    ZI = np.kron(_Z, I2)
    exact = _exact_expectation(state, ZI)

    c = Circuit(n_qubits=2)
    c.h(0).cx(0, 1).rz(0, theta).h(0)
    obs = PauliObservable.single_z(n_qubits=2, qubit=0)
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(f"Bell + RZ({theta:.4f}) on q0 + H → <ZI>", exact, result)
    return result


def example_pauli_sum_obs(n_samples: int = 50_000, seed: int = 4):
    """
    Multi-term observable: phi = (Z + X) / sqrt(2).

    Circuit: |0> -H- RZ(pi/4) -H-  measure (Z + X)/sqrt(2).
    """
    theta = np.pi / 4
    state = _H @ _RZ(theta) @ _H @ np.array([1., 0.])
    obs_mat = (_Z + _X) / np.sqrt(2)
    exact = _exact_expectation(state, obs_mat)

    c = Circuit(n_qubits=1)
    c.h(0).rz(0, theta).h(0)
    obs = PauliObservable(n_qubits=1, terms=[('Z', 1.0/np.sqrt(2)), ('X', 1.0/np.sqrt(2))])
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result("H-RZ(pi/4)-H → <(Z+X)/sqrt(2)>", exact, result)
    return result


def example_clifford_only(n_samples: int = 10_000, seed: int = 5):
    """
    Clifford-only circuit: 1-norm should be exactly 1 (no overhead).

    Circuit: |00> -H(0)- CX(0,1)-  measure <ZZ>.
    Bell state |Phi+>: <ZZ> = 1.
    """
    c = Circuit(n_qubits=2)
    c.h(0).cx(0, 1)
    obs = PauliObservable(n_qubits=2, terms=[('ZZ', 1.0)])
    result = estimate(c, obs, n_samples=n_samples, seed=seed)
    print_result("Clifford-only Bell → <ZZ>", exact=1.0, mc_result=result)
    return result


def example_scan_angles(n_samples: int = 30_000, seed: int = 6):
    """
    Scan theta ∈ [0, 2π] for H-RZ(theta)-H → <Z> = cos(theta).
    """
    print("Scanning θ ∈ [0, 2π]  for  H-RZ(θ)-H → <Z> = cos(θ):")
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    all_pass = True
    for theta in angles:
        exact = np.cos(theta)
        c = Circuit(n_qubits=1)
        c.h(0).rz(0, theta).h(0)
        obs = PauliObservable.single_z(n_qubits=1, qubit=0)
        r = estimate(c, obs, n_samples=n_samples, seed=seed)
        z = abs(r.value - exact) / (r.std_error + 1e-15)
        ok = z < 4
        all_pass = all_pass and ok
        print(
            f"  {'OK  ' if ok else 'FAIL'}  θ={theta:.4f}"
            f"  exact={exact:+.4f}  mc={r.value:+.4f}±{r.std_error:.4f}"
            f"  z={z:.1f}  γ={r.one_norm:.3f}"
        )
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}\n")


def example_depolarize1(p: float = 0.1, n_samples: int = 100_000, seed: int = 7):
    """
    Noisy Bell state via Circuit builder API.

    Circuit: |00> -H(0)- CX(0,1)- DEPOLARIZE1(p)(0)- DEPOLARIZE1(p)(1)-  <ZZ>

    Analytical exact: each depolarizing channel reduces <ZZ> by factor (1 - 4p/3),
    applied independently to both qubits:
        <ZZ> = (1 - 4p/3)^2
    """
    exact = (1.0 - 4 * p / 3) ** 2

    c = Circuit(n_qubits=2)
    c.h(0).cx(0, 1).depolarize1(0, p).depolarize1(1, p)
    obs = PauliObservable(n_qubits=2, terms=[('ZZ', 1.0)])
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(
        f"Bell + DEPOLARIZE1(p={p}) → <ZZ>  (expect (1-4p/3)^2={exact:.4f})",
        exact, result,
    )
    return result


def example_from_stim_depolarize(p: float = 0.05, n_samples: int = 100_000, seed: int = 8):
    """
    Noisy Bell state imported from a Stim circuit via Circuit.from_stim().

    Stim circuit: H 0 / CX 0 1 / DEPOLARIZE1(p) 0 1
    Observable: <ZZ> — same as example_depolarize1 but built via from_stim.
    """
    exact = (1.0 - 4 * p / 3) ** 2

    sc = stim.Circuit(f"""
H 0
CX 0 1
DEPOLARIZE1({p}) 0 1
""")
    c = Circuit.from_stim(sc)
    obs = PauliObservable(n_qubits=2, terms=[('ZZ', 1.0)])
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(
        f"from_stim Bell + DEPOLARIZE1(p={p}) → <ZZ>",
        exact, result,
    )
    return result


def example_from_stim_rz(theta: float = np.pi / 4, p: float = 0.02,
                          n_samples: int = 100_000, seed: int = 9):
    """
    Stim circuit (Clifford + noise) extended with a non-Clifford RZ gate.

    Circuit: |0> -H- DEPOLARIZE1(p)- RZ(theta)- H-  <Z>
    Exact: depolarizing shrinks the Bloch vector by (1-4p/3) before the RZ rotation.
        Starting from |0>: after H → Bloch vector (1, 0, 0) [on X axis]
        After DEPOLARIZE1(p): Bloch vector shrinks to (1-4p/3, 0, 0)
        After RZ(theta): (1-4p/3) rotates in XZ plane → Bloch Z = (1-4p/3)*cos(theta)
        After H: swaps X↔Z so measured Z = (1-4p/3)*cos(theta)  ...but wait, H
            conjugates: H X H = Z, H Z H = X, so H rotates the Bloch sphere.
        Let's just use numpy for the exact value.
    """
    # Exact via density matrix
    I2 = np.eye(2)
    rho = np.array([[1., 0.], [0., 0.]])  # |0><0|
    H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    rho = H2 @ rho @ H2  # after H: |+><+|

    # Depolarizing channel: rho → (1-4p/3)*rho + (4p/3)*(I/2)
    rho = (1 - 4 * p / 3) * rho + (4 * p / 3) * (I2 / 2)

    # RZ channel: rho → RZ rho RZ†
    RZm = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
    rho = RZm @ rho @ RZm.conj().T

    # Final H
    rho = H2 @ rho @ H2
    exact = float(np.real(np.trace(np.diag([1., -1.]) @ rho)))

    sc = stim.Circuit(f"H 0\nDEPOLARIZE1({p}) 0")
    c = Circuit.from_stim(sc)
    c.rz(0, theta).h(0)
    obs = PauliObservable.single_z(n_qubits=1, qubit=0)
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(
        f"from_stim H-DEPOL({p})-RZ({theta:.3f})-H → <Z>",
        exact, result,
    )
    return result


def example_pauli_channel_2(n_samples: int = 100_000, seed: int = 10):
    """
    Two-qubit Pauli noise channel via DEPOLARIZE2.

    Circuit: |00> -H(0)- CX(0,1)- DEPOLARIZE2(p)(0,1)-  <ZZ>

    For DEPOLARIZE2(p), <ZZ> on Bell state |Phi+>:
        Each of the 15 non-II Paulis flips ZZ or not.
        ZZ-flipping Paulis (those anticommuting with ZZ = ZI or IZ part):
            IX, IY, XI, YI, XX, XY, YX, YY → ZZ → -ZZ  (8 terms)
        ZZ-preserving Paulis (commuting with ZZ):
            IZ, XZ, YZ, ZI, ZX, ZY, ZZ → ZZ → +ZZ  (7 terms)
        Wait: let's count by computing P ZZ P† for each P.

        Actually: ZZ commutes with P iff [ZZ, P] = 0.
        ZZ = Z⊗Z. P anticommutes with ZZ iff X or Y appears on an odd number of positions
        (where Z is transparent to Z). But we need to be more careful.

        For now just use numpy for the exact value.
    """
    p = 0.05
    # Exact via density matrix
    I2 = np.eye(2)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    H4 = np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), I2)
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    rho = H4 @ rho @ H4
    rho = CNOT @ rho @ CNOT

    # DEPOLARIZE2(p): apply each of 15 Paulis with prob p/15, identity with prob 1-p
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.diag([1.,-1.])
    paulis_1q = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
    labels = ('IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ')
    rho_new = (1 - p) * rho
    for label in labels:
        P = np.kron(paulis_1q[label[0]], paulis_1q[label[1]])
        rho_new = rho_new + (p / 15) * P @ rho @ P.conj().T
    rho = rho_new

    ZZ = np.kron(Z, Z)
    exact = float(np.real(np.trace(ZZ @ rho)))

    sc = stim.Circuit(f"H 0\nCX 0 1\nDEPOLARIZE2({p}) 0 1")
    c = Circuit.from_stim(sc)
    obs = PauliObservable(n_qubits=2, terms=[('ZZ', 1.0)])
    result = estimate(c, obs, n_samples=n_samples, seed=seed)

    print_result(f"from_stim Bell + DEPOLARIZE2(p={p}) → <ZZ>", exact, result)
    return result


def example_memory_experiment(n_samples: int = 2_000, seed: int = 11):
    """
    MCSampler on a noisy 3-qubit repetition code memory experiment.

    Circuit (2 stabilizer rounds + final readout):
        Qubits 0,1,2 are data qubits.
        Qubits 3,4 are ancilla (syndrome) qubits.

        Round:
            H 3 4
            CX 3 0  CX 3 1   (ZZ stabilizer on 0,1)
            CX 4 1  CX 4 2   (ZZ stabilizer on 1,2)
            DEPOLARIZE1 p on each qubit
            MR 3 4            (mid-circuit measurement + reset)

        Final:
            M 0 1 2           (data qubit readout)

    Validation checks (no RZ → all weights = 1.0):
        1. measurements.shape == (n_samples, 6)   [4 syndrome bits + 2 rounds + 3 data bits = 4+3 = wait, 2 rounds * 2 ancillas + 3 data = 4+3=7]
        2. weights.shape == (n_samples,) and np.allclose(weights, 1.0)
        3. measurements dtype == uint8, values in {0, 1}
        4. one_norm == 1.0
    """
    p = 0.05
    # Correct repetition-code stabilizer readout: H ancilla before and after CNOT block.
    # Each round: H→CX→CX→H→noise→MR (ancilla reset between rounds).
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
    sampler = MCSampler(c)
    result = sampler.sample(n_samples=n_samples, seed=seed)

    # Expected: 2 rounds * 2 ancilla bits + 3 data bits = 7 measurement bits
    expected_m = 7
    checks = [
        result.measurements.shape == (n_samples, expected_m),
        result.weights.shape == (n_samples,),
        bool(np.allclose(result.weights, 1.0)),      # no RZ → all weights ≈ 1
        result.measurements.dtype == np.uint8,
        bool(np.isclose(result.one_norm, 1.0)),      # noise only → 1-norm ≈ 1
        bool(np.all((result.measurements == 0) | (result.measurements == 1))),
    ]
    ok = all(checks)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] memory_experiment: {n_samples} samples")
    print(f"       measurements.shape={result.measurements.shape}  "
          f"weights.shape={result.weights.shape}")
    print(f"       all_weights_unity={np.allclose(result.weights, 1.0)}  "
          f"one_norm={result.one_norm:.6f}")
    print(f"       ancilla_bit_mean={result.measurements[:, :4].mean():.4f}  "
          f"(structural test — shape/weight/dtype checks)")
    print()
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("QC_claude — Monte Carlo Stabilizer Estimator — Validation")
    print("=" * 64)
    print()

    example_hrz_h(theta=np.pi / 4)    # T gate: 1-norm = sqrt(2)
    example_hrz_h(theta=np.pi / 2)    # RZ(pi/2): 1-norm = 1 (pure S channel)
    example_hrz_h(theta=np.pi / 8)    # Small angle
    example_hrz_h(theta=3 * np.pi / 4)
    example_sdg_rz_h()
    example_two_rz()
    example_bell_rz()
    example_pauli_sum_obs()
    example_clifford_only()
    example_scan_angles()

    print("── Noise examples ──────────────────────────────────────────────")
    print()
    example_depolarize1()
    example_from_stim_depolarize()
    example_from_stim_rz()
    example_pauli_channel_2()

    print("── Sampler examples ────────────────────────────────────────────")
    print()
    example_memory_experiment()

    print("Done.")
