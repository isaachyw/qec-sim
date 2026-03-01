"""
Exact damping experiment: Surface code with exact T1/T2 damping (DampOp)
before measurement, instead of the Pauli twirling approximation.

Pipeline per (distance, t_idle) point:
    1. generate stim circuit with depolarizing noise (no idle error)
    2. Convert to QC_claude Circuit, add DampOp via add_idle_exact_before_measurement
    3. Sample with MCSampler (handles DampOp quasi-probability decomposition)
    4. Parse DETECTOR/OBSERVABLE_INCLUDE from the stim circuit
    5. Evaluate detector syndromes and observables via XOR
    6. Decode with PyMatching (DEM from Pauli twirling approximation)
    7. Compute weighted logical error rate: LER = (1/N) * sum(w_i * is_error_i)

Key differences from baseline_reproduce.py:
    - Uses DampOp (exact 3-term stabilizer decomposition: I, Z, Reset)
      instead of PAULI_CHANNEL_1 (Pauli twirling approximation).
    - When T1 >= T2 (e.g. IonTrap): all weights = 1.0 (exact probability).
    - When T1 < T2 <= 2*T1: weights can be negative (quasi-probability),
      and the weighted average correctly handles this.
"""

import sys
from pathlib import Path

# exp/ -> QC_claude/ -> stim-playground/ -> pt_qc/
_pt_qc_dir = str(Path(__file__).resolve().parent.parent.parent / "pt_qc")
if _pt_qc_dir not in sys.path:
    sys.path.insert(0, _pt_qc_dir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import stim
import qldpc
import pymatching
import matplotlib.pyplot as plt
from pauli_t import IonTrapT

from QC_claude.circuit import Circuit
from QC_claude.sampler import MCSampler
from QC_claude.detector import _parse_detectors_and_observables, _evaluate_xor


# -- Circuit generation -------------------------------------------------------


def generate_noisy_circuit_no_idle(
    distance: int,
    rounds: int,
) -> stim.Circuit:
    """Generate stim circuit with depolarizing noise but NO idle error."""
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=distance,
    )
    depol_model = qldpc.circuits.DepolarizingNoiseModel(
        p=5e-3, include_idling_error=False
    )
    return depol_model.noisy_circuit(base)


# -- Single-point evaluation --------------------------------------------------


def evaluate_point(
    distance: int,
    t_idle: float,
    pauli_t: IonTrapT,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> dict:
    """
    Run one (distance, t_idle) point with exact DampOp idle noise.

    Returns dict with: d, t_idle, t_idle_over_t1, shots, one_norm,
        weighted_error_rate, unweighted_errors, unweighted_error_rate
    """
    # 1. Stim circuit: depolarizing noise only (no idle)
    noisy_stim = generate_noisy_circuit_no_idle(distance, rounds=distance)

    # 2. DEM for decoder: use Pauli twirling approximation
    #    (standard practice: approximate decoder, exact sampling)
    approx_stim = pauli_t.add_idle_before_measurement(noisy_stim, t_idle)
    dem = approx_stim.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # 3. Parse detector/observable definitions from the stim circuit
    detector_records, observable_records, total_meas = _parse_detectors_and_observables(
        noisy_stim
    )
    n_observables = max(observable_records.keys()) + 1 if observable_records else 0
    obs_record_lists = [observable_records.get(i, []) for i in range(n_observables)]

    # 4. Convert to QC_claude Circuit and add exact DampOp
    qc_circuit = Circuit.from_stim(noisy_stim)
    qc_circuit = qc_circuit.add_idle_exact_before_measurement(
        t_idle, pauli_t.t1, pauli_t.t2
    )

    # 5. Sample with MCSampler (handles DampOp decomposition + weights)
    sampler = MCSampler(qc_circuit)
    result = sampler.sample(n_samples, seed=seed, n_workers=-1)

    # 6. Evaluate detector syndromes and observables
    detectors = _evaluate_xor(result.measurements, detector_records)
    observables = _evaluate_xor(result.measurements, obs_record_lists)

    # 7. Decode
    predictions = matcher.decode_batch(detectors)

    # 8. Weighted logical error rate
    #    QP estimator: LER = (1/N) * sum(w_i * is_error_i)
    #    When all weights = 1.0 this reduces to simple counting.
    is_error = np.any(predictions != observables, axis=1).astype(np.float64)
    weights = result.weights
    weighted_error_rate = float(np.mean(weights * is_error))

    return {
        "d": distance,
        "t_idle": t_idle,
        "t_idle_over_t1": t_idle / pauli_t.t1,
        "shots": n_samples,
        "one_norm": result.one_norm,
        "weighted_error_rate": weighted_error_rate,
        "unweighted_errors": int(is_error.sum()),
        "unweighted_error_rate": float(is_error.mean()),
    }


# -- Full sweep ---------------------------------------------------------------


def run_sweep(
    distances: list[int],
    t_idles: list[float],
    pauli_t: IonTrapT,
    n_samples: int = 100_000,
    seed: int = 0,
) -> list[dict]:
    """Sweep over all (distance, t_idle) combinations."""
    results = []
    total = len(distances) * len(t_idles)
    for i, d in enumerate(distances):
        for j, t_idle in enumerate(t_idles):
            idx = i * len(t_idles) + j + 1
            print(
                f"[{idx}/{total}] d={d}, t_idle/T1={t_idle / pauli_t.t1:.4e} ...",
                end="",
                flush=True,
            )
            r = evaluate_point(
                distance=d,
                t_idle=t_idle,
                pauli_t=pauli_t,
                n_samples=n_samples,
                seed=seed + idx,
            )
            print(
                f"  1-norm={r['one_norm']:.4f}  "
                f"weighted={r['weighted_error_rate']:.4e}  "
                f"unweighted={r['unweighted_errors']}/{r['shots']}"
            )
            results.append(r)
    return results


# -- Plotting ------------------------------------------------------------------


def plot_results(results: list[dict], output_file: str = "exact_damp_sc.png"):
    """Plot weighted logical error rate vs t_idle/T1 for each distance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    distances = sorted(set(r["d"] for r in results))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(distances)))

    for d, color in zip(distances, colors):
        sub = sorted(
            [r for r in results if r["d"] == d],
            key=lambda r: r["t_idle_over_t1"],
        )
        xs = [r["t_idle_over_t1"] for r in sub]
        ys = [r["weighted_error_rate"] for r in sub]
        valid = [(x, y) for x, y in zip(xs, ys) if y > 0]
        if valid:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, "o-", color=color, label=f"d={d}", markersize=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t_{idle} / T_1$")
    ax.set_ylabel("Weighted Logical Error Rate")
    ax.set_title("Exact Damp: Surface Code Idle Before Measurement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to {output_file}")


def save_results(results: list[dict], output_file: str = "exact_damp_sc.csv"):
    """Save results to CSV."""
    with open(output_file, "w") as f:
        f.write(
            "distance,rounds,t_idle,t_idle_over_t1,shots,one_norm,"
            "weighted_error_rate,unweighted_errors,unweighted_error_rate\n"
        )
        for r in results:
            f.write(
                f"{r['d']},{r['d']},"
                f"{r['t_idle']:.6e},{r['t_idle_over_t1']:.6e},"
                f"{r['shots']},{r['one_norm']:.6f},"
                f"{r['weighted_error_rate']:.6e},"
                f"{r['unweighted_errors']},{r['unweighted_error_rate']:.6e}\n"
            )
    print(f"Results saved to {output_file}")


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    pt = IonTrapT()
    t_idles = list(np.logspace(-3, -1, 12) * pt.t1)

    distances = [3, 5, 7]
    n_samples = 100_000

    print(f"IonTrapT: T1={pt.t1:.2f} ns, T2={pt.t2:.2f} ns")
    print(f"  T2 < T1 => exact probability regime (1-norm = 1.0)")
    print(f"Distances: {distances}")
    print(f"t_idle points: {len(t_idles)}")
    print(f"Samples per point: {n_samples}")
    print()

    results = run_sweep(
        distances=distances,
        t_idles=t_idles,
        pauli_t=pt,
        n_samples=n_samples,
    )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    save_results(results, str(out_dir / "exact_damp_sc.csv"))
    plot_results(results, str(out_dir / "exact_damp_sc.png"))

    print("\nSummary:")
    for r in sorted(results, key=lambda x: (x["d"], x["t_idle_over_t1"])):
        print(
            f"  d={r['d']}, t_idle/T1={r['t_idle_over_t1']:.4e}: "
            f"weighted={r['weighted_error_rate']:.4e}  "
            f"unweighted={r['unweighted_errors']}/{r['shots']}  "
            f"1-norm={r['one_norm']:.4f}"
        )
