"""
Reproduce the baseline experiment from pt_qc/baseline_pt.py using QC_claude's
DetectorSampler instead of sinter.collect().

Pipeline per (distance, t_idle) point:
    1. generate_noisy_circuit() → stim.Circuit  (same as baseline_pt.py)
    2. DetectorSampler(circuit).sample(n_samples)  → detector + observable bits
    3. circuit.detector_error_model() → DEM for PyMatching
    4. PyMatching.decode_batch(detectors) → predicted observables
    5. Compare predictions vs actual observables → logical error rate

Since the circuit is Clifford+noise only (no RZ), all quasiprobability weights
are 1.0 and the error rate is a simple count ratio.
"""

import sys
from pathlib import Path

# Add pt_qc to sys.path so we can import IonTrapT and generate_noisy_circuit
_pt_qc_dir = str(Path(__file__).resolve().parent.parent.parent / "pt_qc")
if _pt_qc_dir not in sys.path:
    sys.path.insert(0, _pt_qc_dir)

import os
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt
from pauli_t import IonTrapT
from baseline_pt import generate_noisy_circuit

# QC_claude imports (package is two levels up from this file)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from QC_claude import DetectorSampler


# ── Single-point evaluation ──────────────────────────────────────────────────


def evaluate_point(
    distance: int,
    t_idle: float,
    pauli_t: IonTrapT,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> dict:
    """
    Run one (distance, t_idle) point: generate circuit, sample with QC_claude,
    decode with PyMatching, return statistics.

    Returns:
        dict with keys: d, t_idle, t_idle_over_t1, shots, errors, error_rate
    """
    circuit = generate_noisy_circuit(
        distance=distance,
        rounds=distance,
        t_idle=t_idle,
        pauli_t=pauli_t,
    )

    # Sample with QC_claude's DetectorSampler
    ds = DetectorSampler(circuit)
    result = ds.sample(n_samples=n_samples, seed=seed, n_workers=-1)

    # Build decoder from the detector error model
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # Decode: predict which observables flipped
    predictions = matcher.decode_batch(result.detectors)

    # Count logical errors (any observable mismatch in a sample → error)
    n_errors = int(np.any(predictions != result.observables, axis=1).sum())
    error_rate = n_errors / n_samples

    return {
        "d": distance,
        "t_idle": t_idle,
        "t_idle_over_t1": t_idle / pauli_t.t1,
        "shots": n_samples,
        "errors": n_errors,
        "error_rate": error_rate,
    }


# ── Full sweep ───────────────────────────────────────────────────────────────


def run_sweep(
    distances: list[int],
    t_idles: list[float],
    pauli_t: IonTrapT,
    n_samples: int = 100_000,
    seed: int = 0,
) -> list[dict]:
    """
    Sweep over all (distance, t_idle) combinations.

    Returns:
        List of result dicts (one per point), sorted by (d, t_idle).
    """
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
            print(f"  errors={r['errors']}/{r['shots']} = {r['error_rate']:.4e}")
            results.append(r)
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_results(results: list[dict], output_file: str = "qc_claude_baseline.png"):
    """Plot logical error rate vs t_idle/T1 for each distance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    distances = sorted(set(r["d"] for r in results))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(distances)))

    for d, color in zip(distances, colors):
        sub = sorted(
            [r for r in results if r["d"] == d],
            key=lambda r: r["t_idle_over_t1"],
        )
        xs = [r["t_idle_over_t1"] for r in sub]
        ys = [r["error_rate"] for r in sub]
        # Filter out zero-error points for log scale
        valid = [(x, y) for x, y in zip(xs, ys) if y > 0]
        if valid:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, "o-", color=color, label=f"d={d}", markersize=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t_{idle} / T_1$")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title("QC_claude Baseline: Surface Code Idle Before Measurement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to {output_file}")


def save_results(results: list[dict], output_file: str = "qc_claude_baseline.csv"):
    """Save results to CSV."""
    with open(output_file, "w") as f:
        f.write("distance,rounds,t_idle,t_idle_over_t1,shots,errors,error_rate\n")
        for r in results:
            f.write(
                f"{r['d']},{r['d']},"
                f"{r['t_idle']:.6e},{r['t_idle_over_t1']:.6e},"
                f"{r['shots']},{r['errors']},{r['error_rate']:.6e}\n"
            )
    print(f"Results saved to {output_file}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pt = IonTrapT()
    t_idles = list(np.logspace(-3, -1, 12) * pt.t1)

    # Start with smaller distances for validation; extend to [3,5,7,9,11]
    # once confirmed to match baseline.
    distances = [3, 5, 7]
    n_samples = 100_000

    print(f"IonTrapT: T1={pt.t1:.2f} ns, T2={pt.t2:.2f} ns")
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

    # Save outputs
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    save_results(results, str(out_dir / "qc_claude_baseline.csv"))
    plot_results(results, str(out_dir / "qc_claude_baseline.png"))

    # Print summary
    print("\nSummary:")
    for r in sorted(results, key=lambda x: (x["d"], x["t_idle_over_t1"])):
        print(
            f"  d={r['d']}, t_idle/T1={r['t_idle_over_t1']:.4e}: "
            f"{r['errors']}/{r['shots']} = {r['error_rate']:.4e}"
        )
