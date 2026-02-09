import os
import sinter
import stim
import qldpc
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pauli_t import GoogleT, IBMT, _PauliT


HARDWARE = "IBM"


def generate_noisy_circuit(
    distance: int,
    rounds: int,
    depol_p: float,
    pauli_t: _PauliT | None = None,
) -> stim.Circuit:
    """Generate surface code circuit with depolarizing noise, optionally + thermal relaxation."""
    base_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=distance,
    )
    depol_model = qldpc.circuits.DepolarizingNoiseModel(
        p=depol_p, include_idling_error=False
    )
    noisy_circuit = depol_model.noisy_circuit(base_circuit)
    if pauli_t is not None:
        noisy_circuit = pauli_t.apply_thermal_relaxation(noisy_circuit)
    return noisy_circuit


def generate_tasks(
    distances: list[int],
    depol_rates: list[float],
    pauli_t: _PauliT | None = None,
) -> list[sinter.Task]:
    """Generate sinter tasks for all parameter combinations, with and without thermal relaxation."""
    tasks = []
    for thermal in [False, True]:
        pt = pauli_t if thermal else None
        for d in distances:
            for p in depol_rates:
                circuit = generate_noisy_circuit(
                    distance=d, rounds=d, depol_p=p, pauli_t=pt
                )
                tasks.append(
                    sinter.Task(
                        circuit=circuit,
                        json_metadata={
                            "d": d,
                            "r": d,
                            "p": p,
                            "thermal": thermal,
                        },
                    )
                )
    return tasks


def run_experiment(
    distances: list[int],
    depol_rates: list[float],
    num_workers: int,
    max_shots: int,
    max_errors: int | None = None,
    pauli_t: _PauliT | None = None,
) -> list[sinter.TaskStats]:
    """Run sinter experiment and return statistics."""
    tasks = generate_tasks(distances, depol_rates, pauli_t=pauli_t)
    stats = sinter.collect(
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        tasks=tasks,
        decoders=["pymatching"],
    )
    return list(stats)


def _extract_curve(
    d_stats: list[sinter.TaskStats],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract valid (ps, rates, err_low, err_high) from a sorted list of stats."""
    ps = [s.json_metadata["p"] for s in d_stats]
    error_rates = []
    error_bars_low = []
    error_bars_high = []
    for s in d_stats:
        if s.shots > 0 and s.errors > 0:
            rate = s.errors / s.shots
            n = s.shots
            z = 1.96
            p_hat = rate
            denom = 1 + z**2 / n
            center = (p_hat + z**2 / (2 * n)) / denom
            margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
            error_rates.append(rate)
            error_bars_low.append(rate - max(0, center - margin))
            error_bars_high.append(min(1, center + margin) - rate)
        else:
            error_rates.append(np.nan)
            error_bars_low.append(0)
            error_bars_high.append(0)
    valid = ~np.isnan(error_rates)
    return (
        np.array(ps)[valid],
        np.array(error_rates)[valid],
        np.array(error_bars_low)[valid],
        np.array(error_bars_high)[valid],
    )


def plot_results(stats: list[sinter.TaskStats], output_file: str = "sinter_plot.png"):
    """Plot logical error rate vs physical error rate, with and without thermal relaxation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    distances = sorted(set(s.json_metadata["d"] for s in stats))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(distances)))
    style_map = {False: ("--", "^", "depol only"), True: ("-", "o", "depol+thermal")}

    with plt.style.context(["science", "no-latex"]):
        for d, color in zip(distances, colors):
            for thermal, (ls, marker, tag) in style_map.items():
                sub = sorted(
                    [
                        s
                        for s in stats
                        if s.json_metadata["d"] == d
                        and s.json_metadata["thermal"] == thermal
                    ],
                    key=lambda s: s.json_metadata["p"],
                )
                if not sub:
                    continue
                ps, rates, el, eh = _extract_curve(sub)
                ax.errorbar(
                    ps,
                    rates,
                    yerr=[el, eh],
                    fmt=f"{marker}{ls}",
                    color=color,
                    label=f"d={d} {tag}",
                    capsize=3,
                    markersize=5,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate (p)")
        ax.set_ylabel("Logical Error Rate")
        ax.set_title("Surface Code: Depolarizing vs Depolarizing + Thermal Relaxation")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Plot saved to {output_file}")


def save_results(stats: list[sinter.TaskStats], output_file: str = "sinter_data.csv"):
    """Save experiment results to CSV."""
    with open(output_file, "w") as f:
        f.write("distance,depol_p,rounds,thermal,shots,errors,discards,error_rate\n")
        for s in stats:
            rate = s.errors / s.shots if s.shots > 0 else 0
            f.write(
                f"{s.json_metadata['d']},{s.json_metadata['p']:.6e},"
                f"{s.json_metadata['r']},{s.json_metadata['thermal']},"
                f"{s.shots},{s.errors},{s.discards},{rate:.6e}\n"
            )
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    stats = run_experiment(
        distances=[3, 5, 7, 9, 11],
        depol_rates=list(np.logspace(-3, -1.5, 8)),
        max_shots=100_000,
        num_workers=os.cpu_count() or 1,
        pauli_t=GoogleT() if HARDWARE == "Google" else IBMT(),
    )

    save_results(stats, f"results/{HARDWARE}_t_data.csv")
    plot_results(stats, f"results/{HARDWARE}_t_plot_science.png")

    print("\nSummary:")
    for s in sorted(
        stats,
        key=lambda x: (
            x.json_metadata["d"],
            x.json_metadata["thermal"],
            x.json_metadata["p"],
        ),
    ):
        d = s.json_metadata["d"]
        p = s.json_metadata["p"]
        thermal = s.json_metadata["thermal"]
        rate = s.errors / s.shots if s.shots > 0 else 0
        print(
            f"  d={d}, thermal={thermal}, p={p:.4f}: {s.errors}/{s.shots} = {rate:.4e}"
        )
