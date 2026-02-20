# reproduction of the experiment in the paper of "Exact and Efficient Stabilizer Simulation of Thermal-Relaxation Noise for Quantum Error Correction"

import os
import sinter
import stim
import qldpc
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pauli_t import IonTrapT


def generate_noisy_circuit(
    distance: int,
    rounds: int,
    t_idle: float,
    pauli_t: IonTrapT,
) -> stim.Circuit:
    base_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=distance,
    )
    depol_model = qldpc.circuits.DepolarizingNoiseModel(
        p=5e-3, include_idling_error=False
    )
    noisy_circuit = depol_model.noisy_circuit(base_circuit)
    noisy_circuit = pauli_t.add_idle_before_measurement(noisy_circuit, t_idle)
    return noisy_circuit


def generate_tasks(
    distances: list[int],
    t_idles: list[float],
    pauli_t: IonTrapT,
) -> list[sinter.Task]:
    tasks = []
    for d in distances:
        for t_idle in t_idles:
            circuit = generate_noisy_circuit(
                distance=d, rounds=d, t_idle=t_idle, pauli_t=pauli_t
            )
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        "d": d,
                        "r": d,
                        "t_idle": t_idle,
                        "t_idle_over_t1": t_idle / pauli_t.t1,
                    },
                )
            )
    return tasks


def run_experiment(
    distances: list[int],
    t_idles: list[float],
    num_workers: int,
    max_shots: int,
    max_errors: int | None = None,
    pauli_t: IonTrapT | None = None,
) -> list[sinter.TaskStats]:
    tasks = generate_tasks(distances, t_idles, pauli_t=pauli_t)
    stats = sinter.collect(
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        tasks=tasks,
        decoders=["pymatching"],
        print_progress=True,
    )
    return list(stats)


def _extract_curve(
    d_stats: list[sinter.TaskStats],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = [s.json_metadata["t_idle_over_t1"] for s in d_stats]
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
        np.array(xs)[valid],
        np.array(error_rates)[valid],
        np.array(error_bars_low)[valid],
        np.array(error_bars_high)[valid],
    )


def plot_results(stats: list[sinter.TaskStats], output_file: str = "sinter_plot.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    distances = sorted(set(s.json_metadata["d"] for s in stats))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(distances)))

    with plt.style.context(["science"]):
        for d, color in zip(distances, colors):
            sub = sorted(
                [s for s in stats if s.json_metadata["d"] == d],
                key=lambda s: s.json_metadata["t_idle_over_t1"],
            )
            if not sub:
                continue
            xs, rates, el, eh = _extract_curve(sub)
            ax.errorbar(
                xs,
                rates,
                yerr=[el, eh],
                fmt="o-",
                color=color,
                label=f"{d}",
                capsize=4,
                markersize=6,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$t_{idle} / T_1$")
        ax.set_ylabel("Logical Error Rate")
        ax.set_title("Surface Code: Idle Before Measurement")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Plot saved to {output_file}")


def save_results(stats: list[sinter.TaskStats], output_file: str = "sinter_data.csv"):
    with open(output_file, "w") as f:
        f.write(
            "distance,rounds,t_idle,t_idle_over_t1,shots,errors,discards,error_rate\n"
        )
        for s in stats:
            rate = s.errors / s.shots if s.shots > 0 else 0
            f.write(
                f"{s.json_metadata['d']},{s.json_metadata['r']},"
                f"{s.json_metadata['t_idle']:.6e},{s.json_metadata['t_idle_over_t1']:.6e},"
                f"{s.shots},{s.errors},{s.discards},{rate:.6e}\n"
            )
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    pt = IonTrapT()
    t_idles = list(np.logspace(-3, -1, 12) * pt.t1)
    stats = run_experiment(
        distances=[3, 5, 7],
        t_idles=t_idles,
        max_shots=100_000,
        num_workers=os.cpu_count() or 1,
        pauli_t=pt,
    )

    save_results(stats, "results/iontrap_idle_data.csv")
    plot_results(stats, "results/iontrap_idle_plot.png")

    print("\nSummary:")
    for s in sorted(
        stats,
        key=lambda x: (x.json_metadata["d"], x.json_metadata["t_idle_over_t1"]),
    ):
        d = s.json_metadata["d"]
        ratio = s.json_metadata["t_idle_over_t1"]
        rate = s.errors / s.shots if s.shots > 0 else 0
        print(f"  d={d}, t_idle/T1={ratio:.4e}: {s.errors}/{s.shots} = {rate:.4e}")
