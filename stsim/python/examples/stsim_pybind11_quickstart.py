from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys


def ensure_nwqsim_importable() -> None:
    build_dir = Path(__file__).resolve().parents[1] / "build"
    sys.path.insert(0, str(build_dir))


def main() -> None:
    ensure_nwqsim_importable()

    from nwqsim import Circuit, config, create_state

    num_qubits = 3
    backend = "cpu"
    method = "stab"

    circuit = Circuit(num_qubits)
    circuit.h(0).cx(0, 1).cx(1, 2)
    for qubit in range(num_qubits):
        circuit.m(qubit)

    shots = 1024
    counts: Counter[int] = Counter()

    state = create_state(backend, num_qubits, method)
    total_time = 0.0
    for _ in range(shots):
        state.reset()
        total_time += state.simulate(circuit)
        for v in state.measurement_results():
            counts[v] += 1

    print(f"simulate time: {total_time:.3f} ms ({shots} shots)")
    for value, count in sorted(counts.items()):
        bitstring = format(value, f"0{num_qubits}b")
        print(f"{bitstring}: {count}")


if __name__ == "__main__":
    main()
