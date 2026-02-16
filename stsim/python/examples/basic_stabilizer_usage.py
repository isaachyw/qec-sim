from __future__ import annotations

from collections import Counter

import sys
import random

sys.path.insert(0, "/home/isaachyw/research/stabsim/stsim/python/build")
from nwqsim import Circuit, create_state, available_backends


def main() -> None:
    num_qubits = 10
    backend = "cpu"
    method = "STAB"

    circuit = Circuit(num_qubits)

    for i in range(num_qubits):
        print(i)
        circuit.h(i)

    for i in range(num_qubits):
        circuit.m(i)

    state = create_state(backend, num_qubits, method)
    state.print_config(method)
    num_shots = 10
    for _ in range(num_shots):
        # generate a random seed
        seed = random.randint(0, 1000000)
        state.reset()
        state.set_seed(seed)
        sim_time_ms = state.simulate(circuit)
        print(f"Simulation time: {sim_time_ms:.3f} ms")
        # samples = state.measure_all(1024)
        samples = state.measurement_results()
        print(samples)


if __name__ == "__main__":
    main()
