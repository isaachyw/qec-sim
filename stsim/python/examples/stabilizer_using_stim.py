from __future__ import annotations
import sys

sys.path.insert(0, "/home/isaachyw/research/stabsim/stsim/python/build")
import argparse
from pathlib import Path
from typing import List

import stim

from nwqsim import (
    Circuit,
    circuit_from_stim_file,
    circuit_from_stim_text,
    create_state,
)


def build_sample_stim_circuit() -> str:
    if stim is None:
        raise RuntimeError(
            "The sample Stim circuit requires the 'stim' package. Install stim or provide --stim-path."
        )

    circuit = stim.Circuit()
    circuit.append("H", [0, 1, 2])
    circuit.append("S", [0, 1, 2])
    circuit.append("CX", [0, 3, 1, 4, 2, 5])
    circuit.append("M", [3, 4, 5])
    circuit.append("R", [3, 4, 5])
    circuit.append("CX", [0, 3, 1, 4, 2, 5])
    circuit.append("H", [0, 1, 2])
    circuit.append("S_DAG", [0, 1, 2])
    circuit.append("M", [0, 1, 2, 3, 4, 5])
    return str(circuit)


def load_circuit_from_args(args: argparse.Namespace) -> Circuit:
    if args.stim_path:
        path = Path(args.stim_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find Stim file: {path}")
        return circuit_from_stim_file(str(path))

    stim_text = build_sample_stim_circuit()
    if args.export_sample:
        Path(args.export_sample).write_text(stim_text, encoding="utf-8")
    return circuit_from_stim_text(stim_text)


def simulate_with_stabilizer(circuit: Circuit, backend: str, method: str) -> None:
    state = create_state(backend, circuit.num_qubits(), method)
    sim_time_ms = state.simulate(circuit)
    results: List[int] = state.measurement_results()

    print(
        f"Simulated Stim circuit with {circuit.num_qubits()} qubits "
        f"using the {backend.upper()} {method.upper()} backend in {sim_time_ms:.3f} ms."
    )

    if not results:
        print("No measurements were produced by this circuit.")
        return

    preview = min(500, len(results))
    print(f"Showing the first {preview} measurement results:")
    for value in results[:preview]:
        bitstring = format(value, f"0{circuit.num_qubits()}b")
        print(f"  {bitstring}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Stim circuits into NWQSim stabilizer circuits while exercising multi-target H/S "
            "instructions and batched CX fan-outs."
        )
    )
    parser.add_argument(
        "--stim-path",
        type=str,
        help="Optional path to an existing .stim file to import instead of the built-in demo.",
    )
    parser.add_argument(
        "--export-sample",
        type=str,
        help="When building the demo Stim circuit, also write the text to this path for inspection.",
    )
    parser.add_argument(
        "--backend",
        default="cpu",
        help="Backend identifier passed to nwqsim.create_state (default: cpu).",
    )
    parser.add_argument(
        "--method",
        default="stab",
        help="Simulation method to request from NWQSim (default: stab).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    circuit = load_circuit_from_args(args)
    simulate_with_stabilizer(circuit, backend=args.backend, method=args.method)


if __name__ == "__main__":
    main()
