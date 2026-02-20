import sys
import os

# exp/ → QC_claude/ → stim-playground/
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from QC_claude import MCSampler, Circuit
from pt_qc.pauli_t import IonTrapT
from pt_qc.baseline_pt import generate_noisy_circuit


if __name__ == "__main__":
    circ = generate_noisy_circuit(distance=3, rounds=3, t_idle=0.01, pauli_t=IonTrapT())
    print(circ)
    print(circ.num_qubits)
    MCcirc = Circuit.from_stim(circ)
    # print(MCcirc)
