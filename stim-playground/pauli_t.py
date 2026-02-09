import qldpc
import numpy as np
import stim
from qldpc.circuits.noise_model import OP_TYPES
from stim_utils import save_stim_svg
# Physical Readout Pulse: 50--200 ns RIKEN / ArXiv:2412.13849
# Total Measure + Logic Window: 1.1 us
# Google (Willow)Decoding Latency: 480 ns
# IBM (Loon): 2.0 us


# for a given noisy circuit, deplorzing without idle error, apply pauli(tau) based on the gate type to all the qubits in the circuit


class _PauliT:
    # subclasses define these (all times in ns)
    t1: float
    t2: float
    sqgate: float
    tqgate: float
    measurement_time: float
    reset_time: float

    def apply_idle(
        self,
        circ: stim.Circuit,
        t: float,
        idle_qubits: list[int],
    ):
        # Idle errors
        p_x = 0.25 * (1 - np.exp(-t * 1.0 / self.t1))
        p_y = p_x
        p_z = 0.5 * (1 - np.exp(-t * 1.0 / self.t2)) - p_x
        circ.append("PAULI_CHANNEL_1", idle_qubits, (p_x, p_y, p_z))

    def categorize_gates(
        self, circ: stim.CircuitInstruction | stim.CircuitRepeatBlock
    ) -> str:
        return OP_TYPES[circ.name]

    def get_gate_time(self, category: str) -> float | None:
        """
        Get the duration of a gate based on its category.

        Args:
            category: Gate category from OP_TYPES (e.g., "C1", "C2", "M1", etc.)

        Returns:
            Gate duration in nanoseconds, or None for annotations/noise
        """
        match category:
            case "C1":
                return self.sqgate
            case "C2":
                return self.tqgate
            case "M1" | "M2" | "MPP":
                return self.measurement_time
            case "R1":
                return self.reset_time
            case "MR1":
                return self.measurement_time + self.reset_time
            case _:
                return None  # Annotations ("info") and noise ("!?")

    def _compute_pauli_probs(self, t: float) -> tuple[float, float, float]:
        """
        Compute Pauli error probabilities from T1/T2 relaxation.

        Args:
            t: Time duration in nanoseconds

        Returns:
            Tuple of (p_x, p_y, p_z) error probabilities
        """
        p_x = 0.25 * (1 - np.exp(-t / self.t1))
        p_y = p_x
        p_z = 0.5 * (1 - np.exp(-t / self.t2)) - p_x
        return (p_x, p_y, p_z)

    def apply_thermal_relaxation(self, circuit: stim.Circuit) -> stim.Circuit:
        """
        Apply thermal relaxation errors to all qubits after each gate operation.

        For each gate operation in the circuit, this function:
        1. Categorizes the gate to determine its duration
        2. Computes T1/T2 decoherence probabilities for that duration
        3. Appends a PAULI_CHANNEL_1 error to all qubits after the gate

        REPEAT blocks are processed recursively.

        Args:
            circuit: Input Stim circuit (with noise except idle errors)

        Returns:
            New circuit with PAULI_CHANNEL_1 inserted after each gate
        """
        num_qubits = circuit.num_qubits
        if num_qubits == 0:
            return circuit.copy()

        all_qubits = list(range(num_qubits))
        output = stim.Circuit()

        for instr in circuit:
            if isinstance(instr, stim.CircuitRepeatBlock):
                # Recursively process the body of the REPEAT block
                processed_body = self.apply_thermal_relaxation(instr.body_copy())
                output.append(
                    stim.CircuitRepeatBlock(
                        repeat_count=instr.repeat_count,
                        body=processed_body,
                    )
                )
            else:
                # It's a CircuitInstruction
                # First, append the original instruction
                output.append(instr)

                # Get the gate category and corresponding time
                try:
                    category = self.categorize_gates(instr)
                    gate_time = self.get_gate_time(category)
                except KeyError:
                    # Unknown gate type, skip thermal relaxation
                    gate_time = None

                if gate_time is not None:
                    # Compute Pauli error probabilities and apply to all qubits
                    p_x, p_y, p_z = self._compute_pauli_probs(gate_time)
                    # Only add noise if there's a non-zero probability
                    if p_x > 0 or p_y > 0 or p_z > 0:
                        output.append(
                            "PAULI_CHANNEL_1", instr.targets_copy(), (p_x, p_y, p_z)
                        )

        return output


class GoogleT(_PauliT):
    # https://quantumai.google/hardware/datasheet/weber.pdf Sycamore
    # google willow data (all times in ns)
    t1 = 15e3
    t2 = 20e3  # estimate
    sqgate = 25
    tqgate = 22
    measurement_time = 1.1e3  # not sure (the paper said one full cycle is 1.1 us)
    reset_time = 25  # not sure

    def __str__(self) -> str:
        return "GoogleT"


class IBMT(_PauliT):
    # https://arxiv.org/abs/2503.04642 (all times in ns, converted from us)
    # device data from https://arxiv.org/abs/2503.04642
    # in (us)
    # H,S (single qubit gate) gate 0.02, CNOT, CZ(two qubit gate) 0.04, Measurement 0.6, reset 0.5, T1 30 (us), T2 30 (us)
    t1 = 30e3
    t2 = 30e3
    sqgate = 20  # 0.02 us
    tqgate = 40  # 0.04 us
    measurement_time = 600  # 0.6 us
    reset_time = 500  # 0.5 us

    def __str__(self) -> str:
        return "IBMT"


if __name__ == "__main__":
    sg = GoogleT()
    deplor_noise_model = qldpc.circuits.DepolarizingNoiseModel(
        p=0.001, include_idling_error=False
    )
    circ = stim.Circuit.generated("surface_code:rotated_memory_x", rounds=3, distance=3)
    # circ = deplor_noise_model.noisy_circuit(circ)
    print("Original circuit gate categories:")
    for instr in circ:
        if isinstance(instr, stim.CircuitRepeatBlock):
            print("REPEAT block")
        else:
            print(f"  {instr.name} -> {sg.categorize_gates(instr)}")

    print("\nApplying thermal relaxation...")
    noisy_circ = sg.apply_thermal_relaxation(circ)
    print(f"Original circuit: {len(circ)} instructions")
    print(f"Noisy circuit: {len(noisy_circ)} instructions")
    print(noisy_circ.detector_error_model(decompose_errors=True))
    save_stim_svg(noisy_circ, "noisy_circuit.svg")
    save_stim_svg(circ, "original_circuit.svg")
