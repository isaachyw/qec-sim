import qldpc
import stim


def generate_stim_circuit(distance: int, error_rate: float, num_rounds: int):
    d = distance
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",  # correct Z errors
        rounds=num_rounds,  # number of error correction rounds (typically d)
        distance=d,
    )
    noise_model = qldpc.circuits.SI1000NoiseModel(p=error_rate)
    return noise_model.noisy_circuit(circ)


if __name__ == "__main__":
    circ = generate_stim_circuit(distance=3, error_rate=0.001, num_rounds=3)
    print(circ)
