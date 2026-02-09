from sinter_exp import generate_noisy_circuit

circuit = generate_noisy_circuit(distance=3, rounds=3, depol_p=0.001)
circuit.to_file("noisy_circuit.stim")
