# reproduce the results from ken brown's paper

from qldpc import circuits, codes, decoders, objects
import stim
# reset, sq, tq, readout, idling,additional wait for measurement

noise = circuits.noise_model.SI1000NoiseModel(p=0.001)
deplor_noise_model = circuits.DepolarizingNoiseModel(
    p=0.001, include_idling_error=False
)

surface_code = codes.SurfaceCode(3, rotated=True)
# circuit = circuits.get_memory_experiment(
#     surface_code, basis=objects.Pauli.Z, num_rounds=3
# )
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", rounds=3, distance=3)
with open("noiseless_circuit.svg", "w") as f:
    f.write(str(circuit.diagram("timeline-svg")))
noisy_circuit = deplor_noise_model.noisy_circuit(circuit)
print(deplor_noise_model.get_noise_rule(stim.CircuitInstruction("X", [0])))
# save the circuit as svg
diagram = str(noisy_circuit.diagram("timeline-svg"))
print(noisy_circuit)
with open("noisy_circuit.svg", "w") as f:
    f.write(diagram)
