# STABSim
STABSim is a quantum stabilizer simulator for QEC and more, with GPU integration. STABSim circuits can be created directly in C++ or using Python. QASM circuits can also be passed through to create circuits using qasm_extraction.hpp. GPU integreation provides a speedup in very wide circuits by utilizing CUDA warp primitives and wide parallelism for individual gates. MPI is utilized in both methods as a speedup for sampling -- utilizing more CPU cores and more GPU's allow many more samples to be produced in noisy simulation.

## Supported Gates

Gate support is being continuouly updated. Currently supported Clifford gates:
- CX(control,target), H, S, SDG, CY, CZ, RX(+-pi/2), RY(+-pi/2)

Currently supported noise gates:
- CHAN1(x,y,z), CHAN2(XI,XX,XY,XZ,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ,IZ,IX,IY): Depolarizing noise channels with independent noise probabilities for each Pauli
- DEP1(p), DEP2(p): Depolarizing noise channels with uniform noise probabilities
- DAMP(relax p, decohere p): Qubit relaxation and phase decoherence
