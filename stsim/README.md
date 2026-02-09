# STABSim
STABSim is a quantum stabilizer simulator for QEC and more, with GPU integration. STABSim circuits can be created directly in C++ or made using Python. QASM circuits can also be passed through to create circuits using qasm_extraction.hpp. GPU integration provides a speedup in very wide circuits by utilizing CUDA warp primitives and wide parallelism for individual gates. MPI is utilized in both methods as a speedup for sampling -- utilizing more CPU cores and more GPU's allow many more samples to be produced in noisy simulation. 

### Implementation details and citation for usage can be found here [arXiv:2507.03092](https://arxiv.org/abs/2507.03092)

##

## Supported Gates

Gate support is being continuouly updated. Currently supported Clifford gates:
- CX(control,target), H, S, SDG, CY, CZ, RX($\pm \pi$/2), RY($\pm \pi$/2)

Supported non-Clifford gates(in testing):
- One non-commuting RZ($\phi$) per-circuit, by adding an extra phase column and tracking branch coefficients

Currently supported noise gates:
- CHAN1(x,y,z), CHAN2(XI,XX,XY,XZ,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ,IZ,IX,IY): Depolarizing noise channels with independent noise probabilities for each Pauli
- DEP1(p), DEP2(p): Depolarizing noise channels with uniform noise probabilities
- DAMP(relax p, decohere p): Qubit relaxation and phase decoherence
- T1(relax p): Qubit relaxation
- T2(decohere p): Qubit phase decoherence
### Note: Non-Clifford noise sources are decomposed into probabilistic stabilizer gates. Since the stabilizer decomposition has some negative coefficients, it requires quasiprobability sampling to converge exactly. Details on how this is done can be found in [arXiv:2507.03092](https://arxiv.org/abs/2512.09189)
