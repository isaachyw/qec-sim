"""
QC_claude: Monte Carlo estimator for quantum circuits using stabilizer decompositions.

Implements the quasiprobability Monte Carlo algorithm (algo.md):
    - Non-Clifford RZ(theta) gates decomposed into Clifford channels (I, Z, S).
    - Pauli noise channels (DEPOLARIZE1/2, PAULI_CHANNEL_1/2) sampled exactly.
    - Each sample runs an exact stabilizer simulation via Stim's TableauSimulator.
    - Expectation value estimated as a weighted average over samples.

Build from scratch:
    >>> import numpy as np
    >>> from QC_claude import Circuit, PauliObservable, estimate
    >>> c = Circuit(n_qubits=1)
    >>> c.h(0).rz(0, np.pi / 4).h(0)
    >>> result = estimate(c, PauliObservable.single_z(1, 0), n_samples=50_000, seed=0)

Import from Stim:
    >>> import stim
    >>> sc = stim.Circuit("H 0\\nCX 0 1\\nDEPOLARIZE1(0.01) 0 1")
    >>> c = Circuit.from_stim(sc)
    >>> c.rz(0, np.pi / 4)   # add non-Clifford gate after import
    >>> result = estimate(c, PauliObservable(2, [('ZZ', 1.0)]), n_samples=50_000)
"""

from .circuit import (
    Circuit,
    CliffordOp,
    RZOp,
    PauliNoiseOp,
    TwoQubitPauliNoiseOp,
    DampOp,
    MeasureOp,
    TickOp,
)
from .decompositions import (
    rz_decomposition,
    pauli_noise_decomp_1q,
    pauli_noise_decomp_2q,
    damp_decomposition,
    GateDecomposition,
    CliffordTerm,
)
from .observable import PauliObservable
from .estimator import MCEstimator, EstimationResult, estimate
from .sampler import MCSampler, SamplerResult, sample as sample_circuit
from .detector import DetectorSampler, DetectorSamplerResult, sample_detectors

__all__ = [
    # Circuit ops
    "Circuit",
    "CliffordOp",
    "RZOp",
    "PauliNoiseOp",
    "TwoQubitPauliNoiseOp",
    "DampOp",
    "MeasureOp",
    "TickOp",
    # Decompositions
    "rz_decomposition",
    "pauli_noise_decomp_1q",
    "pauli_noise_decomp_2q",
    "damp_decomposition",
    "GateDecomposition",
    "CliffordTerm",
    # Observable
    "PauliObservable",
    # Estimator
    "MCEstimator",
    "EstimationResult",
    "estimate",
    # Sampler
    "MCSampler",
    "SamplerResult",
    "sample_circuit",
    # Detector sampler
    "DetectorSampler",
    "DetectorSamplerResult",
    "sample_detectors",
]
