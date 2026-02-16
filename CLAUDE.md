# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STABSim is a quantum stabilizer simulator for quantum error correction (QEC) with GPU integration. It uses the tableau formalism (2n+1 × 2n representation) for Clifford simulation, with backends for CPU and CUDA GPU. The Python module is called `nwqsim`.

## Build Commands

### Python extension (pybind11)

```bash
# Initial setup (from stsim/)
cd stsim && uv sync          # install Python deps (pybind11, stim)

# Configure (CPU only)
cd stsim/python && mkdir -p build && cd build && cmake ..

# Configure (with CUDA backend)
cd stsim/python/build && cmake -DSTSIM_PY_ENABLE_CUDA_BACKEND=ON ..

# Build
cd stsim/python/build && make

# Quick rebuild after code changes
cd stsim/python && ./rebuild.sh   # just runs: cd build && make
```

The built `nwqsim*.so` lands in `stsim/python/build/`. The venv at `stsim/.venv/` is auto-detected by CMake.

### QASM executable

```bash
cd stsim/qasm && mkdir -p build && cd build && cmake .. && make
```

## Architecture

### Backends (header-only, in `stsim/include/stabsim/`)

- **`stab_cpu.hpp`** — CPU stabilizer simulation. Tableau stored as `vector<vector<int8_t>>`. Core Clifford gate logic (H, S, CX, etc.), measurement via Gottesman-Knill, noise channel decomposition.
- **`stab_cuda.cuh`** — CUDA GPU backend. Packed bit representation (32 rows per `int32`), uses warp primitives and cooperative groups. Targets SM 80+ (Ampere). Requires `-DSTSIM_PY_ENABLE_CUDA_BACKEND=ON`.
- **`stab_avx.hpp`** — AVX vectorization (experimental).

### Key abstractions (`stsim/include/`)

- **`circuit.hpp`** — `Circuit` class: gate list, qubit count, metrics.
- **`gate.hpp`** — `OP` enum with all supported gates (Cliffords, rotations, noise channels, measurement/reset).
- **`state.hpp`** — Base `QuantumState` class that backends inherit from.
- **`backendManager.hpp`** — Factory for creating backend instances at runtime.
- **`config.hpp`** — Global flags: `PRINT_SIM_TRACE`, `ENABLE_NOISE`, `ENABLE_FUSION`, `ENABLE_TENSOR_CORE`, `RANDOM_SEED`.

### Circuit input parsing (`stsim/include/stabsim/src/`)

- **`stim_extraction.hpp`** — Parses Google Stim format into STABSim circuits.
- **`qasm_extraction.hpp`** — Parses OpenQASM format.
- **`T_separation.hpp`** — Non-Clifford T-gate handling via quasiprobability sampling.

### Python bindings (`stsim/python/src/nwqsim_py.cpp`)

pybind11 module exposing `Circuit`, `State`, `SimType` enum (SV, DM, STAB), gate methods, `simulate()`, `measure_all()`. Type stubs in `nwqsim.pyi`.

### Research workspace (`stim-playground/`)

Separate uv project with Jupyter notebooks for surface code experiments, Stim comparisons, and sinter threshold analysis. Independent from the main library build.

## Key Conventions

- **Namespace**: `NWQSim`
- **Type aliases**: `IdxType` = `long long`, `ValType` = `double`
- **C++ standard**: C++17
- **CUDA arch**: SM 80 (Ampere+)
- **CMake minimum**: 4.1
- **Python**: 3.13+
- **Headers are the implementation** — the core library is header-only; there are no `.cpp` source files for the simulator itself.
