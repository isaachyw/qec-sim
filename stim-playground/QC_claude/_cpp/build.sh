#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
VENV_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python3"

# Clean previous build (Python version may have changed)
rm -rf "$BUILD_DIR"

# PYTHON_EXECUTABLE is what pybind11's FindPythonInterp uses
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$VENV_PYTHON"

cmake --build "$BUILD_DIR" --parallel "$(nproc)"

# Copy the .so into QC_claude/ so "from ._circuit import ..." works
cp "$BUILD_DIR"/_circuit*.so "$SCRIPT_DIR/.."

echo "Built _circuit module -> $(ls "$SCRIPT_DIR/../"_circuit*.so)"
