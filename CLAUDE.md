# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STABSim is a quantum stabilizer simulator for quantum error correction (QEC) with GPU integration. It uses the tableau formalism (2n+1 × 2n representation) for Clifford simulation, with backends for CPU and CUDA GPU. The Python module is called `nwqsim`. It is under stsim/
stim-playground/QC_claude is a quantum circuit simulator that uses the QC_claude library to simulate quantum circuits. It is under stim-playground/QC_claude.
other files are for scratch for now

## environment
I use uv to manage the environment. The venv under stim-playground/.venv/bin/python3 (ther) is the one for developing QC_claude and stim-playground. The one under stsim/.venv/bin/active is the one for using stabsim.

## Code style
when developing the code, whenever you encounter some big uncertain, ask me first.

you don't need to worry about the robustness or the compatibility of the code for different platforms for now, the code is for scratch and research not for industrial use.