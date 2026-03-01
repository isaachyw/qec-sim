cmake_minimum_required(VERSION 3.18)
project(_circuit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# pybind11 via FetchContent (no pip install needed)
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_circuit circuit_module.cpp)
target_compile_options(_circuit PRIVATE -O3)

# Install into QC_claude package directory (one level up from _cpp/)
install(TARGETS _circuit LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
