cmake_minimum_required(VERSION 4.1)
project(stabsim-python)

# Prefer the local uv virtual environment if present.
if(NOT DEFINED VENV_PATH AND EXISTS "${PROJECT_SOURCE_DIR}/../.venv/bin/python3")
    set(VENV_PATH "${PROJECT_SOURCE_DIR}/../.venv")
endif()

# Use venv Python if VENV_PATH is set, otherwise use system Python.
if(DEFINED VENV_PATH)
    set(Python3_EXECUTABLE "${VENV_PATH}/bin/python3")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Get pybind11 cmake dir from Python environment
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_result
)

if(NOT pybind11_result EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install it with: pip install pybind11")
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(nwqsim MODULE src/nwqsim_py.cpp)

target_compile_features(nwqsim PRIVATE cxx_std_17)

target_include_directories(nwqsim PRIVATE ${PROJECT_SOURCE_DIR}/../include)

if(OpenMP_FOUND)
    target_link_libraries(nwqsim PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(nwqsim PRIVATE OMP_ENABLED)
endif()

if(MPI_FOUND)
    target_link_libraries(nwqsim PRIVATE MPI::MPI_CXX)
    target_compile_definitions(nwqsim PRIVATE MPI_ENABLED)
endif()

if(CUDAToolkit_FOUND)
    target_compile_definitions(nwqsim PRIVATE CUDA_ENABLED)
endif()

if(HIP_FOUND)
    target_compile_definitions(nwqsim PRIVATE HIP_ENABLED)
endif()

set_target_properties(nwqsim PROPERTIES OUTPUT_NAME "nwqsim")
