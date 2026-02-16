cmake_minimum_required(VERSION 4.1)

if(NOT DEFINED CUDAToolkit_ROOT)
    foreach(_cuda_root IN ITEMS "/usr/local/cuda-12.8" "/usr/local/cuda-12" "/usr/local/cuda")
        if(EXISTS "${_cuda_root}")
            set(CUDAToolkit_ROOT "${_cuda_root}")
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(_nvcc_path
        NAMES nvcc
        HINTS
        "${CUDAToolkit_ROOT}/bin"
        "/usr/local/cuda-12.8/bin"
        "/usr/local/cuda-12/bin"
        "/usr/local/cuda/bin"
    )

    if(_nvcc_path)
        set(CMAKE_CUDA_COMPILER "${_nvcc_path}" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

project(stabsim-python LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(STSIM_PY_ENABLE_CUDA_BACKEND "Enable CUDA backend in nwqsim Python module" OFF)

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
find_package(OpenMP QUIET)
find_package(MPI QUIET)

find_package(CUDAToolkit QUIET)

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
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    target_link_libraries(nwqsim PRIVATE CUDA::cudart)
    target_include_directories(nwqsim PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    if(STSIM_PY_ENABLE_CUDA_BACKEND)
        # nwqsim_py.cpp includes CUDA kernel launch syntax when CUDA backend is enabled.
        set_source_files_properties(src/nwqsim_py.cpp PROPERTIES LANGUAGE CUDA)
        set_target_properties(nwqsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # STAB CUDA kernels use intrinsics that require newer SM targets.
        set_target_properties(nwqsim PROPERTIES CUDA_ARCHITECTURES "80")
        target_compile_options(nwqsim PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fpermissive>
            $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
        )
        target_compile_definitions(nwqsim PRIVATE CUDA_ENABLED)
    else()
        message(STATUS "Building with CUDA toolkit linked but CUDA backend bindings disabled (set -DSTSIM_PY_ENABLE_CUDA_BACKEND=ON to enable).")
    endif()
endif()

if(HIP_FOUND)
    target_compile_definitions(nwqsim PRIVATE HIP_ENABLED)
endif()

set_target_properties(nwqsim PROPERTIES OUTPUT_NAME "nwqsim")
