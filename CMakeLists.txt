# CMakeLists.txt — LOB-Core Component C (zero-copy IPC bridge).
#
# Builds the pybind11 module 'lob_bridge' plus the standalone 'producer_demo'.
# pybind11_DIR and the Python interpreter are wired to the project's .venv by
# scripts/build_cpp.sh (it passes -Dpybind11_DIR=$(.venv/bin/python -m pybind11
# --cmakedir) and -DPython3_EXECUTABLE=.venv/bin/python).

cmake_minimum_required(VERSION 3.20)
project(lob_core_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -O3 + native tuning for the M-series performance cores on the hot path.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=native -DNDEBUG")

# Locate the venv Python first so pybind11 binds against the right ABI.
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11_DIR is provided on the command line by build_cpp.sh. If absent, try
# to discover it from the active interpreter.
if(NOT pybind11_DIR)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(pybind11 CONFIG REQUIRED)

# --- Shared-memory + ring core (used by both the module and the demo) --------
add_library(lob_shm STATIC src/cpp/shm_lob.cpp)
target_include_directories(lob_shm PUBLIC src/cpp)
set_property(TARGET lob_shm PROPERTY POSITION_INDEPENDENT_CODE ON)

# --- pybind11 module: lob_bridge --------------------------------------------
pybind11_add_module(lob_bridge src/bindings/lob_bridge.cpp)
target_link_libraries(lob_bridge PRIVATE lob_shm)
target_include_directories(lob_bridge PRIVATE src/cpp)

# --- standalone synthetic producer ------------------------------------------
add_executable(producer_demo src/cpp/producer_demo.cpp)
target_link_libraries(producer_demo PRIVATE lob_shm)
target_include_directories(producer_demo PRIVATE src/cpp)

# --- MCTS ring-variant benchmark (Part IV Milestone 1) ----------------------
find_package(Threads REQUIRED)
add_executable(bench_rings src/cpp/bench_rings.cpp)
target_link_libraries(bench_rings PRIVATE Threads::Threads)
