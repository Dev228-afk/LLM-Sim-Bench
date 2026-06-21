cmake_minimum_required(VERSION 3.20)
project(LLMSimBench
    VERSION     0.1.0
    DESCRIPTION "High-fidelity behavioural simulator for LLM inference serving"
    LANGUAGES   CXX
)

# ── C++20 Standard ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings & debug symbols ───────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    # Debug symbols are included by default in Debug builds on MSVC
else()
    add_compile_options(-Wall -Wextra -Wpedantic -g)
endif()

# ── Find or fetch pybind11 ──────────────────────────────────────────────────
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── Header-only core library (interface target) ────────────────────────────
# All C++ simulation logic lives in headers under src/
add_library(llmsimbench_headers INTERFACE)
target_include_directories(llmsimbench_headers
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ── Python extension module ─────────────────────────────────────────────────
pybind11_add_module(llmsimbench_core
    src/bindings/pybind_module.cpp
)
target_link_libraries(llmsimbench_core PRIVATE llmsimbench_headers)

# Install the extension next to the Python scripts
set_target_properties(llmsimbench_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
)
# On multi-config generators (MSVC), also set per-config output dirs
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${config} CONFIG_UPPER)
    set_target_properties(llmsimbench_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}/python"
        RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
endforeach()

# ── C++ test executable ─────────────────────────────────────────────────────
add_executable(test_des_engine tests/test_des_engine.cpp)
target_link_libraries(test_des_engine PRIVATE llmsimbench_headers)

enable_testing()
add_test(NAME DESEngineTests COMMAND test_des_engine)

# ── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== LLM Sim Bench v${PROJECT_VERSION} ===")
message(STATUS "  C++ Standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  pybind11     : ${pybind11_VERSION}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "")
