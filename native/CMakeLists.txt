cmake_minimum_required(VERSION 3.24)

project(MLIRMetamorphicTestingSuperbuild LANGUAGES C CXX)

option(BUILD_TESTING "Build the native oracle regression tests" OFF)
include(CTest)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(MRACLE_ENABLE_TSAN "Instrument the engine and enable optional TSan triage" OFF)
if(MRACLE_ENABLE_TSAN)
    add_compile_definitions(MRACLE_WITH_TSAN=1)
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ffi)

find_package(MLIR CONFIG REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_library(llvm_mlir_headers INTERFACE)
target_include_directories(llvm_mlir_headers INTERFACE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

add_subdirectory(mlir-mracle)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
add_subdirectory(ffi)
