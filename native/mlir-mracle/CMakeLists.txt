enable_language(CXX)

# Optimisation level for all mlir_mracle targets. Optional ThreadSanitizer
# instrumentation is selected by the Cargo build; ordinary builds omit it.
set(MLIR_MRACLE_OPT_LEVEL "2" CACHE STRING
    "Optimisation level for mlir_mracle targets")
add_compile_options(-O${MLIR_MRACLE_OPT_LEVEL})
if(MRACLE_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# Apple clang does not auto-detect Homebrew's libomp; hint FindOpenMP when
# present. Done here so all native libraries resolve the same
# OpenMP runtime.
if(APPLE AND NOT OpenMP_ROOT)
    foreach(dir /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS "${dir}/lib" AND EXISTS "${dir}/include/omp.h")
            set(OpenMP_ROOT "${dir}")
            break()
        endif()
    endforeach()
endif()
find_package(OpenMP REQUIRED)

add_subdirectory(src/lib)

# Optional: uncomment when metamorphic transforms need CONQuER's TOSA
# quantisation passes or custom dialect directly.
# target_link_libraries(mlir_mracle INTERFACE conquer_lib)

target_include_directories(mlir_mracle INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
