add_library(mracle_runner OBJECT native_runner.c)
target_link_libraries(mracle_runner PRIVATE OpenMP::OpenMP_C)
add_library(mracle_ffi STATIC mracle.cpp coverage.cpp native_backend.cpp)
add_dependencies(mracle_ffi mracle_runner)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/runner_bytes.h"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=$<TARGET_OBJECTS:mracle_runner>"
        "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/runner_bytes.h"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/EmbedRunner.cmake"
    DEPENDS mracle_runner $<TARGET_OBJECTS:mracle_runner> EmbedRunner.cmake
    VERBATIM)
target_sources(mracle_ffi PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/runner_bytes.h")
target_include_directories(mracle_ffi PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(mracle_ffi PRIVATE
    MRACLE_NATIVE_CC="${CMAKE_C_COMPILER}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/native-cc.txt" CONTENT "${CMAKE_C_COMPILER}\n")
set_source_files_properties(coverage.cpp PROPERTIES COMPILE_OPTIONS "-fno-sanitize=all")
set_target_properties(mracle_ffi PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
target_compile_options(mracle_ffi PRIVATE -fexceptions)
if(MRACLE_ENABLE_TSAN)
    target_compile_options(mracle_ffi PRIVATE -fsanitize=thread)
endif()
target_link_libraries(mracle_ffi PRIVATE MLIRMRACLE::mlir_mracle OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})

# Rust needs the same dynamic sanitizer runtime as the C++ objects, loaded
# at worker startup. Never load a second (e.g. GCC + Clang) TSan runtime.
set(tsan_library "")
if(MRACLE_ENABLE_TSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(tsan_name "libclang_rt.tsan-${CMAKE_SYSTEM_PROCESSOR}.so")
    else()
        set(tsan_name "libtsan.so")
    endif()
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${tsan_name}"
        OUTPUT_VARIABLE tsan_library OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    if(NOT EXISTS "${tsan_library}")
        message(FATAL_ERROR "Shared ThreadSanitizer runtime unavailable: ${tsan_library}")
    endif()
endif()
target_compile_definitions(mracle_ffi PRIVATE MRACLE_TSAN_RUNTIME="${tsan_library}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tsan-library.txt"
    CONTENT "${tsan_library}\n")

include(StaticLink.cmake)
mracle_export_static_link(mracle_ffi)

if(BUILD_TESTING)
    add_executable(mracle_execution_test tests/execution.cpp coverage.cpp)
    target_link_libraries(mracle_execution_test PRIVATE mracle_ffi
        MLIRMRACLE::mlir_mracle OpenMP::OpenMP_CXX)
    target_compile_features(mracle_execution_test PRIVATE cxx_std_23)
    target_link_options(mracle_execution_test PRIVATE -Wl,--export-dynamic)
    if(MRACLE_ENABLE_TSAN)
        get_filename_component(tsan_dir "${tsan_library}" DIRECTORY)
        target_link_options(mracle_execution_test PRIVATE
            -fsanitize=thread -shared-libsan "-Wl,-rpath,${tsan_dir}")
    endif()
    foreach(backend jit native)
        add_test(NAME execution_${backend} COMMAND mracle_execution_test
            "${CMAKE_CURRENT_BINARY_DIR}/test-${backend}" ${backend} none normal)
        set_tests_properties(execution_${backend} PROPERTIES TIMEOUT 30
            ENVIRONMENT "TSAN_OPTIONS=ignore_noninstrumented_modules=1")
        if(MRACLE_ENABLE_TSAN)
            add_test(NAME execution_${backend}_tsan COMMAND mracle_execution_test
                "${CMAKE_CURRENT_BINARY_DIR}/test-${backend}-tsan" ${backend} thread normal)
            set_tests_properties(execution_${backend}_tsan PROPERTIES TIMEOUT 30
                ENVIRONMENT "TSAN_OPTIONS=ignore_noninstrumented_modules=1")
        endif()
    endforeach()
endif()
