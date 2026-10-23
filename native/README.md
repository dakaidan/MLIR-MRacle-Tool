# Native MLIR-MRacle Library

This directory contains the native C++ MLIR-MRacle library and the narrow C ABI used
by the Rust worker. It has no fuzzing executable or Python campaign runner.
Build and run through the [root instructions](../README.md).

`mlir-mracle/src/lib/` contains the transform registry, memory-model
gating, agitation, LLVM lowering, LLJIT execution, outcome oracle and artifact
serialization. `ffi/` adds the ABI, compiler-coverage collector, and optional
native object/executable backend. CMake builds
`libmracle_ffi.a` and the engine's static component archives; Cargo links those
and the compiler dependencies into `mracle-worker`. OpenMP remains a system
shared-library dependency. ThreadSanitizer is opt-in through
`MRACLE_SANITIZER=thread`; default builds have no TSan dependency. Both JIT and
native execution support optional final TSan probes (`--tsan-triage`).

Seed programs are in [`../corpus/seeds`](../corpus/seeds). The small manual
critical-region reproducers are in [`../examples/reproducers`](../examples/reproducers).
They are deliberately separate from the default fuzzing corpus.

The Rust driver runs the existing compare pipeline through `runPipeline`.
The removed Python/native CLI also exposed emit-only, execute-only, legacy,
and in-place campaign-resume modes. These are not Rust CLI features; the
underlying C++ functions are retained, while a saved Rust queue can seed a new
campaign. Transform application, execution, agitation and oracle comparison
run through the native implementation.

Native oracle and execution-contract test executables are optional and disabled by default. Configure
with `-DBUILD_TESTING=ON` to build them. They are regression tests, not fuzzing
entry points. The native library license remains in [LICENSE](LICENSE).
