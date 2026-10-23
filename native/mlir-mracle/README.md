# mlir-mracle

The C++ library root for the MLIR-MRacle project. It provides MLIR parsing,
the metamorphic transform registry, pipeline orchestration, JIT lowering and
execution, and the outcome oracles: the per-outcome agitation judge, the
campaign-based distributional comparison, and the noise-calibrated comparison.
The default pipeline runs the agitation judge; `ComparisonMethod` selects a
campaign method instead.

`mlir_mracle` is a CMake INTERFACE target that aggregates the static component
libraries under `src/lib/`. The C ABI archive in `../ffi/` consumes it; Cargo
links the archives into the Rust worker. There is no native fuzzing CLI.
See the [repository README](../../README.md) for build and usage instructions.
