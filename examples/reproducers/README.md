# Manual critical-region reproducers

These six text files are manual reductions from the original debug corpus. They
exercise inserted and nested OpenMP `omp.critical` regions and contain reduced
variants for manual investigation. This is consistent with the nested-critical
verifier/hang issue described in Section 4 of the supplied paper; the files
alone do not identify an upstream issue number or establish current behavior.

* `seed7_insert_critical.mlir`: atomic increment example with critical regions.
* `seed7_nested_critical.mlir`: a related program containing nested critical regions.
* `seed7_nested_critical_copy.mlir` and `.llvm`: small nested-critical reductions.
* `seed7_nested_critical_copy_2.mlir` and `.llvm`: corresponding variants with
  one critical region.

The `.llvm` files are **textual MLIR**, using `llvm.func`/`llvm.return` alongside
OpenMP operations. They are neither executables nor ordinary LLVM `.ll` files.
Names and contents are preserved for traceability.

These are debugging artifacts, not an automated regression suite. Keep them
outside `corpus/seeds`: nested critical constructs may be rejected or hang,
and some reductions have a different entry-point signature from normal seeds.
