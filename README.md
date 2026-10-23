# MLIR-MRacle

MLIR-MRacle is a metamorphic testing tool for concurrent MLIR programs. It
combines a Rust campaign driver with a native MLIR/OpenMP pipeline that applies
metamorphic relations, lowers through MLIR/LLVM, executes the original and
transformed programs, and compares their observed outcomes.

The current relations focus on Armv8-style weak-memory behaviour. The JIT
backend is the default execution path; a native-object backend is also available
for separate process execution and optional ThreadSanitizer triage.

## Build

Prerequisites on Linux:

* Rust 1.93.1, via `rust-toolchain.toml`
* Git, CMake, Ninja, Python 3
* Clang/LLVM 21 with libomp and libatomic

Build the release binaries:

```sh
cargo build --locked --release
```

The first build fetches the pinned LLVM/MLIR revision
`373c54d763c98487df27599247412f80d933e177`, builds the required compiler
libraries with coverage instrumentation, builds the native MLIR-MRacle
libraries, and links:

* `target/release/mracle-fuzz`
* `target/release/mracle-worker`
* `target/release/mracle-replay`

Build artifacts are cached under `build/unified` by default. Useful build
overrides:

| Variable | Meaning |
| --- | --- |
| `MRACLE_BUILD_JOBS` | Native/LLVM build parallelism; default `2`. |
| `MRACLE_BUILD_DIR` | Build cache directory; default `build/unified`. |
| `MRACLE_SANITIZER` | `none` or `thread`; default `none`. |
| `CC`, `CXX` | Host Clang commands; defaults `clang-21`, `clang++-21`. |
| `MRACLE_LLVM_SOURCE` | Existing llvm-project checkout to build. |
| `MRACLE_LLVM_PREFIX` | Existing compatible LLVM/MLIR install prefix. |

Inspect the embedded compiler configuration with:

```sh
target/release/mracle-worker --build-info
```

## Docker

A Dockerfile is provided for local reproduction without installing the toolchain
directly:

```sh
docker build -t mracle-tool .
```

Prebuilt Docker archives can be loaded with `docker load`; see
`DOCKER_QUICKSTART.md`.

## Run

Run a small campaign:

```sh
target/release/mracle-fuzz \
  --input corpus/seeds \
  --output findings \
  --model armv8 \
  --backend jit \
  --comparison outcome \
  --iterations 10000
```

The output directory must not already exist. Important outputs are:

* `summary.json`: campaign totals and stop reason.
* `events.jsonl`: machine-readable progress log.
* `queue/`: retained source programs and provenance.
* `runs/`: per-trial requests, verdicts, logs, MLIR/LLVM artifacts, and coverage.
* `findings/`: references to oracle failures, warnings, crashes, timeouts, and
  pipeline failures.
* `sources/`: per-source saturation evidence.

Replay a saved trial with a larger sampling budget:

```sh
target/release/mracle-replay findings/runs/00000000/request.json \
  --output replay-00000000 \
  --reps 500 \
  --replay-reps 500 \
  --max-runs 20000
```

Concurrency outcomes are not expected to reproduce exactly. A finding is a
candidate for investigation, not by itself a confirmed compiler defect.

## Main Options

Common campaign controls:

* `--transform NAME`: restrict to one transform; empty uses the full registry.
* `--model armv8`: select the memory-model transform set.
* `--backend jit|native`: execute via LLJIT or linked native executables.
* `--comparison outcome|tolerance|adaptive`: choose the oracle.
* `--duration-secs N`: stop starting new trials after `N` seconds.
* `--stop-file PATH`: finish cleanly after the active trial once the file exists.
* `--reps`, `--replay-reps`, `--max-runs`: sampling budgets.
* `--coverage-artifacts summary|full|none`: retained compiler coverage detail.
* `--tsan-triage`: final TSan probes for inconclusive trials; requires a
  `MRACLE_SANITIZER=thread` build.

`--comparison tolerance` and `--comparison adaptive` use campaign-style
distributional comparisons. Their additional controls are
`--campaigns`, `--max-campaigns`, `--executions-per-campaign`,
`--team-size`, `--bootstrap-replicates`, `--tvd-threshold`,
`--min-change`, and `--confidence-level`.

## Layout

* `src/`: Rust campaign driver, replay command, protocol types, coverage
  collection, and worker supervision.
* `crates/worker/`: Rust wrapper around the native C ABI.
* `native/ffi/`: C ABI, coverage bridge, and native executable backend.
* `native/mlir-mracle/src/lib/`: MLIR transforms, memory-model gating,
  lowering, execution, agitation, oracles, and artifact serialization.
* `corpus/seeds/`: seed MLIR programs.
* `scripts/cloudlab_campaigns.py`: CloudLab Portal API driver for ARM campaign
  runs and figure generation.

## Checks

Fast Rust checks:

```sh
cargo fmt --all -- --check
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features --all-targets
python3 tests/coverage_runtime.py
```

Native checks after a full build:

```sh
cmake -S native -B build/unified/native -DBUILD_TESTING=ON
cmake --build build/unified/native --target \
  mlir_mracle_oracle_test mlir_mracle_legacy_oracle_test mracle_execution_test
ctest --test-dir build/unified/native --output-on-failure
python3 tests/native_backend.py build/unified/native/ffi/mracle_execution_test
```

## CloudLab

The CloudLab helper reads credentials from `.env`:

```sh
cp .env.example .env
$EDITOR .env
python3 scripts/cloudlab_campaigns.py run
```

Use a CloudLab Portal API token as `CLOUDLAB_TOKEN`, the project that can
allocate m400 nodes as `CLOUDLAB_PROJECT`, and your login as `CLOUDLAB_USER`.
The default campaign matrix stays under 25 runs and assigns at most four
concurrent runs per m400 node. Results and figures are written under
`build/cloudlab/eval` unless another output directory is selected.

## License

The Rust driver and integration code are MIT licensed. The native library keeps
its original license in `native/LICENSE`.
