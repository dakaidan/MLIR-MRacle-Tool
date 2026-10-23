# MLIR-MRacle Docker Quickstart

These images run MLIR-MRacle without rebuilding LLVM/MLIR.

Use the ARM64 image on an ARM64 host for Armv8 experiments: the current
metamorphic relations target Armv8-style memory-model behavior. The x86_64 image
is useful for a quick pipeline smoke test on ordinary laptops, but it is not an
x86 memory-model evaluation artifact.

## Load The Image

```sh
zstd -dc mracle-tool-arm64-docker-image.tar.zst | docker load
docker image ls mracle-tool
IMAGE=mracle-tool:arm64
```

On an x86_64 host, the same commands work with
`mracle-tool-x86_64-docker-image.tar.zst` instead, subject to the caveat above;
set `IMAGE=mracle-tool:latest` after loading that archive.

## Smoke Test

Run one tiny JIT campaign and write results to `docker-runs/demo`:

```sh
mkdir -p docker-runs
docker run --rm -v "$PWD/docker-runs:/runs" "$IMAGE" \
  mracle-fuzz \
  --input corpus/seeds/atomic_increment_two.mlir \
  --output /runs/demo \
  --model armv8 \
  --backend jit \
  --comparison outcome \
  --iterations 2 \
  --reps 5 \
  --replay-reps 5 \
  --max-runs 5 \
  --timeout-ms 180000
```

Expected result: the command prints two `trial=... ok ...` lines and a JSON
summary. The output appears under `docker-runs/demo`.

## Inspect Results

```sh
cat docker-runs/demo/summary.json
find docker-runs/demo/runs/00000000 -maxdepth 1 -type f | sort
```

Each run directory contains the saved request, response, verdict, transformed
input, logs, compiler build metadata, and coverage summary.

## Replay A Trial

```sh
docker run --rm -v "$PWD/docker-runs:/runs" "$IMAGE" \
  mracle-replay /runs/demo/runs/00000000/request.json \
  --output /runs/replay-00000000 \
  --reps 5 \
  --replay-reps 5 \
  --max-runs 5 \
  --timeout-ms 180000
```

Expected result: `oracle status: OK`.

## Run A Longer Campaign

```sh
docker run --rm -v "$PWD/docker-runs:/runs" "$IMAGE" \
  mracle-fuzz \
  --input corpus/seeds \
  --output /runs/campaign \
  --model armv8 \
  --backend jit \
  --comparison tolerance \
  --duration-secs 3600
```

Use a new `--output` directory for each campaign; existing directories are not
overwritten.
