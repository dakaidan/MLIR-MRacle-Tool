#!/usr/bin/env python3
"""Exercise the real coverage ABI; requires Clang, but not LLVM/MLIR libraries."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
compiler = os.environ.get("CXX", "clang++-21")
with tempfile.TemporaryDirectory(prefix="mracle-coverage-") as temporary:
    tmp = Path(temporary)
    subprocess.run([compiler, "-std=c++17", "-c", str(root / "native/ffi/coverage.cpp"),
                    "-o", str(tmp / "collector.o")], check=True)
    subprocess.run([compiler, "-std=c++17", "-O1", "-fsanitize-coverage=trace-pc-guard",
                    "-c", str(root / "tests/fixtures/coverage_target.cpp"),
                    "-o", str(tmp / "target.o")], check=True)
    subprocess.run([compiler, str(tmp / "target.o"), str(tmp / "collector.o"),
                    "-o", str(tmp / "target")], check=True)
    def run(value):
        coverage = tmp / "coverage.map"
        coverage.write_bytes(bytes(16 + 4096))
        result = subprocess.run([str(tmp / "target"), str(value)],
                                env={**os.environ, "MRACLE_COVERAGE_FILE": str(coverage)})
        data = coverage.read_bytes()
        assert data[:8] == b"MRACOV01"
        guards = struct.unpack_from("<I", data, 8)[0]
        assert 0 < guards < 4096 and data[12] == 0
        hits = {i for i, value in enumerate(data[16:16 + guards]) if value}
        assert hits
        return result.returncode, guards, hits
    zero, one, crash = run(0), run(1), run(2)
    assert zero[0] == one[0] == 0 and crash[0] < 0
    assert zero[1] == one[1] == crash[1]
    assert zero[2] != one[2], "branches must produce different edge sets"
    assert run(0)[2] == zero[2], "guard IDs must be stable across fresh processes"
    print(f"coverage ABI passed: {zero[1]} guards; branch novelty and crash persistence verified")
