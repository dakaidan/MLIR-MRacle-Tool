#!/usr/bin/env python3
"""Deterministic protocol fixture; never used for real MLIR campaigns."""
import json
from pathlib import Path
import sys
import struct
import time

request = json.loads(Path(sys.argv[1]).read_text())
source = Path(request["input"]).read_text()
if "slow-fixture" in source:
    time.sleep(0.6)
status = "WARN" if "warn-fixture" in source else "OK"
if "coverage-fixture" in source:
    trial = int(Path(request["artifacts"]).name)
    with Path(request["artifacts"], "coverage.map").open("r+b") as mapping:
        mapping.write(b"MRACOV01" + struct.pack("<II", 2, 0))
        mapping.seek(16 + min(trial, 1))
        mapping.write(b"\x01")
report = {
    "status": status,
    "transform_applied": True,
    "source_runs": 10,
    "transformed_runs": 10,
    "applied": [{"name": "fixture", "target_function": "main"}],
    "relation": {"type": "equality"},
    "source_outcomes": [{"outcome": [0], "count": 10}],
    "transformed_outcomes": [{"outcome": [0], "count": 10}],
    "transformed_mlir": source + f"\n// fixture mutation {request['seed']}\n",
    "ir_features": ["count:builtin.module:1"],
}
Path(request["artifacts"], "response.json").write_text(json.dumps(report))
