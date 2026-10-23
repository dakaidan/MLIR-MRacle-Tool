#!/usr/bin/env python3
"""Check failure propagation using the optional native execution test binary."""
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tempfile


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


binary = str(Path(sys.argv[1]).resolve())
sanitizer = sys.argv[2] if len(sys.argv) > 2 else "none"
# LLVM lowers llvm.trap to BRK on AArch64 and UD2 on x86-64.
trap_signal = signal.SIGTRAP if platform.machine().lower() in ("aarch64", "arm64") else signal.SIGILL
env = dict(os.environ, TSAN_OPTIONS="ignore_noninstrumented_modules=1:halt_on_error=0:exitcode=66")
with tempfile.TemporaryDirectory(prefix="mracle-native-test-") as directory:
    for backend in ("jit", "native"):
        args = [binary, directory, backend, sanitizer]
        if sanitizer == "thread":
            race = subprocess.run(args + ["race"], env=env, capture_output=True, timeout=30, preexec_fn=no_core)
            assert race.returncode == 66, (backend, race.returncode, race.stderr.decode())
            assert b"ThreadSanitizer: data race" in race.stderr, race.stderr.decode()
        else:
            crash = subprocess.run(args + ["crash"], env=env, capture_output=True, timeout=30, preexec_fn=no_core)
            assert crash.returncode == -trap_signal, (backend, crash.returncode, crash.stderr.decode())
            with subprocess.Popen(args + ["hang"], env=env, start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  preexec_fn=no_core) as process:
                try:
                    process.wait(timeout=2)
                    raise AssertionError(f"{backend} hang completed unexpectedly")
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                    assert process.returncode == -signal.SIGKILL
print(f"Execution failure checks passed ({sanitizer})")
