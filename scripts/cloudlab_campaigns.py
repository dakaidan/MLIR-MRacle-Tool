#!/usr/bin/env python3
"""Submit, run, collect, and summarize a reviewer-sized CloudLab evaluation.

The default plan is intentionally modest: 8 exposed
configurations, 3 repetitions each, capped at 24 runs.  Each m400 node receives
at most 4 concurrent runs and fuzzes for 12 hours after the local build.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API_BASE = "https://boss.emulab.net:43794"
DEFAULT_PROFILE = "small-lan"
DEFAULT_PROFILE_PROJECT = "PortalProfiles"
DEFAULT_DURATION = 12 * 60 * 60
DEFAULT_EXPERIMENT_HOURS = 16
DEFAULT_CONCURRENCY = 4
DEFAULT_REPS = 3
DEFAULT_MAX_RUNS = 24
EXPERIMENT_NAME_MAX = 16
DEFAULT_LAUNCH_WORKERS = 3
DEFAULT_MATRIX = [
    ("outcome", "jit", "none"),
    ("outcome", "native", "none"),
    ("outcome", "jit", "thread"),
    ("outcome", "native", "thread"),
    ("tolerance", "jit", "none"),
    ("tolerance", "native", "none"),
    ("adaptive", "jit", "none"),
    ("adaptive", "native", "none"),
]

COMPARISON_ALIASES = {
    "agitation": "outcome",
    "outcome": "outcome",
    "tvd": "tolerance",
    "tolerance": "tolerance",
    "calibrated": "adaptive",
    "adaptive": "adaptive",
}


def canonical_comparison(value: str) -> str:
    try:
        return COMPARISON_ALIASES[value]
    except KeyError as error:
        valid = ", ".join(sorted(COMPARISON_ALIASES))
        raise argparse.ArgumentTypeError(f"unknown comparison {value!r}; expected one of {valid}") from error


def parse_matrix(value: str) -> list[tuple[str, str, str]]:
    matrix = []
    for raw in value.split(","):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split("/")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("matrix entries must be comparison/backend/sanitizer")
        comparison = canonical_comparison(parts[0])
        backend = parts[1]
        sanitizer = parts[2]
        if backend not in {"jit", "native"}:
            raise argparse.ArgumentTypeError(f"unknown backend {backend!r}")
        if sanitizer not in {"none", "thread"}:
            raise argparse.ArgumentTypeError(f"unknown sanitizer {sanitizer!r}")
        if comparison != "outcome" and sanitizer == "thread":
            raise argparse.ArgumentTypeError("thread sanitizer triage is only supported for outcome jobs")
        matrix.append((comparison, backend, sanitizer))
    if not matrix:
        raise argparse.ArgumentTypeError("matrix must contain at least one entry")
    return matrix


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    digits = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def default_experiment_name() -> str:
    return f"mr{base36(int(time.time()))}{base36(os.getpid() % 1296)}"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load(path: Path):
    return json.loads(path.read_text())


def load_env_file(path: Path, *, explicit: bool = False) -> None:
    if not path.exists():
        if explicit:
            raise SystemExit(f"env file not found: {path}")
        return
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"{path}:{number}: expected NAME=value")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name:
            raise SystemExit(f"{path}:{number}: empty variable name")
        os.environ.setdefault(name, value)


def env_file_from_argv(argv: list[str]) -> tuple[Path, bool]:
    for index, value in enumerate(argv):
        if value == "--env-file" and index + 1 < len(argv):
            return Path(argv[index + 1]), True
        if value.startswith("--env-file="):
            return Path(value.split("=", 1)[1]), True
    return Path(".env"), False


def digest(stream) -> str:
    if hasattr(hashlib, "file_digest"):
        return hashlib.file_digest(stream, "sha256").hexdigest()
    result = hashlib.sha256()
    for data in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(data)
    return result.hexdigest()


def run(command, *, cwd=None, env=None, log=None, check=True) -> subprocess.CompletedProcess:
    if log is None:
        return subprocess.run(command, cwd=cwd, env=env, text=True, check=check)
    with log.open("a") as stream:
        stream.write(f"\n$ {' '.join(map(str, command))}\n")
        stream.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=check,
        )


def archive_trial(directory: Path) -> str:
    archive_dir = directory.parent.parent / "archived-runs"
    archive_dir.mkdir(exist_ok=True)
    destination = archive_dir / f"{directory.name}.tar.gz"
    temporary = destination.with_suffix(".partial")
    if destination.exists():
        raise RuntimeError(f"archive already exists: {destination}")
    original = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"unexpected symlink in trial artifacts: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                original[str(path.relative_to(directory.parent))] = digest(stream)
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        archive.add(directory, arcname=directory.name)
    restored = {}
    with tarfile.open(temporary, "r:gz") as archive:
        for member in archive:
            if member.isfile():
                with archive.extractfile(member) as stream:
                    restored[member.name] = digest(stream)
    if restored != original:
        raise RuntimeError(f"archive verification failed: {directory}")
    temporary.replace(destination)
    shutil.rmtree(directory)
    return str(destination)


def default_jobs(reps: int, max_runs: int, matrix: list[tuple[str, str, str]] | None = None) -> list[dict]:
    matrix = matrix or DEFAULT_MATRIX
    jobs = []
    for rep in range(reps):
        for comparison, backend, sanitizer in matrix:
            if len(jobs) >= max_runs:
                return jobs
            jobs.append(
                {
                    "id": f"r{rep:02d}-{comparison}-{backend}-{sanitizer}",
                    "rep": rep,
                    "comparison": comparison,
                    "backend": backend,
                    "sanitizer": sanitizer,
                    "seed": 7000 + rep * 101 + len(jobs),
                }
            )
    return jobs


def split_jobs(jobs: list[dict], nodes: int) -> list[list[dict]]:
    buckets = [[] for _ in range(nodes)]
    for index, job in enumerate(jobs):
        buckets[index % nodes].append(job)
    return buckets


def job_label(job: dict) -> str:
    return f"{job['comparison']}/{job['backend']}/{job['sanitizer']}"


class CloudLabClient:
    def __init__(self, base: str, token: str, retries: int = 3):
        self.base = base.rstrip("/")
        self.token = token
        self.retries = retries

    def url(self, path: str) -> str:
        path_part, separator, query = path.partition("?")
        if not path_part.endswith("/"):
            path_part += "/"
        return self.base + path_part + (separator + query if separator else "")

    def request(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        last = None
        for attempt in range(self.retries):
            url = self.url(path)
            for redirect in range(5):
                req = urllib.request.Request(url, data=data, method=method)
                req.add_header("X-Api-Token", self.token)
                req.add_header("Accept", "application/json")
                req.add_header("Content-Type", "application/json")
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        text = response.read().decode()
                        return json.loads(text) if text else {}
                except urllib.error.HTTPError as error:
                    if error.code in (307, 308):
                        location = error.headers.get("Location")
                        if not location:
                            raise RuntimeError(f"{method} {path} failed: HTTP {error.code}: missing Location") from error
                        if redirect == 4:
                            raise RuntimeError(f"{method} {path} failed: too many redirects") from error
                        url = urllib.parse.urljoin(url, location)
                        continue
                    detail = error.read().decode(errors="replace")
                    if error.code != 429 or attempt + 1 >= self.retries:
                        raise RuntimeError(f"{method} {path} failed: HTTP {error.code}: {detail}") from error
                    last = error
                    break
                except OSError as error:
                    if attempt + 1 >= self.retries:
                        raise
                    last = error
                    break
            time.sleep(2**attempt)
        raise RuntimeError(f"{method} {path} failed after retries: {last}")

    def create_experiment(self, body: dict) -> dict:
        return self.request("POST", "/experiments", body)

    def get_experiment(self, experiment_id: str) -> dict:
        return self.request("GET", f"/experiments/{experiment_id}")

    def extend_experiment(self, experiment_id: str, extend_by_hours: int, reason: str = "") -> dict:
        body = {"extend_by": extend_by_hours}
        if reason:
            body["reason"] = reason
        return self.request("PUT", f"/experiments/{experiment_id}", body)

    def manifests(self, experiment_id: str) -> dict:
        return self.request("GET", f"/experiments/{experiment_id}/manifests")

    def delete_experiment(self, experiment_id: str) -> None:
        self.request("DELETE", f"/experiments/{experiment_id}")


def manifest_nodes(manifests: dict) -> list[dict]:
    nodes = {}
    for aggregate, xml in manifests.items():
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        for node in root.findall(".//{*}node"):
            client_id = node.attrib.get("client_id")
            hostname = node.attrib.get("hostname")
            if not hostname:
                host = node.find(".//{*}host")
                hostname = host.attrib.get("name") if host is not None else None
            if client_id and hostname:
                nodes[client_id] = {"client_id": client_id, "hostname": hostname, "aggregate": aggregate}
    return [nodes[key] for key in sorted(nodes)]


def aggregate_nodes(experiment: dict) -> list[dict]:
    nodes = {}
    for aggregate in experiment.get("aggregates", {}).values():
        for node in aggregate.get("nodes", []):
            if node.get("hostname"):
                nodes[node["client_id"]] = {
                    "client_id": node.get("client_id"),
                    "hostname": node.get("hostname"),
                    "ipv4": node.get("ipv4"),
                    "status": node.get("status"),
                }
    return [nodes[key] for key in sorted(nodes)]


def wait_ready(client: CloudLabClient, experiment_id: str, timeout_secs: int, poll_secs: int) -> dict:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        experiment = client.get_experiment(experiment_id)
        status = experiment.get("status")
        if status == "ready":
            return experiment
        if status == "failed":
            raise RuntimeError(f"CloudLab experiment failed: {experiment}")
        print(f"{stamp()} experiment {experiment_id} status={status}", flush=True)
        time.sleep(poll_secs)
    raise TimeoutError(f"experiment {experiment_id} did not become ready within {timeout_secs}s")


def maybe_extend_experiment(args, client: CloudLabClient, state: dict) -> None:
    if args.extend_days <= 0:
        return
    extend_by_hours = int(args.extend_days * 24)
    if extend_by_hours < 1:
        raise SystemExit("--extend-days must be at least 1/24 when set")
    requested = {
        "extend_days": args.extend_days,
        "extend_by_hours": extend_by_hours,
        "reason": args.extend_reason,
    }
    for extension in state.get("extensions", []):
        if all(extension.get(key) == value for key, value in requested.items()):
            print(f"experiment already extended by {extend_by_hours}h for this state; skipping")
            return
    experiment = client.extend_experiment(state["experiment"]["id"], extend_by_hours, args.extend_reason)
    state.setdefault("extensions", []).append(
        {
            "time": stamp(),
            **requested,
            "experiment": experiment,
        }
    )
    save(args.state, state)
    expires_at = experiment.get("expires_at", "unknown")
    print(f"extended experiment {state['experiment']['id']} by {extend_by_hours}h; expires_at={expires_at}")


def submit(args) -> dict:
    token = args.token or os.environ.get("CLOUDLAB_TOKEN")
    if not token:
        raise SystemExit("set CLOUDLAB_TOKEN or pass --token")
    if not args.project:
        raise SystemExit("set CLOUDLAB_PROJECT or pass --project")
    if len(args.name) > EXPERIMENT_NAME_MAX:
        raise SystemExit(f"--name must be at most {EXPERIMENT_NAME_MAX} characters for CloudLab")
    if args.experiment_hours < 1 or args.experiment_hours > 16:
        raise SystemExit("--experiment-hours must be between 1 and 16 for this CloudLab project")
    jobs = default_jobs(args.reps, args.max_runs, args.matrix)
    nodes = min(args.max_nodes, max(1, math.ceil(len(jobs) / args.concurrency)))
    client = CloudLabClient(args.api_base, token)
    bindings = {
        "bestEffort": "False",
        "exclusiveVMs": "True",
        "linkSpeed": "0",
        "nodeCount": str(nodes),
        "osImage": args.os_image,
        "phystype": args.node_type,
        "sameSwitch": "False",
        "startVNC": "False",
        "tempFileSystemMax": "False",
        "tempFileSystemMount": "/mydata",
        "tempFileSystemSize": "0",
        "useVMs": "False",
    }
    body = {
        "name": args.name,
        "project": args.project,
        "profile_name": args.profile,
        "profile_project": args.profile_project,
        "duration": args.experiment_hours,
        "bindings": bindings,
    }
    if args.group:
        body["group"] = args.group
    if args.sshpubkey:
        body["sshpubkey"] = Path(args.sshpubkey).expanduser().read_text()
    experiment = client.create_experiment(body)
    state = {
        "created": stamp(),
        "api_base": args.api_base,
        "project": args.project,
        "profile": {"name": args.profile, "project": args.profile_project},
        "experiment": experiment,
        "bindings": bindings,
        "jobs": jobs,
        "node_count": nodes,
        "concurrency": args.concurrency,
        "duration_secs": args.duration_secs,
    }
    save(args.state, state)
    print(f"submitted {experiment['id']} ({experiment.get('url', 'no portal url')})")
    return state


def ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}" if user else host


def ssh_base(args) -> list[str]:
    command = ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if args.ssh_key:
        command += ["-i", str(args.ssh_key)]
    return command


def scp_base(args) -> list[str]:
    command = ["scp", "-F", "/dev/null", "-q", "-o", "StrictHostKeyChecking=accept-new"]
    if args.ssh_key:
        command += ["-i", str(args.ssh_key)]
    return command


def ssh(args, host: str, remote_command: str, *, check=True, capture=False) -> subprocess.CompletedProcess:
    command = ssh_base(args) + [ssh_target(args.ssh_user, host), remote_command]
    if capture:
        return subprocess.run(command, check=check, text=True, capture_output=True)
    return run(command, check=check)


def copy_repo(args, host: str, remote_dir: str) -> None:
    excludes = {"./.git", "./target", "./build", "./.pytest_cache"}
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        with tarfile.open(tmp.name, "w:gz", compresslevel=1) as archive:
            for path in sorted(Path.cwd().rglob("*")):
                rel = "./" + str(path.relative_to(Path.cwd()))
                if any(rel == ex or rel.startswith(ex + "/") for ex in excludes):
                    continue
                if path.is_file() and not path.is_symlink():
                    archive.add(path, arcname=str(path.relative_to(Path.cwd())), recursive=False)
        run(ssh_base(args) + [ssh_target(args.ssh_user, host), f"rm -rf {remote_dir} && mkdir -p {remote_dir}"])
        with open(tmp.name, "rb") as stream:
            proc = subprocess.run(
                ssh_base(args) + [ssh_target(args.ssh_user, host), f"tar -xzf - -C {remote_dir}"],
                stdin=stream,
                check=True,
            )
            assert proc.returncode == 0


def seed_build_artifacts(args, source_host: str, destination_host: str, remote_dir: str) -> None:
    if source_host == destination_host:
        return
    key_path = ensure_seed_key(args, source_host)
    remote_key = "/tmp/mracle-seed-key"
    run(scp_base(args) + [str(key_path), f"{ssh_target(args.ssh_user, destination_host)}:{remote_key}"])
    quoted_root = shlex.quote(remote_dir)
    quoted_source = shlex.quote(ssh_target(args.ssh_user, source_host))
    quoted_key = shlex.quote(remote_key)
    source_command = (
        f"test -d {quoted_root}/target && test -d {quoted_root}/build/unified && "
        f"tar -cf - -C {quoted_root} target build/unified"
    )
    destination_command = (
        f"chmod 600 {quoted_key} && mkdir -p {quoted_root} && "
        f"ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        f"-i {quoted_key} {quoted_source} {shlex.quote(source_command)} | "
        f"tar -xf - -C {quoted_root}; status=$?; rm -f {quoted_key}; exit $status"
    )
    try:
        ssh(args, destination_host, destination_command)
    finally:
        ssh(args, destination_host, f"rm -f {quoted_key}")


def ensure_seed_key(args, source_host: str) -> Path:
    key_path = Path(args.output) / "seed-copy-key"
    if not key_path.exists():
        run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path), "-C", "mracle-cloudlab-seed"])
    key_path.chmod(0o600)
    public_key = key_path.with_suffix(key_path.suffix + ".pub").read_text().strip()
    marker = f"mracle-seed-{hashlib.sha256(public_key.encode()).hexdigest()[:12]}"
    command = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
        f"(grep -F {shlex.quote(marker)} ~/.ssh/authorized_keys >/dev/null || "
        f"printf '%s %s\\n' {shlex.quote(public_key)} {shlex.quote(marker)} >> ~/.ssh/authorized_keys) && "
        "chmod 600 ~/.ssh/authorized_keys"
    )
    ssh(args, source_host, command)
    return key_path


def remote_run_exists(args, node: dict) -> str:
    host = node["hostname"]
    node_id = node["client_id"]
    remote = f"{args.remote_dir}/build/cloudlab/node-{node_id}"
    marker = shlex.quote(f"node-{node_id}")
    command = (
        f"test -f {remote}/completed.json && echo completed || "
        f"((ps -eo args= | grep 'python3 scripts/cloudlab_campaigns.py run-local' | grep -F -- {marker} | grep -v grep >/dev/null || "
        f"ps -eo args= | grep 'mracle-fuzz' | grep -F -- {marker} | grep -v grep >/dev/null) && echo running) || true"
    )
    result = ssh(args, host, command, capture=True)
    return result.stdout.strip() if result.stdout else ""


def setup_command() -> str:
    return r'''
set -euo pipefail
export PATH="/local/mracle/cmake-venv/bin:$HOME/.cargo/bin:$PATH"
if command -v apt-get >/dev/null 2>&1; then
  sudo dpkg --configure -a
  sudo env DEBIAN_FRONTEND=noninteractive apt-get -f install -y
  sudo env DEBIAN_FRONTEND=noninteractive apt-get update
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake ninja-build python3 python3-venv curl ca-certificates git pkg-config libomp-dev libatomic1 clang lld
fi
sudo mkdir -p /local/mracle
sudo chown -R "$(id -u):$(id -g)" /local/mracle
python3 - <<'PY'
import re
import shutil
import subprocess
import sys

cmake = shutil.which("cmake")
version = (0, 0, 0)
if cmake:
    output = subprocess.check_output([cmake, "--version"], text=True).splitlines()[0]
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    if match:
        version = tuple(map(int, match.groups()))
if version < (3, 24, 0):
    subprocess.check_call([sys.executable, "-m", "venv", "/local/mracle/cmake-venv"])
    subprocess.check_call([
        "/local/mracle/cmake-venv/bin/python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "cmake>=3.24,<4",
    ])
PY
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
'''.strip()


def start_remote_runs(args, state: dict, nodes: list[dict]) -> None:
    buckets = split_jobs(state["jobs"], len(nodes))
    remote_root = args.remote_dir
    existing = {entry["node"]["client_id"]: entry for entry in state.get("launched", {}).get("nodes", [])}
    launch = dict(existing)
    if args.seed_build_host:
        ensure_seed_key(args, args.seed_build_host)

    def launch_node(node: dict, jobs: list[dict]) -> dict:
        host = node["hostname"]
        node_id = node["client_id"]
        jobs_path = Path(args.output) / "assigned" / f"{node_id}.json"
        save(jobs_path, {"node": node, "jobs": jobs})
        status = remote_run_exists(args, node)
        if status:
            print(f"{node_id}: remote run already {status}; keeping existing launch", flush=True)
            return {"node": node, "jobs": jobs, "pid": status, "resumed": True}
        if args.setup_nodes:
            print(f"{node_id}: setting up packages", flush=True)
            ssh(args, host, setup_command())
        print(f"{node_id}: copying repository", flush=True)
        copy_repo(args, host, remote_root)
        if args.seed_build_host:
            print(f"{node_id}: seeding build artifacts from {args.seed_build_host}", flush=True)
            seed_build_artifacts(args, args.seed_build_host, host, remote_root)
        remote_jobs = f"{remote_root}/assigned-jobs.json"
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump({"node": node, "jobs": jobs}, tmp)
            tmp.write("\n")
            tmp_path = tmp.name
        try:
            run(scp_base(args) + [tmp_path, f"{ssh_target(args.ssh_user, host)}:{remote_jobs}"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        command = (
            f"export PATH=\"/local/mracle/cmake-venv/bin:$HOME/.cargo/bin:$PATH\" && "
            f"cd {remote_root} && mkdir -p build/cloudlab; "
            f"nohup python3 scripts/cloudlab_campaigns.py run-local "
            f"--jobs-json assigned-jobs.json --output build/cloudlab/node-{node_id} "
            f"--duration-secs {state['duration_secs']} --concurrency {state['concurrency']} "
            f"< /dev/null > build/cloudlab/node-{node_id}.log 2>&1 & "
            f"pid=$!; echo $pid; exit 0"
        )
        result = ssh(args, host, command, capture=True)
        print(f"{node_id}: launched pid {result.stdout.strip() if result.stdout else 'unknown'}", flush=True)
        return {"node": node, "jobs": jobs, "pid": result.stdout.strip() if result.stdout else ""}

    work = list(zip(nodes, buckets))
    if work:
        with ThreadPoolExecutor(max_workers=max(1, min(args.launch_workers, len(work)))) as pool:
            futures = {pool.submit(launch_node, node, jobs): node for node, jobs in work}
            for future in as_completed(futures):
                entry = future.result()
                launch[entry["node"]["client_id"]] = entry
                state["launched"] = {"time": stamp(), "nodes": [launch[key] for key in sorted(launch)]}
                save(args.state, state)
    state["launched"] = {"time": stamp(), "nodes": [launch[key] for key in sorted(launch)]}
    save(args.state, state)


def collect_from_node(args, node: dict, destination: Path) -> bool:
    host = node["hostname"]
    remote = f"{args.remote_dir}/build/cloudlab/node-{node['client_id']}"
    destination.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ssh_base(args) + [ssh_target(args.ssh_user, host), f"test -f {remote}/completed.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    completed = proc.returncode == 0
    with (destination / f"{node['client_id']}.tar.gz").open("wb") as stream:
        tar = subprocess.run(
            ssh_base(args) + [ssh_target(args.ssh_user, host), f"tar -czf - -C {remote} ."],
            stdout=stream,
            stderr=subprocess.DEVNULL,
        )
    if tar.returncode == 0:
        extract = destination / node["client_id"]
        if extract.exists():
            shutil.rmtree(extract)
        extract.mkdir()
        with tarfile.open(destination / f"{node['client_id']}.tar.gz") as archive:
            archive.extractall(extract)
    return completed


def collect(args) -> None:
    state = load(args.state)
    token = args.token or os.environ.get("CLOUDLAB_TOKEN")
    client = CloudLabClient(state["api_base"], token) if token else None
    experiment = state["experiment"]
    nodes = [entry["node"] for entry in state.get("launched", {}).get("nodes", [])]
    if not nodes and client:
        experiment = client.get_experiment(experiment["id"])
        nodes = aggregate_nodes(experiment) or manifest_nodes(client.manifests(experiment["id"]))
    all_done = bool(nodes)
    for node in nodes:
        done = collect_from_node(args, node, args.output / "nodes")
        all_done = all_done and done
    summarize(args.output)
    save(args.output / "collection.json", {"collected": stamp(), "complete": all_done, "nodes": nodes})
    if args.terminate and client and all_done:
        client.delete_experiment(experiment["id"])


def command_for_job(binary_dir: Path, output: Path, job: dict, duration_secs: int) -> list[str]:
    command = [
        str(binary_dir / "mracle-fuzz"),
        "--input",
        "corpus/seeds",
        "--output",
        str(output),
        "--model",
        "armv8",
        "--backend",
        job["backend"],
        "--seed",
        str(job["seed"]),
        "--duration-secs",
        str(duration_secs),
        "--iterations",
        str(2**64 - 1),
        "--timeout-ms",
        "180000",
        "--max-trials",
        "1024",
        "--max-corpus",
        "10000",
    ]
    if canonical_comparison(job["comparison"]) == "outcome":
        command += ["--reps", "100", "--replay-reps", "100", "--max-runs", "10000"]
        if job["sanitizer"] == "thread":
            command.append("--tsan-triage")
    else:
        command += [
            "--comparison",
            canonical_comparison(job["comparison"]),
            "--campaigns",
            "10",
            "--max-campaigns",
            "50",
            "--executions-per-campaign",
            "2000",
        ]
    return command


def build_binaries(root: Path, sanitizers: set[str], output: Path) -> None:
    env = dict(os.environ)
    env.setdefault("MRACLE_BUILD_JOBS", str(max(1, min(8, os.cpu_count() or 1))))
    env.setdefault("CC", shutil.which("clang-21") or shutil.which("clang") or "clang-21")
    env.setdefault("CXX", shutil.which("clang++-21") or shutil.which("clang++") or "clang++-21")
    log = output / "build.log"
    if "none" in sanitizers and not (root / "target/release/mracle-fuzz").exists():
        run(["cargo", "build", "--locked", "--release"], cwd=root, env={**env, "MRACLE_SANITIZER": "none"}, log=log)
    if "thread" in sanitizers and not (root / "target/tsan/release/mracle-fuzz").exists():
        run(
            ["cargo", "build", "--locked", "--release", "--target-dir", "target/tsan"],
            cwd=root,
            env={**env, "MRACLE_SANITIZER": "thread"},
            log=log,
        )


def run_one_job(root: Path, out: Path, job: dict, duration_secs: int) -> dict:
    job_out = out / "jobs" / job["id"]
    binary_dir = root / ("target/tsan/release" if job["sanitizer"] == "thread" else "target/release")
    env = dict(os.environ)
    if job["sanitizer"] == "thread":
        env.setdefault("TSAN_OPTIONS", "ignore_noninstrumented_modules=1")
        archer = Path("/usr/lib/llvm-21/lib/libarcher.so")
        if archer.exists():
            env.setdefault("OMP_TOOL_LIBRARIES", str(archer))
    else:
        for variable in ("TSAN_OPTIONS", "OMP_TOOL_LIBRARIES", "ARCHER_OPTIONS"):
            env.pop(variable, None)
    job_out.parent.mkdir(parents=True, exist_ok=True)
    log = out / "logs" / f"{job['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = stamp()
    code = 125
    try:
        result = run(command_for_job(binary_dir, job_out, job, duration_secs), cwd=root, env=env, log=log, check=False)
        code = result.returncode
        runs_dir = job_out / "runs"
        if runs_dir.exists():
            for trial in sorted(runs_dir.iterdir()):
                if trial.is_dir():
                    archive_trial(trial)
    finally:
        summary = load(job_out / "summary.json") if (job_out / "summary.json").exists() else None
        result = {"job": job, "started": started, "ended": stamp(), "exit_code": code, "summary": summary}
        save(job_out / "job-result.json", result)
    return result


def run_local(args) -> None:
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.jobs_json)
    jobs = data["jobs"]
    save(out / "plan.json", {"started": stamp(), "jobs": jobs, "duration_secs": args.duration_secs, "concurrency": args.concurrency})
    build_binaries(root, {job["sanitizer"] for job in jobs}, out)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one_job, root, out, job, args.duration_secs) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save(out / "status.json", {"updated": stamp(), "finished": len(results), "total": len(jobs), "results": results})
    save(out / "completed.json", {"completed": stamp(), "results": results})
    summarize(out)


def read_events(job_dir: Path) -> list[dict]:
    path = job_dir / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def summarize(root: Path) -> dict:
    rows = []
    for job_result in sorted(root.rglob("job-result.json")):
        result = load(job_result)
        job_dir = job_result.parent
        summary = result.get("summary") or {}
        events = read_events(job_dir)
        counts = Counter(event.get("category", "unknown") for event in events)
        label = job_label(result["job"])
        rows.append(
            {
                "id": result["job"]["id"],
                "label": label,
                "job": result["job"],
                "exit_code": result.get("exit_code"),
                "attempts": summary.get("attempts", len(events)),
                "usable_trials": summary.get("usable_trials", counts.get("ok", 0)),
                "findings": summary.get("findings", 0),
                "compiler_edges": summary.get("compiler_edges", max([event.get("compiler_edges", 0) for event in events] or [0])),
                "corpus_size": summary.get("corpus_size", max([event.get("corpus_size", 0) for event in events] or [0])),
                "elapsed_seconds": summary.get("elapsed_seconds", max([event.get("elapsed_seconds", 0) for event in events] or [0])),
                "stop_reason": summary.get("stop_reason", "missing-summary"),
                "categories": dict(counts),
            }
        )
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    aggregate = {}
    for label, group in by_label.items():
        aggregate[label] = {
            "runs": len(group),
            "attempts_mean": mean(row["attempts"] for row in group),
            "usable_mean": mean(row["usable_trials"] for row in group),
            "findings_total": sum(row["findings"] for row in group),
            "compiler_edges_mean": mean(row["compiler_edges"] for row in group),
            "corpus_size_mean": mean(row["corpus_size"] for row in group),
            "attempts_per_hour_mean": mean(rate(row["attempts"], row["elapsed_seconds"]) for row in group),
            "categories": dict(sum_counters(Counter(row["categories"]) for row in group)),
        }
    report = {"updated": stamp(), "jobs": rows, "aggregate": aggregate}
    save(root / "summary.json", report)
    write_report(root, report)
    write_figures(root, report)
    return report


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def rate(count: float, seconds: float) -> float:
    return count * 3600 / seconds if seconds else 0.0


def sum_counters(counters) -> Counter:
    total = Counter()
    for counter in counters:
        total.update(counter)
    return total


def svg_bar_chart(path: Path, title: str, values: list[tuple[str, float]], ylabel: str) -> None:
    width = max(900, 90 * max(1, len(values)))
    height = 460
    left, right, top, bottom = 90, 30, 60, 135
    plot_w = width - left - right
    plot_h = height - top - bottom
    maximum = max([value for _, value in values] or [1.0]) or 1.0
    bar_w = plot_w / max(1, len(values)) * 0.68
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{xml_escape(title)}</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="13">{xml_escape(ylabel)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(values):
        x = left + (index + 0.16) * plot_w / max(1, len(values))
        h = value / maximum * plot_h
        y = top + plot_h - h
        color = "#3768a6" if index % 2 == 0 else "#2f8f70"
        pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        pieces.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1f}</text>')
        pieces.append(f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 16}" transform="rotate(35 {x + bar_w/2:.1f} {top + plot_h + 16})" font-family="sans-serif" font-size="10">{xml_escape(label)}</text>')
    pieces.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces) + "\n")


def xml_escape(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_figures(root: Path, report: dict) -> None:
    aggregate = report["aggregate"]
    items = sorted(aggregate.items())
    svg_bar_chart(root / "figures" / "coverage.svg", "Compiler Coverage By Configuration", [(k, v["compiler_edges_mean"]) for k, v in items], "Mean compiler edges")
    svg_bar_chart(root / "figures" / "throughput.svg", "Throughput By Configuration", [(k, v["attempts_per_hour_mean"]) for k, v in items], "Mean attempts/hour")
    svg_bar_chart(root / "figures" / "findings.svg", "Findings By Configuration", [(k, v["findings_total"]) for k, v in items], "Total findings")
    svg_bar_chart(root / "figures" / "corpus.svg", "Corpus Growth By Configuration", [(k, v["corpus_size_mean"]) for k, v in items], "Mean final corpus size")


def write_report(root: Path, report: dict) -> None:
    lines = [
        "# MLIR-MRacle CloudLab Evaluation",
        "",
        "| Configuration | Runs | Attempts/h | Compiler edges | Corpus size | Findings |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, data in sorted(report["aggregate"].items()):
        lines.append(
            f"| {label} | {data['runs']} | {data['attempts_per_hour_mean']:.1f} | "
            f"{data['compiler_edges_mean']:.1f} | {data['corpus_size_mean']:.1f} | {data['findings_total']} |"
        )
    lines += [
        "",
        "Figures:",
        "- `figures/coverage.svg`",
        "- `figures/throughput.svg`",
        "- `figures/findings.svg`",
        "- `figures/corpus.svg`",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_cloudlab(args) -> None:
    token = args.token or os.environ.get("CLOUDLAB_TOKEN")
    client = CloudLabClient(args.api_base, token)
    if args.state.exists():
        state = load(args.state)
        print(f"resuming CloudLab experiment {state['experiment']['id']}")
    else:
        state = submit(args)
    if "ready" not in state:
        experiment = wait_ready(client, state["experiment"]["id"], args.ready_timeout_secs, args.poll_secs)
        nodes = aggregate_nodes(experiment) or manifest_nodes(client.manifests(experiment["id"]))
        if len(nodes) < state["node_count"]:
            raise RuntimeError(f"expected {state['node_count']} nodes, saw {len(nodes)}")
        state["ready"] = {"time": stamp(), "experiment": experiment, "nodes": nodes}
        save(args.state, state)
    maybe_extend_experiment(args, client, state)
    nodes = state["ready"]["nodes"]
    start_remote_runs(args, state, nodes[: state["node_count"]])
    deadline = time.monotonic() + args.duration_secs + args.collect_slack_secs
    while time.monotonic() < deadline:
        collect(args)
        status = load(args.output / "collection.json")
        if status["complete"]:
            break
        time.sleep(args.poll_secs)
    collect(args)


def terminate(args) -> None:
    state = load(args.state)
    token = args.token or os.environ.get("CLOUDLAB_TOKEN")
    if not token:
        raise SystemExit("set CLOUDLAB_TOKEN or pass --token")
    CloudLabClient(state["api_base"], token).delete_experiment(state["experiment"]["id"])


def extend_lifetime(args) -> None:
    if args.extend_days <= 0:
        raise SystemExit("pass --extend-days N with N greater than zero")
    state = load(args.state)
    token = args.token or os.environ.get("CLOUDLAB_TOKEN")
    if not token:
        raise SystemExit("set CLOUDLAB_TOKEN or pass --token")
    client = CloudLabClient(state["api_base"], token)
    maybe_extend_experiment(args, client, state)


def add_common(parser) -> None:
    ssh_key = os.environ.get("CLOUDLAB_SSH_KEY")
    sshpubkey = os.environ.get("CLOUDLAB_SSHPUBKEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--state", type=Path, default=Path("build/cloudlab/eval/state.json"))
    parser.add_argument("--output", type=Path, default=Path("build/cloudlab/eval"))
    parser.add_argument("--api-base", default=os.environ.get("CLOUDLAB_API_BASE", API_BASE))
    parser.add_argument("--token")
    parser.add_argument("--project", default=os.environ.get("CLOUDLAB_PROJECT", ""))
    parser.add_argument("--group", default=os.environ.get("CLOUDLAB_GROUP", ""))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--profile-project", default=os.environ.get("CLOUDLAB_PROFILE_PROJECT", DEFAULT_PROFILE_PROJECT))
    parser.add_argument("--name", default=default_experiment_name())
    parser.add_argument("--node-type", default="m400")
    parser.add_argument("--os-image", default=os.environ.get("CLOUDLAB_OS_IMAGE", "default"))
    parser.add_argument("--max-nodes", type=int, default=6)
    parser.add_argument("--max-runs", type=int, default=DEFAULT_MAX_RUNS)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument(
        "--matrix",
        type=parse_matrix,
        help=(
            "comma-separated comparison/backend/sanitizer entries; "
            "for example outcome/jit/none,outcome/native/none"
        ),
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--duration-secs", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--experiment-hours", type=int, default=DEFAULT_EXPERIMENT_HOURS)
    parser.add_argument("--extend-days", type=float, default=0.0, help="after the experiment is ready, request this many extra days via the Portal API")
    parser.add_argument("--extend-reason", default="MLIR-MRacle evaluation data collection")
    parser.add_argument("--ready-timeout-secs", type=int, default=45 * 60)
    parser.add_argument("--collect-slack-secs", type=int, default=3 * 60 * 60)
    parser.add_argument("--poll-secs", type=int, default=300)
    parser.add_argument("--ssh-user", default=os.environ.get("CLOUDLAB_USER", os.environ.get("USER", "")))
    parser.add_argument("--ssh-key", type=Path, default=Path(ssh_key).expanduser() if ssh_key else None)
    parser.add_argument("--sshpubkey", type=Path, default=Path(sshpubkey).expanduser() if sshpubkey else None)
    parser.add_argument("--remote-dir", default="/local/mracle/MLIR-MRacle-Tool")
    parser.add_argument("--seed-build-host", default=os.environ.get("CLOUDLAB_SEED_BUILD_HOST", ""), help="optional existing node to copy target/ and build/unified/ from before launching")
    parser.add_argument("--launch-workers", type=int, default=DEFAULT_LAUNCH_WORKERS, help="number of CloudLab nodes to prepare and launch concurrently")
    parser.add_argument("--setup-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminate", action=argparse.BooleanOptionalAction, default=False)


def parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("run", "submit, wait, launch node jobs, collect, and make figures"),
        ("submit", "instantiate the CloudLab experiment"),
        ("collect", "pull node data from a submitted run and regenerate figures"),
        ("extend", "request extra lifetime for the experiment in state.json"),
        ("terminate", "terminate the experiment from state.json"),
    ]:
        child = sub.add_parser(name, help=help_text)
        add_common(child)
    local = sub.add_parser("run-local", help="run assigned jobs on one CloudLab node")
    local.add_argument("--root", type=Path, default=Path.cwd())
    local.add_argument("--jobs-json", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--duration-secs", type=int, default=DEFAULT_DURATION)
    local.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    figures = sub.add_parser("figures", help="regenerate REPORT.md and SVG figures from collected data")
    figures.add_argument("--output", type=Path, default=Path("build/cloudlab/eval"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    env_file, explicit_env = env_file_from_argv(argv)
    load_env_file(env_file.expanduser(), explicit=explicit_env)
    args = parse_args(argv)
    if getattr(args, "max_runs", DEFAULT_MAX_RUNS) > 25:
        raise SystemExit("--max-runs must stay at or below 25 for the reviewer-sized evaluation")
    if args.command == "run":
        run_cloudlab(args)
    elif args.command == "submit":
        submit(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "extend":
        extend_lifetime(args)
    elif args.command == "terminate":
        terminate(args)
    elif args.command == "run-local":
        run_local(args)
    elif args.command == "figures":
        summarize(args.output)


if __name__ == "__main__":
    main()
