#!/usr/bin/env bash
# Print the latest local CloudLab evaluation report, or collect first when a
# state file and CLOUDLAB_TOKEN are available.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
output=${1:-build/cloudlab/eval}
state=${2:-$output/state.json}
if [ -f "$state" ]; then
  python3 scripts/cloudlab_campaigns.py collect --state "$state" --output "$output" || true
fi
if [ -f "$output/REPORT.md" ]; then
  cat "$output/REPORT.md"
elif [ -f "$output/collection.json" ]; then
  cat "$output/collection.json"
else
  echo "No CloudLab evaluation report found under $output"
fi
