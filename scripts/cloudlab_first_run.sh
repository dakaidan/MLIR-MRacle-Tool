#!/usr/bin/env bash
# Compatibility wrapper for the API-based CloudLab evaluation controller.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
exec python3 scripts/cloudlab_campaigns.py run "$@"
