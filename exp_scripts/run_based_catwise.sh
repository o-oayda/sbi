#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHONUNBUFFERED=1 \
python "$REPO_ROOT/dipolesbi/scripts/based_catwise.py" "$@"
