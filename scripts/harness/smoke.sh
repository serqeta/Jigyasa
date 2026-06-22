#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root_dir"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Running smoke tests..."
python -m pytest -q tests/unit/test_smoke.py || exit 1
echo "Smoke test complete."
