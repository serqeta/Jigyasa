#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root_dir"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Running mypy..."
python -m mypy voiceshield --ignore-missing-imports || exit 1
echo "Typecheck complete."
