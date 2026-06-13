#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

sphinx-autobuild \
  docs \
  docs/_build/html \
  --host 127.0.0.1 \
  --port 8000 \
  --ignore docs/_build \
  --ignore docs/api/autoapi
