#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Usage:
#   ./docs/serve_docs.sh          fast live preview: incremental, no source pages
#   ./docs/serve_docs.sh --full   production-like: clean rebuild, viewcode [source] links
#
# Extra arguments are passed through to sphinx-autobuild.

FULL=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    *) ARGS+=("$arg") ;;
  esac
done

if (( FULL )); then
  BUILD_DIR=docs/_build/html
  SPHINX_OPTS=(-E -a)
else
  # sphinx.ext.viewcode regenerates all _modules/*.html pages on every build,
  # which costs ~25 s even when nothing changed. conf.py drops it in fast mode.
  export PAOFLOW_DOCS_FAST=1
  # Separate output dir so the two modes don't invalidate each other's cache.
  BUILD_DIR=docs/_build/fast
  SPHINX_OPTS=()
fi

sphinx-autobuild \
  docs \
  "$BUILD_DIR" \
  "${SPHINX_OPTS[@]+"${SPHINX_OPTS[@]}"}" \
  --host 127.0.0.1 \
  --port 8000 \
  --ignore docs/_build \
  --ignore docs/api/autoapi \
  "${ARGS[@]+"${ARGS[@]}"}"
