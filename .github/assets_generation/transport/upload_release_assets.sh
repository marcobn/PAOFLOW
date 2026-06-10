#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: [environment variables] upload_release_assets.sh [TAG]

Upload transport release assets from the local _assets directory.

Positional arguments:
  TAG                   Existing release tag to upload to.
                        Default: integration-assets-v1

Environment variables:
  ASSET_DIR             Directory containing transport_test_assets.tar.gz.
                        Default: .github/assets_generation/transport/_assets
  REPO                  GitHub repository in OWNER/REPO form.
                        Default: marcobn/PAOFLOW

Behavior:
  - Regenerates transport_SHA256SUMS from transport_test_assets.tar.gz before upload.
  - Refuses to create a missing release.
  - Replaces existing assets with the same name.
EOF
}

TAG="integration-assets-v1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TAG="$1"
      shift
      if [[ $# -gt 0 ]]; then
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ASSET_DIR="${ASSET_DIR:-${REPO_ROOT}/.github/assets_generation/transport/_assets}"
TRANSPORT_ASSET="${ASSET_DIR}/transport_test_assets.tar.gz"
CHECKSUM_ASSET="transport_SHA256SUMS"

REPO="${REPO:-marcobn/PAOFLOW}"

cd "${ASSET_DIR}"

if [[ ! -f "${TRANSPORT_ASSET}" ]]; then
  echo "Missing asset: ${TRANSPORT_ASSET}"
  exit 1
fi

if ! gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "Release '${TAG}' does not exist. Refusing to create a new release."
  exit 1
fi

echo "Generating transport SHA256SUMS..."
sha256sum transport_test_assets.tar.gz > "${CHECKSUM_ASSET}"

echo "Uploading transport assets to existing release: ${TAG}"
gh release upload "${TAG}" \
  transport_test_assets.tar.gz \
  "${CHECKSUM_ASSET}" \
  --repo "${REPO}" \
  --clobber

echo "Transport assets uploaded successfully."
