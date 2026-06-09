#!/usr/bin/env bash

set -euo pipefail

TAG="${1:-integration-assets-v1}"

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
