#!/usr/bin/env bash

set -euo pipefail

TAG="${1:-transport-integration-assets-v1}"

ASSET_DIR="/home/anooja/Work/software/PAOFLOW/.github/assets_generation/transport/_assets"
TRANSPORT_ASSET="${ASSET_DIR}/transport_test_assets.tar.gz"
CHECKSUM_ASSET="transport_SHA256SUMS"

REPO="marcobn/PAOFLOW"

cd "${ASSET_DIR}"

if [[ ! -f "${TRANSPORT_ASSET}" ]]; then
  echo "Missing asset: ${TRANSPORT_ASSET}"
  exit 1
fi

echo "Generating SHA256SUMS..."
sha256sum transport_test_assets.tar.gz > "${CHECKSUM_ASSET}"

if gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "Release '${TAG}' exists. Uploading transport assets..."
  gh release upload "${TAG}" \
    transport_test_assets.tar.gz \
    "${CHECKSUM_ASSET}" \
    --repo "${REPO}" \
    --clobber
else
  echo "Creating GitHub release: ${TAG}"
  gh release create "${TAG}" \
    transport_test_assets.tar.gz \
    "${CHECKSUM_ASSET}" \
    --repo "${REPO}" \
    --title "${TAG}" \
    --notes "Transport integration test assets"
fi

echo "Release created successfully."
