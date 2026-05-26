#!/usr/bin/env bash

set -euo pipefail

TAG="${1:-transport-integration-assets-v1}"

ASSET_DIR="/home/anooja/Work/software/PAOFLOW/.github/assets_generation/transport/_assets"
TRANSPORT_ASSET="${ASSET_DIR}/transport_test_assets.tar.gz"

REPO="marcobn/PAOFLOW"

cd "${ASSET_DIR}"

if [[ ! -f "${TRANSPORT_ASSET}" ]]; then
  echo "Missing asset: ${TRANSPORT_ASSET}"
  exit 1
fi

echo "Generating SHA256SUMS..."
sha256sum transport_test_assets.tar.gz > SHA256SUMS

echo "Creating GitHub release: ${TAG}"

gh release create "${TAG}" \
  transport_test_assets.tar.gz \
  SHA256SUMS \
  --repo "${REPO}" \
  --title "${TAG}" \
  --notes "Transport integration test assets"

echo "Release created successfully."
