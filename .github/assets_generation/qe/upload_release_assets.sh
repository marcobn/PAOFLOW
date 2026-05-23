#!/bin/bash

set -euo pipefail

TAG="${1:-integration-assets-v1}"

ASSET_DIR="/home/anooja/Work/software/PAOFLOW/.github/assets_generation/qe/_assets"

QE_ASSET="${ASSET_DIR}/qe_assets.tar.gz"
PAOFLOW_ASSET="${ASSET_DIR}/paoflow_assets.tar.gz"

REPO="marcobn/PAOFLOW"

cd "${ASSET_DIR}"

if [[ ! -f "${QE_ASSET}" ]]; then
    echo "Missing asset: ${QE_ASSET}"
    exit 1
fi

if [[ ! -f "${PAOFLOW_ASSET}" ]]; then
    echo "Missing asset: ${PAOFLOW_ASSET}"
    exit 1
fi

echo "Generating SHA256SUMS..."
sha256sum qe_assets.tar.gz paoflow_assets.tar.gz > SHA256SUMS

echo "Creating GitHub release: ${TAG}"

gh release create "${TAG}" \
    qe_assets.tar.gz \
    paoflow_assets.tar.gz \
    SHA256SUMS \
    --repo "${REPO}" \
    --title "${TAG}" \
    --notes "Integration test assets"

echo "Release created successfully."
