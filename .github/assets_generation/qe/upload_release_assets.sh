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

if ! gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
    echo "Creating release ${TAG}..."

    gh release create "${TAG}" \
        --repo "${REPO}" \
        --title "${TAG}" \
        --notes "Integration test assets"
else
    echo "Release ${TAG} already exists."
fi

echo "Uploading/replacing assets..."

gh release upload "${TAG}" \
    qe_assets.tar.gz \
    paoflow_assets.tar.gz \
    SHA256SUMS \
    --repo "${REPO}" \
    --clobber

echo "Assets uploaded successfully."
