#!/usr/bin/env bash

set -euo pipefail

TAG="${1:-integration-assets-v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ASSET_DIR="${ASSET_DIR:-${REPO_ROOT}/.github/assets_generation/qe/_assets}"

QE_TEST_ASSET="${ASSET_DIR}/qe_test_assets.tar.gz"
CHECKSUM_ASSET="qe_SHA256SUMS"

REPO="${REPO:-marcobn/PAOFLOW}"

cd "${ASSET_DIR}"

if [[ ! -f "${QE_TEST_ASSET}" ]]; then
    echo "Missing asset: ${QE_TEST_ASSET}"
    exit 1
fi

echo "Generating QE SHA256SUMS..."
sha256sum qe_test_assets.tar.gz > "${CHECKSUM_ASSET}"

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
    qe_test_assets.tar.gz \
    "${CHECKSUM_ASSET}" \
    --repo "${REPO}" \
    --clobber

echo "Assets uploaded successfully."
