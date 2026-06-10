#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: [environment variables] upload_release_assets.sh [TAG]

Upload QE test release assets from the local _assets directory.

Positional arguments:
  TAG                   Release tag to upload to.
                        Default: integration-assets-v1

Environment variables:
  ASSET_DIR             Directory containing qe_test_assets.tar.gz.
                        Default: .github/assets_generation/qe/_assets
  REPO                  GitHub repository in OWNER/REPO form.
                        Default: marcobn/PAOFLOW

Behavior:
  - Regenerates qe_SHA256SUMS from qe_test_assets.tar.gz before upload.
  - Creates the release if TAG does not already exist.
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
