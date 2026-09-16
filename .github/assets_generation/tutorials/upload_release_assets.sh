#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: [environment variables] upload_release_assets.sh [TAG]

Upload tutorial assets to a dedicated versioned GitHub release.

Positional arguments:
  TAG                   Tutorial asset release tag.
                        Default: tutorial-assets-v1

Environment variables:
  ASSET_DIR             Directory containing tutorial_assets.tar.gz.
                        Default: .github/assets_generation/tutorials/
  REPO                  GitHub repository in OWNER/REPO form.
                        Default: marcobn/PAOFLOW

Behavior:
  - Regenerates tutorial_SHA256SUMS before upload.
  - Creates the tutorial asset release when it does not exist.
  - Refuses to replace an existing archive or checksum. Publish a new
    tutorial-assets-vN release when the payload changes.
EOF
}

TAG="tutorial-assets-v1"

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

if [[ ! "${TAG}" =~ ^tutorial-assets-v[1-9][0-9]*$ ]]; then
  echo "Invalid tutorial asset tag: ${TAG}" >&2
  echo "Expected tutorial-assets-vN, where N is a positive integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ASSET_DIR="${ASSET_DIR:-${REPO_ROOT}/.github/assets_generation/tutorials/}"
TUTORIAL_ASSET="${ASSET_DIR}/tutorial_assets.tar.gz"
CHECKSUM_ASSET="tutorial_SHA256SUMS"

REPO="${REPO:-marcobn/PAOFLOW}"

cd "${ASSET_DIR}"

if [[ ! -f "${TUTORIAL_ASSET}" ]]; then
  echo "Missing asset: ${TUTORIAL_ASSET}"
  exit 1
fi

echo "Generating tutorial SHA256SUMS..."
sha256sum tutorial_assets.tar.gz > "${CHECKSUM_ASSET}"

if ! gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "Creating tutorial asset release: ${TAG}"
  gh release create "${TAG}" \
    --repo "${REPO}" \
    --title "${TAG}" \
    --notes "Versioned assets for PAOFLOW tutorials"
fi

existing_assets="$(gh release view "${TAG}" --repo "${REPO}" --json assets --jq '.assets[].name')"
if grep -Fxq 'tutorial_assets.tar.gz' <<< "${existing_assets}" || \
  grep -Fxq "${CHECKSUM_ASSET}" <<< "${existing_assets}"; then
  echo "Release '${TAG}' already contains tutorial assets." >&2
  echo "Publish a new tutorial-assets-vN release instead of replacing them." >&2
  exit 1
fi

echo "Uploading tutorial assets to release: ${TAG}"
gh release upload "${TAG}" \
  tutorial_assets.tar.gz \
  "${CHECKSUM_ASSET}" \
  --repo "${REPO}"

echo "Tutorial assets uploaded successfully."
