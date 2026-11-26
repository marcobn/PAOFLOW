#!/usr/bin/env bash
# Upload case data archives (<case>-vN.tar.gz, from build_case_data.py) to the release case-data.
# Creates the release when it does not exist and replaces an archive of the same name: until the
# first PAOFLOW release that ships src/PAOFLOW/gen/case_data.json, a rebuilt archive keeps its name
# (update its checksum there in the same commit); after it, build a new version instead.
#
#   upload_case_data.sh si_bands_transport-v1.tar.gz [more archives...]
#
# REPO (default marcobn/PAOFLOW) selects the repository.

set -euo pipefail

TAG="case-data"
REPO="${REPO:-marcobn/PAOFLOW}"

if [[ $# -eq 0 ]]; then
  sed -n '2,9p' "$0"
  exit 1
fi

for archive in "$@"; do
  if [[ ! "$(basename "${archive}")" =~ ^[a-z0-9_]+-v[1-9][0-9]*\.tar\.gz$ ]]; then
    echo "Not a case data archive (<case>-vN.tar.gz): ${archive}" >&2
    exit 1
  fi
  [[ -f "${archive}" ]] || { echo "Missing: ${archive}" >&2; exit 1; }
done

if ! gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  gh release create "${TAG}" --repo "${REPO}" --title "${TAG}" \
    --notes "Data of the example cases (examples/cases), one archive per case, pinned in src/PAOFLOW/gen/case_data.json."
fi

gh release upload "${TAG}" "$@" --repo "${REPO}" --clobber
