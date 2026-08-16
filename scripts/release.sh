#!/usr/bin/env bash
# Cut a ComfyUI-Stimma release: bumps version.py + pyproject.toml, rebuilds the
# manager UI, commits, and tags vX.Y.Z. Push the tag to publish (GitHub Actions
# creates the release; the in-app updater follows tags).
#
#   scripts/release.sh 1.2.0
set -euo pipefail
cd "$(dirname "$0")/.."
VER="${1:?usage: scripts/release.sh X.Y.Z}"
[[ "$VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "bad version: $VER"; exit 1; }
sed -i.bak "s/^PRODUCT_VERSION = \".*\"/PRODUCT_VERSION = \"$VER\"/" stp_server/version.py && rm stp_server/version.py.bak
sed -i.bak "0,/^version = \".*\"/s//version = \"$VER\"/" pyproject.toml && rm pyproject.toml.bak
( cd manage-ui && npm ci --silent && npm run build --silent )
git add stp_server/version.py pyproject.toml stp_server/manage/ui
git commit -m "Release v$VER"
git tag -a "v$VER" -m "ComfyUI-Stimma v$VER"
echo "Tagged v$VER. Push with: git push && git push origin v$VER"
