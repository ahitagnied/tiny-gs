#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(cd "$SCRIPT_DIR/.." && pwd)/data/mipnerf360"
TMP="$DEST/.tmp"

mkdir -p "$DEST" "$TMP"

curl -L --fail --retry 5 --retry-delay 5 \
    -o "$TMP/360_v2.zip" \
    "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

curl -L --fail --retry 5 --retry-delay 5 \
    -o "$TMP/360_extra_scenes.zip" \
    "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"

unzip -q -o "$TMP/360_v2.zip"           -d "$DEST"
unzip -q -o "$TMP/360_extra_scenes.zip" -d "$DEST"

rm -rf "$TMP"

echo "mipnerf360 -> $DEST"
du -sh "$DEST"/*
