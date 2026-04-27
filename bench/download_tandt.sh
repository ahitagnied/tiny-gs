#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(cd "$SCRIPT_DIR/.." && pwd)/data"
TMP="$DEST/.tmp"

mkdir -p "$DEST" "$TMP"

curl -L --fail --retry 5 --retry-delay 5 \
    -o "$TMP/tandt_db.zip" \
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"

unzip -q -o "$TMP/tandt_db.zip" -d "$DEST"

rm -rf "$TMP"

echo "tandt -> $DEST/tandt"
echo "db    -> $DEST/db"
du -sh "$DEST/tandt"/* "$DEST/db"/*
