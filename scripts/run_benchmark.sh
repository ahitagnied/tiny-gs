#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

run() {
    local ds="$1" scene="$2" res="$3"
    local src="data/$ds/$scene"
    local out="output/$ds/$scene"
    local final="$out/point_cloud/iteration_30000/point_cloud.ply"

    if [[ ! -d "$src" ]]; then
        echo "[miss] $ds/$scene (no $src)"; return
    fi
    if [[ -f "$final" ]]; then
        echo "[skip] $ds/$scene (already trained)"; return
    fi

    echo "[run]  $ds/$scene -r $res"
    mkdir -p "$out"
    stdbuf -oL -eL python -u main.py "$src" --model_path "$out" --resolution "$res" \
        2>&1 | tee "$out/train.log"
}

for s in bicycle garden stump flowers treehill; do run mipnerf360 "$s" 4; done

for s in bonsai counter kitchen room; do run mipnerf360 "$s" 2; done

for s in train truck;        do run tandt "$s" 1; done
for s in drjohnson playroom; do run db    "$s" 1; done

python scripts/aggregate_metrics.py

echo "all done. summary -> output/summary.md"
