#!/usr/bin/env python3
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import TrainConfig, train  # noqa: E402

ITERS = TrainConfig.iterations

# (dataset, scene, image downsample factor)
SCENES: list[tuple[str, str, int]] = [
    *(("mipnerf360", s, 4) for s in ("bicycle", "garden", "stump", "flowers", "treehill")),
    *(("mipnerf360", s, 2) for s in ("bonsai", "counter", "kitchen", "room")),
    *(("tandt", s, 1) for s in ("train", "truck")),
    *(("db",    s, 1) for s in ("drjohnson", "playroom")),
]

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_metrics(out: Path) -> dict | None:
    f = out / f"test/iteration_{ITERS}/metrics.json"
    return json.loads(f.read_text()) if f.is_file() else None


def run_one(ds: str, scene: str, res: int) -> dict:
    src = ROOT / "data"   / ds / scene
    out = ROOT / "output" / ds / scene
    final_ply = out / f"point_cloud/iteration_{ITERS}/point_cloud.ply"

    if not src.is_dir():
        return {"status": "missing_data"}

    train_s: float | None = None
    if not final_ply.is_file():
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        train(TrainConfig(source_path=str(src), model_path=str(out), resolution=res))
        train_s = round(time.time() - t0, 1)

    metrics = load_metrics(out) or {"status": "no_renders"}
    if train_s is not None:
        metrics["train_s"] = train_s
    return metrics


def print_table(results: dict) -> None:
    print(f"\n{'scene':<28} {'PSNR':>7} {'SSIM':>7} {'N':>4} {'train_s':>9}")
    print("-" * 60)
    for key, m in results.items():
        psnr = f"{m['psnr']:>7.2f}" if "psnr" in m else f"{m.get('status','?'):>7}"
        ssim_ = f"{m['ssim']:>7.4f}" if "ssim" in m else f"{'-':>7}"
        n = f"{m.get('n_test','-'):>4}"
        ts = f"{m.get('train_s','-'):>9}"
        print(f"{key:<28} {psnr} {ssim_} {n} {ts}")


if __name__ == "__main__":
    results: dict[str, dict] = {}
    for ds, scene, res in SCENES:
        key = f"{ds}/{scene}"
        print(f"\n=== {key} (r={res}) ===")
        results[key] = run_one(ds, scene, res)

    out_path = ROOT / "bench/results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"\nresults -> {out_path}")
