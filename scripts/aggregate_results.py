#!/usr/bin/env python3
"""Aggregate MACRO evaluation results across scenes into a summary table.

Reads the per-scene ``aggregate_results_progressive.json`` files written by
``src/evaluate.py`` (one per scene, under --results-dir) and reports the mean
over scenes for each config, per metric — the same mean-of-scene-means used
for the paper tables.

Usage:
    python scripts/aggregate_results.py --results-dir /path/to/results_out
    python scripts/aggregate_results.py --results-dir /path/to/results_out --configs macro difix 3dgs
"""
import argparse
import glob
import json
import os


METRICS = ["avg_psnr", "avg_ssim", "avg_lpips", "avg_dreamsim", "avg_dinov2"]
PRETTY = {"avg_psnr": "PSNR", "avg_ssim": "SSIM", "avg_lpips": "LPIPS",
          "avg_dreamsim": "DreamSim", "avg_dinov2": "DINOv2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing per-scene subfolders with "
                         "aggregate_results_progressive.json")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="Configs to report (default: all found).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "**",
                                          "aggregate_results_progressive.json"),
                             recursive=True))
    if not files:
        raise SystemExit(f"No aggregate_results_progressive.json under {args.results_dir}")

    # config -> list of per-scene metric dicts
    by_config = {}
    for f in files:
        data = json.load(open(f))
        for cfg, entries in data.get("per_config", {}).items():
            if args.configs and cfg not in args.configs:
                continue
            for e in entries:
                by_config.setdefault(cfg, []).append(e)

    hdr = f"{'config':16s} {'#scenes':>7s} " + " ".join(f"{PRETTY[m]:>9s}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for cfg in sorted(by_config):
        rows = by_config[cfg]
        means = []
        for m in METRICS:
            vals = [r[m] for r in rows if r.get(m) is not None]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        print(f"{cfg:16s} {len(rows):7d} " + " ".join(f"{v:9.3f}" for v in means))


if __name__ == "__main__":
    main()
