#!/usr/bin/env python3
"""Verify that a downloaded scene folder has everything MACRO needs.

A valid scene folder contains:
    <scene>/
      images/           training wide-shot photos
      closeup_gt/       ground-truth close-up photos (for scoring)
      sparse/0/         COLMAP model (cameras.bin, images.bin, points3D.bin, ...)
      split.json        train/closeup split + poses + intrinsics

Usage:
    python verify_scene.py /path/to/scene
    python verify_scene.py /path/to/dataset_dir --all   # check every subfolder
"""
import argparse
import json
import os
import sys


def check_scene(scene):
    problems = []
    if not os.path.isdir(os.path.join(scene, "images")) or not os.listdir(os.path.join(scene, "images")):
        problems.append("missing/empty images/")
    if not os.path.isdir(os.path.join(scene, "closeup_gt")):
        problems.append("missing closeup_gt/")
    s0 = os.path.join(scene, "sparse", "0")
    if not os.path.isdir(s0):
        problems.append("missing sparse/0/")
    else:
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            if not os.path.exists(os.path.join(s0, b)):
                problems.append(f"missing sparse/0/{b}")
    sj = os.path.join(scene, "split.json")
    if not os.path.exists(sj):
        problems.append("missing split.json")
    else:
        try:
            d = json.load(open(sj))
        except Exception as e:
            problems.append(f"split.json not valid JSON: {e}")
            d = {}
        # DS1 uses 'training_frames', DS3 uses 'train_frames' — accept either.
        if not (d.get("training_frames") or d.get("train_frames")):
            problems.append("split.json has no training_frames/train_frames")
        for k in ("closeup_eval_pairs", "closeup_poses", "intrinsics"):
            if k not in d:
                problems.append(f"split.json missing '{k}'")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Scene folder, or a dataset dir with --all")
    ap.add_argument("--all", action="store_true",
                    help="Treat path as a parent dir and check every subfolder.")
    args = ap.parse_args()

    scenes = (
        [os.path.join(args.path, d) for d in sorted(os.listdir(args.path))
         if os.path.isdir(os.path.join(args.path, d))]
        if args.all else [args.path]
    )
    n_ok = 0
    for sc in scenes:
        probs = check_scene(sc)
        if probs:
            print(f"[FAIL] {sc}")
            for p in probs:
                print(f"        - {p}")
        else:
            print(f"[ OK ] {sc}")
            n_ok += 1
    print(f"\n{n_ok}/{len(scenes)} scene(s) valid.")
    sys.exit(0 if n_ok == len(scenes) else 1)


if __name__ == "__main__":
    main()
