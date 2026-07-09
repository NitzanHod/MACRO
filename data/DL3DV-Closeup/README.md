# DL3DV-Closeup splits

This folder holds the **40 DL3DV-Closeup benchmark splits** — one `<scene_id>.json` per
scene. DL3DV-Closeup is defined entirely by these splits, selected from the public
[DL3DV-10K Benchmark](https://github.com/DL3DV-10K/Dataset); we do not redistribute DL3DV
imagery. Bring your own DL3DV-10K Benchmark copy and assemble each scene folder using the
split as the manifest.

## Target scene-folder layout

```
DL3DV-Closeup/<scene_id>/
  images/        # the training_frames, from the scene's DL3DV images (images_4 / data_factor 4)
  closeup_gt/    # the closeup frames (keys of closeup_poses), same source
  sparse/0/      # COLMAP model covering those views (cameras.bin, images.bin, points3D.bin)
  split.json     # copy the matching <scene_id>.json from this folder
```

`<scene_id>` is the 64-char DL3DV hash (the split filename without `.json`).

## split.json fields

| Key | Meaning |
|---|---|
| `scene` | DL3DV scene id (64-char hash). |
| `data_factor` | Downsample factor of the images used (4 → DL3DV `nerfstudio/images_4`). |
| `intrinsics` | `w, h, fl_x, fl_y, cx, cy` (+ OPENCV distortion `k1,k2,p1,p2`, `camera_model`). |
| `training_frames` | List of wide-shot frame filenames (e.g. `frame_00101.png`) → go in `images/`. |
| `closeup_poses` | `{closeup_frame: 4×4 camera-to-world (OpenGL)}` → the close-up targets → `closeup_gt/`. |
| `closeup_eval_pairs` | Scored `(closeup_frame, training_frame, depth_ratio, iou, …)` tuples. |
| `num_training_frames`, `num_closeup_pairs` | Convenience counts. |

Frame filenames match DL3DV's `nerfstudio` naming, so the manifest maps directly onto a
standard DL3DV scene. Poses are in the same world frame as the scene's COLMAP model, which
is why any COLMAP covering the listed views works for rendering.
