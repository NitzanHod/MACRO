"""
Reference selection via greedy set cover on depth-based forward projections.

Given an input close-up frame and the scene's training views, selects K
references that maximize pixel coverage of the input by forward-projecting
each candidate's rendered depth into the input camera. MACRO uses K=3.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict
from warp import load_master_depth


def select_references(
    input_frame_name: str,
    train_frame_names: List[str],
    frame_lookup: Dict,
    transforms_path: str,
    depth_folder: str,
    depth_mode: str = 'master',
    forward_params: Optional[Dict] = None,
    K: int = 4,
    coverage_resolution: int = 256,
    device: str = 'cuda',
    mvs_depth_folder: Optional[str] = None,
    depth_source: str = 'gsplat',
) -> Tuple[List[str], List[np.ndarray], float]:
    """
    Select K references that maximize pixel coverage of the input frame.

    Args:
        input_frame_name: file_path value from transforms.json (e.g. "images/frame_00040.png")
        train_frame_names: list of N reference file_path values
        frame_lookup: dict mapping frame basename -> transforms.json frame entry
        transforms_path: path to transforms.json
        depth_folder: path to gsplat depth folder (train_renders/)
        depth_mode: 'master' (only supported mode)
        forward_params: dict with 'scene_center' and 'forward_step_ratio', or None
        K: number of references to select
        coverage_resolution: downsampled resolution for coverage computation
        device: 'cuda' or 'cpu'
        mvs_depth_folder: path to mvsanywhere_predictions/ (optional)
        depth_source: 'gsplat' or 'mvs' — which ref depth to use for coverage

    Returns:
        selected_refs: list of K frame file_path strings
        coverage_maps: list of K boolean ndarrays (coverage_resolution x coverage_resolution)
        total_coverage: fraction of input pixels covered by union of K refs
    """
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    W_json, H_json = float(data['w']), float(data['h'])

    # Base intrinsics (at transforms.json resolution)
    K_base = torch.tensor([
        [float(data['fl_x']), 0, float(data['cx'])],
        [0, float(data['fl_y']), float(data['cy'])],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)

    Fix_S = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))

    def get_pose(fname):
        """Get c2w pose from transforms.json frame entry."""
        # fname is file_path like "images/frame_00040.png"
        basename = os.path.basename(fname)
        if basename not in frame_lookup:
            raise KeyError(f"Frame {basename} not in frame_lookup")
        entry = frame_lookup[basename]
        return torch.tensor(entry["transform_matrix"], device=device, dtype=torch.float32)

    # Input camera pose (with forward_params correction)
    c2w_input = get_pose(input_frame_name) @ Fix_S
    if forward_params is not None:
        scene_center = torch.tensor(forward_params['scene_center'], device=device, dtype=torch.float32)
        ratio = forward_params['forward_step_ratio']
        R_in = c2w_input[:3, :3]
        t_in = c2w_input[:3, 3]
        fwd_dir = R_in[:, 2]
        fwd_dir = fwd_dir / fwd_dir.norm()
        step_dist = ratio * (t_in - scene_center).norm()
        c2w_input[:3, 3] = t_in + step_dist * fwd_dir

    w2c_input = torch.linalg.inv(c2w_input)

    # Scale K for coverage resolution (square for simplicity)
    K_cov = K_base.clone()
    K_cov[0, :] *= coverage_resolution / W_json
    K_cov[1, :] *= coverage_resolution / H_json

    # Compute coverage mask for each reference
    all_coverage = []  # list of (ref_name, bool_mask)

    for ref_name in train_frame_names:
        ref_basename = os.path.basename(ref_name)
        depth_base = os.path.splitext(ref_basename)[0]  # e.g. "frame_00040"

        # Resolve depth path based on depth_source
        if depth_source == 'mvs' and mvs_depth_folder is not None:
            depth_path = os.path.join(mvs_depth_folder, ref_basename.replace('.png', '.npz'))
        else:
            depth_path = os.path.join(depth_folder, f"{depth_base}_depth.tiff")

        if not os.path.exists(depth_path):
            print(f"[references] Depth not found for {ref_basename}, skipping: {depth_path}")
            all_coverage.append((ref_name, np.zeros((coverage_resolution, coverage_resolution), dtype=bool)))
            continue

        # Load depth and ref pose
        depth_ref = load_master_depth(depth_path, device=device)  # (1, 1, H_d, W_d)
        H_d, W_d = depth_ref.shape[-2:]

        c2w_ref = get_pose(ref_name) @ Fix_S
        # No forward_params on ref pose — only input gets corrected

        # Scale K for ref depth resolution
        K_ref = K_base.clone()
        K_ref[0, :] *= W_d / W_json
        K_ref[1, :] *= H_d / H_json
        K_ref_inv = torch.linalg.inv(K_ref)

        # Unproject ref depth pixels to world, project into input camera
        y_r, x_r = torch.meshgrid(
            torch.arange(H_d, device=device),
            torch.arange(W_d, device=device),
            indexing='ij'
        )
        grid_homo = torch.stack([x_r.flatten(), y_r.flatten(), torch.ones(H_d * W_d, device=device)], dim=0).float()
        d_flat = depth_ref.view(1, -1)

        # Unproject from ref camera
        cam_pts = (K_ref_inv @ grid_homo) * d_flat  # (3, N)
        ones = torch.ones((1, cam_pts.shape[1]), device=device)
        world_pts = c2w_ref @ torch.cat([cam_pts, ones], dim=0)  # (4, N)

        # Project into input camera at coverage resolution
        cam_in = w2c_input @ world_pts  # (4, N)
        z = cam_in[2:3, :]
        uv = K_cov @ cam_in[:3, :]
        u = uv[0, :] / (uv[2, :] + 1e-6)
        v = uv[1, :] / (uv[2, :] + 1e-6)

        # Valid: positive depth, within coverage grid
        valid = (z.squeeze() > 0) & (u >= 0) & (u < coverage_resolution) & (v >= 0) & (v < coverage_resolution)

        # Build binary coverage mask
        mask = np.zeros((coverage_resolution, coverage_resolution), dtype=bool)
        if valid.any():
            u_valid = u[valid].long().cpu().numpy()
            v_valid = v[valid].long().cpu().numpy()
            mask[v_valid, u_valid] = True

        all_coverage.append((ref_name, mask))

    # Greedy set cover
    covered = np.zeros((coverage_resolution, coverage_resolution), dtype=bool)
    selected_refs = []
    coverage_maps = []
    remaining = list(range(len(all_coverage)))

    for _ in range(min(K, len(train_frame_names))):
        best_idx = -1
        best_new_count = -1

        for idx in remaining:
            _, mask = all_coverage[idx]
            new_pixels = np.sum(mask & ~covered)
            if new_pixels > best_new_count:
                best_new_count = new_pixels
                best_idx = idx

        if best_idx < 0 or best_new_count == 0:
            # No more useful references — pick any remaining
            if remaining:
                best_idx = remaining[0]
            else:
                break

        ref_name, mask = all_coverage[best_idx]
        selected_refs.append(ref_name)
        coverage_maps.append(mask)
        covered = covered | mask
        remaining.remove(best_idx)

    # Pad if fewer than K refs available
    while len(selected_refs) < K and len(train_frame_names) > 0:
        # Duplicate last selected ref
        selected_refs.append(selected_refs[-1])
        coverage_maps.append(coverage_maps[-1])

    total_pixels = coverage_resolution * coverage_resolution
    total_coverage = np.sum(covered) / total_pixels

    print(f"[references] Selected {len(selected_refs)} refs, coverage: {total_coverage*100:.1f}%")
    for i, name in enumerate(selected_refs):
        print(f"  ref[{i}]: {os.path.basename(name)}")

    return selected_refs, coverage_maps, total_coverage
