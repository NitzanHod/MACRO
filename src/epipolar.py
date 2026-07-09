"""
Per-(input, reference) cache used by the cross-view attention.

For each (close-up input, reference) pair, `get_epipolar_cache` computes the
fundamental matrix F from the two COLMAP poses + intrinsics and packs it —
together with the images, zoom, and crop bbox — into an `EpipolarCache`.

Note on what MACRO actually uses: MACRO's attention (mask_mode="all") routes
each close-up token to the reference tokens of *its own depth plane* — the
bin-routing + occlusion masks built from `latent_bin_map` / `ref_to_bin` in
attention.py. It does NOT use the epipolar fundamental matrix F for masking.
The F / epipolar-line masking here is a legacy path (mask_mode="epipolar",
not in the public config set); on the macro path F is only read by the
attention-visualization dashboards. The cache is still constructed on the live
path because attention.py reads `ref_crop_bbox` (mask-cache key) and the
images/zoom (viz) from it.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import os
from PIL import Image

class EpipolarCache:
    def __init__(self, F_norm, original_size, img_query_pil=None, img_target_pil=None, zoom=1.0, ref_crop_bbox=None):
        """
        F_norm: Fundamental Matrix mapping p_query -> line_target
                line_target = F @ p_query
                (Calculated from Query -> Target)
        original_size: (W, H) of the full images.
        img_query_pil: The full resolution Input/Query image.
        img_target_pil: The full resolution Reference/Target image.
        zoom: The focal length multiplier applied to the Query Image.
        ref_crop_bbox: (u_min, v_min, u_max, v_max) in normalized [0,1] UV of the full reference, or None.
        """
        self.F_norm = F_norm
        self.original_size = original_size
        self.img_query_pil = img_query_pil
        self.img_target_pil = img_target_pil
        self.zoom = zoom
        self.ref_crop_bbox = ref_crop_bbox

    def get_lines_for_layer(self, height, width):
        """
        Generates epipolar lines for a specific latent layer resolution.
        Automatically applies the stored zoom factor to the Query grid.
        """
        device = self.F_norm.device
        
        # 1. Calculate the UV Box for the Query Image
        # If zoom=1.0, we cover [0, 1].
        # If zoom=2.0, we cover [0.25, 0.75].
        crop_size = 1.0 / self.zoom
        center = 0.5
        
        u_min = center - crop_size / 2.0
        u_max = center + crop_size / 2.0
        v_min = center - crop_size / 2.0
        v_max = center + crop_size / 2.0
        
        # 2. Generate Grid (Pixel Centers)
        # We define pixel 0 as the center of the first grid cell in the crop.
        y_step = (v_max - v_min) / height
        x_step = (u_max - u_min) / width
        
        # Standard Pixel Center logic: Start + Half Step
        y_range = torch.linspace(v_min + y_step/2, v_max - y_step/2, height, device=device)
        x_range = torch.linspace(u_min + x_step/2, u_max - x_step/2, width, device=device)
        
        grid_v, grid_u = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # 3. Project Query Points to Target Lines (F @ p)
        # These points are in the Normalized Coordinate System of the *Original* Query Image.
        pts1_flat = torch.stack([grid_u.flatten(), grid_v.flatten(), torch.ones_like(grid_u.flatten())], dim=0)
        lines_flat = self.F_norm @ pts1_flat
        a, b, c = lines_flat[0], lines_flat[1], lines_flat[2]

        # 4. Intersect with TARGET Viewport [0, 1]
        # Note: The Reference (Target) is NOT zoomed; we search the whole reference view.
        eps = 1e-6
        v_at_0 = -c / (b + eps); v_at_1 = -(c + a) / (b + eps)
        u_at_0 = -c / (a + eps); u_at_1 = -(c + b) / (a + eps)

        cands_u = torch.stack([torch.zeros_like(v_at_0), torch.ones_like(v_at_1), u_at_0, u_at_1])
        cands_v = torch.stack([v_at_0, v_at_1, torch.zeros_like(u_at_0), torch.ones_like(u_at_1)])
        valid_mask = (cands_u >= -1e-4) & (cands_u <= 1.0001) & (cands_v >= -1e-4) & (cands_v <= 1.0001)

        # Robust Clipper
        scores = cands_u * b.unsqueeze(0) - cands_v * a.unsqueeze(0)
        scores_min = scores.clone(); scores_min[~valid_mask] = float('inf')
        scores_max = scores.clone(); scores_max[~valid_mask] = float('-inf')
        
        idx_start = torch.argmin(scores_min, dim=0)
        idx_end   = torch.argmax(scores_max, dim=0)
        
        def gather_val(src, idx): return torch.gather(src, 0, idx.unsqueeze(0)).squeeze(0)
        u_s, v_s = gather_val(cands_u, idx_start), gather_val(cands_v, idx_start)
        u_e, v_e = gather_val(cands_u, idx_end),   gather_val(cands_v, idx_end)

        # Handle Invalid
        invalid = (scores_min.min(dim=0)[0] == float('inf'))
        u_s[invalid] = -1; v_s[invalid] = -1
        u_e[invalid] = -1; v_e[invalid] = -1

        output_flat = torch.stack([u_s, v_s, u_e, v_e], dim=1)
        
        # Log once
        if not hasattr(self, '_logged_zoom'):
            print(f"[CACHE] Layer {height}x{width} | Zoom: {self.zoom:.2f}")
            print(f"  > Query Crop Box: U[{u_min:.3f}-{u_max:.3f}], V[{v_min:.3f}-{v_max:.3f}]")
            self._logged_zoom = True

        return output_flat.view(1, height, width, 4).permute(0, 3, 1, 2)

def get_epipolar_cache(img_in_pil, img_ref_pil, img_input_name, img_ref_name, transforms_path, device, viz_folder=None, frame_idx=None, zoom=1.0, forward_params=None, ref_crop_bbox=None):
    """
    Factory function to build the cache.
    img_in_pil:  The Input Image (Subject to Zoom).
    img_ref_pil: The Reference Image (Static).
    zoom: Focal length multiplier for the Input Image.
    forward_params: dict with 'scene_center' and 'forward_step_ratio', or None.
    ref_crop_bbox: (u_min, v_min, u_max, v_max) in normalized [0,1] UV of the full reference, or None.
    """
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    def get_matrix(fname):
        for frame in data["frames"]:
            if frame["file_path"] == fname:
                return torch.tensor(frame["transform_matrix"], device=device, dtype=torch.float32)
        raise ValueError(f"{fname} not found.")

    W_full = float(data["w"])
    H_full = float(data["h"])

    K_raw = torch.tensor([
        [float(data["fl_x"]), 0, float(data["cx"])],
        [0, float(data["fl_y"]), float(data["cy"])],
        [0, 0, 1]
    ], device=device)

    # Normalize K to [0, 1] UV space
    Norm_T = torch.tensor([[1.0/W_full, 0, 0], [0, 1.0/H_full, 0], [0, 0, 1]], device=device)
    K_norm = Norm_T @ K_raw
    K_inv = torch.linalg.inv(K_norm)

    # Coordinate System Fix (Blender/Colmap Y/Z flip)
    Fix_S = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    
    # 1. Get Matrices for Query (Input) and Target (Ref)
    M_query = get_matrix(img_input_name) @ Fix_S
    M_target = get_matrix(img_ref_name) @ Fix_S

    # Apply physical forwarding correction to query pose if provided
    if forward_params is not None:
        scene_center = torch.tensor(forward_params['scene_center'], device=device, dtype=torch.float32)
        ratio = forward_params['forward_step_ratio']
        R_close = M_query[:3, :3]
        t_close = M_query[:3, 3]
        forward_dir = R_close[:, 2]
        forward_dir = forward_dir / forward_dir.norm()
        step_dist = ratio * (t_close - scene_center).norm()
        M_query[:3, 3] = t_close + step_dist * forward_dir

    # 2. Calculate Relative Motion: QUERY -> TARGET
    # F matrix will map: Point in Query -> Line in Target
    
    w2c_query = torch.linalg.inv(M_query)
    w2c_target = torch.linalg.inv(M_target)
    
    R_query, t_query = w2c_query[:3, :3], w2c_query[:3, 3]
    R_target, t_target = w2c_target[:3, :3], w2c_target[:3, 3]

    # P_target = R_rel * P_query + t_rel
    R_rel = R_target @ R_query.T
    t_rel = t_target - R_rel @ t_query
    
    # Essential Matrix [t]x R
    t_skew = torch.tensor([
        [0, -t_rel[2], t_rel[1]], 
        [t_rel[2], 0, -t_rel[0]], 
        [-t_rel[1], t_rel[0], 0]
    ], device=device)
    
    E = t_skew @ R_rel
    
    # Fundamental Matrix (Normalized)
    F_norm = K_inv.T @ E @ K_inv
    
    return EpipolarCache(F_norm, (W_full, H_full), img_query_pil=img_in_pil, img_target_pil=img_ref_pil, zoom=zoom, ref_crop_bbox=ref_crop_bbox)