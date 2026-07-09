"""
Depth-based reference warping and crop synthesis.

`warp_reference_to_closeup` projects a reference view into the close-up frame
using the rendered depth (forward or backward warp), producing the scale-
matched reference crop that conditions the DiFix step. `load_master_depth`
reads the float32 depth .tiff written by the eval driver; `superres_pil` is the
crop-upscaling entry that hands off to the PFT-SR subprocess.
"""
import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio
import json
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


def superres_pil(crop_pil, target_size, sr_model, device='cuda'):
    """
    When sr_model is True (flag), return raw canvas — batch SR handles upscaling later.
    When sr_model is None, return LANCZOS-resized version.
    """
    if sr_model:
        # Return raw canvas unchanged — batch SR in enhance.py will upscale
        lanczos_pil = crop_pil.resize(target_size, Image.LANCZOS)
        return crop_pil, lanczos_pil  # output_pil = raw, lanczos for viz comparison
    else:
        lanczos_pil = crop_pil.resize(target_size, Image.LANCZOS)
        return lanczos_pil, lanczos_pil


def load_master_depth(path, device='cuda'):
    """
    Restores a float32 depth map into a PyTorch tensor.
    Supports .tiff (imageio) and .npz (numpy, key='depth', shape (1,H,W) or (H,W)).
    
    Returns:
        torch.Tensor: Shape (1, 1, H, W), Float32
    """
    if path.endswith('.npz'):
        data = np.load(path)
        depth_np = data['depth'].astype(np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[0]  # (1, H, W) -> (H, W)
        return torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # 1. Read using imageio (handles TIFF natively)
    depth_np = iio.imread(path)
    
    # 2. Ensure shape is (H, W) or (H, W, 1) -> Convert to Tensor
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]
        
    depth_tensor = torch.from_numpy(depth_np).float()
    
    # 3. Add Batch and Channel dimensions: (1, 1, H, W)
    return depth_tensor.unsqueeze(0).unsqueeze(0).to(device)





def warp_reference_to_closeup(
    img_ref_pil, img_closeup_pil,
    depth_ref_path, depth_closeup_path,
    img_ref_name, img_closeup_name,
    transforms_path,
    device='cuda',
    method='backward', # 'backward', 'forward'
    mode='warp',       # 'warp', 'crop'
    zoom_bbox=None,    # (left, top, right, bottom) integers
    depth_mode='master',
    forward_params=None,  # dict with 'scene_center' and 'forward_step_ratio', or None
    return_heatmap=False,  # if True, viz_pil shows projection density heatmap over reference
    fg_mask=None,  # (H_close, W_close) bool tensor on device — True=foreground, filters crop bbox
    crop_strategy='relative',  # 'relative' (existing) or 'geometric' (quadrilateral-based)
    bin_avg_depth=None,  # average depth of the bin (for geometric crop strategy)
    occlusion_depth_path=None,  # separate depth for occlusion filtering (e.g. MVS depth)
    return_warp_map=False,  # if True, return per-input-pixel warp coords in cropped ref space
    sr_model=None,  # PFT-SR model for super-resolution upscaling (None = LANCZOS)
):
    """
    Synthesizes the Closeup view (or a specific CROP of it) using the Reference texture.
    Returns: (synthesized_pil, viz_pil)
    """
    
    # ==========================================
    # 1. Load & Normalize Images/Depth
    # ==========================================
    
    def load_depth(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Depth not found: {path}")
        d = iio.imread(path)
        d = torch.from_numpy(d).float().to(device)
        if d.ndim == 3: d = d[..., 0]
        return d.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    def pil_to_tensor(pil_img):
        arr = np.array(pil_img)
        t = torch.from_numpy(arr).float().to(device) / 255.0
        return t.permute(2, 0, 1).unsqueeze(0)

    img_ref = pil_to_tensor(img_ref_pil)
    depth_ref = load_master_depth(depth_ref_path) if 'master' in depth_mode else load_depth(depth_ref_path)
    depth_close = load_depth(depth_closeup_path) if depth_closeup_path is not None else None

    # Separate dimensions for ref and closeup (they may differ)
    H_ref, W_ref = img_ref_pil.height, img_ref_pil.width
    H_close, W_close = img_closeup_pil.height, img_closeup_pil.width
    
    # Resize depths to match their respective images
    if depth_ref.shape[-2:] != (H_ref, W_ref):
        depth_ref = F.interpolate(depth_ref, size=(H_ref, W_ref), mode='nearest')
    if depth_close is not None and depth_close.shape[-2:] != (H_close, W_close):
        depth_close = F.interpolate(depth_close, size=(H_close, W_close), mode='nearest')

    # ==========================================
    # 2. Parse Camera Parameters
    # ==========================================
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    W_json, H_json = float(data['w']), float(data['h'])
    
    # Base intrinsics (at transforms.json resolution)
    K_base = torch.tensor([
        [float(data['fl_x']), 0, float(data['cx'])],
        [0, float(data['fl_y']), float(data['cy'])],
        [0, 0, 1]
    ], device=device)
    
    # Scale K for closeup image resolution
    K_close = K_base.clone()
    K_close[0, :] *= W_close / W_json
    K_close[1, :] *= H_close / H_json
    K_close_inv = torch.linalg.inv(K_close)

    # Scale K for reference image resolution
    K_ref = K_base.clone()
    K_ref[0, :] *= W_ref / W_json
    K_ref[1, :] *= H_ref / H_json
    K_ref_inv = torch.linalg.inv(K_ref)

    def get_pose(fname):
        for frame in data["frames"]:
            if frame["file_path"] == fname:
                return torch.tensor(frame["transform_matrix"], device=device)
        raise ValueError(f"{fname} not found")

    # OpenGL -> OpenCV Fix
    Fix_S = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    c2w_ref = get_pose(img_ref_name) @ Fix_S
    c2w_close = get_pose(img_closeup_name) @ Fix_S

    # Apply physical forwarding correction if provided
    if forward_params is not None:
        scene_center = torch.tensor(forward_params['scene_center'], device=device, dtype=torch.float32)
        ratio = forward_params['forward_step_ratio']
        R_close = c2w_close[:3, :3]
        t_close = c2w_close[:3, 3]
        forward_dir = R_close[:, 2]
        forward_dir = forward_dir / forward_dir.norm()
        step_dist = ratio * (t_close - scene_center).norm()
        c2w_close[:3, 3] = t_close + step_dist * forward_dir

    w2c_ref = torch.linalg.inv(c2w_ref)
    w2c_close = torch.linalg.inv(c2w_close)

    # ==========================================
    # 3. Handle Zoom/Crop Logic
    # ==========================================
    if zoom_bbox is None:
        left, top, right, bottom = 0, 0, W_close, H_close
    else:
        left, top, right, bottom = zoom_bbox
    
    target_w = right - left
    target_h = bottom - top

    # Geometry Helpers
    def unproject(uv_homo, depth_flat, c2w_mat, K_inv):
        uv_homo = uv_homo.float()
        cam_points = (K_inv @ uv_homo) * depth_flat
        ones = torch.ones((1, cam_points.shape[1]), device=device)
        world_points = c2w_mat @ torch.cat([cam_points, ones], dim=0)
        return world_points

    def project(world_points, w2c_mat, K):
        cam_points = w2c_mat @ world_points
        z = cam_points[2:3, :] + 1e-6
        p_homo = K @ cam_points[:3, :]
        uv = p_homo[:2, :] / p_homo[2:3, :]
        return uv, z

    # ==========================================
    # 4. Processing
    # ==========================================
    
    viz_points_u = []
    viz_points_v = []

    # --- BACKWARD: Driven by Closeup Pixels ---
    if method == 'backward':
        y_range = torch.arange(top, bottom, device=device)
        x_range = torch.arange(left, right, device=device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
        
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        ones = torch.ones_like(flat_x)
        grid_homo = torch.stack([flat_x, flat_y, ones], dim=0)
        
        d_crop = depth_close[..., top:bottom, left:right]
        d_flat = d_crop.reshape(1, -1)
        
        # Unproject from closeup (K_close), project into ref (K_ref)
        pts_world = unproject(grid_homo, d_flat, c2w_close, K_close_inv)
        uv_proj, z_proj = project(pts_world, w2c_ref, K_ref)
        
        valid_mask = (z_proj > 0) & \
                     (d_flat.squeeze() > 0) & \
                     (uv_proj[0] >= 0) & (uv_proj[0] < W_ref) & \
                     (uv_proj[1] >= 0) & (uv_proj[1] < H_ref)
        
        u_sample = uv_proj[0]
        v_sample = uv_proj[1]
        
        idx_viz = torch.where(valid_mask.squeeze())[0]
        if len(idx_viz) > 0:
            perm = torch.randperm(len(idx_viz))[:500]
            sel = idx_viz[perm]
            viz_points_u = u_sample[sel].cpu().numpy()
            viz_points_v = v_sample[sel].cpu().numpy()

    # --- FORWARD: Driven by Ref Pixels ---
    elif method == 'forward':
        # Grid over reference image pixels
        y_r, x_r = torch.meshgrid(torch.arange(H_ref, device=device), torch.arange(W_ref, device=device), indexing='ij')
        grid_homo = torch.stack([x_r.flatten(), y_r.flatten(), torch.ones_like(x_r.flatten())], dim=0)
        d_flat = depth_ref.view(1, -1)
        
        # Unproject from ref (K_ref), project into closeup (K_close)
        pts_world = unproject(grid_homo, d_flat, c2w_ref, K_ref_inv)
        uv_close, z_close = project(pts_world, w2c_close, K_close)
        
        # 3. Filter: Which Ref pixels land inside our ZOOM CROP?
        u_c = uv_close[0]
        v_c = uv_close[1]
        
        in_crop = (z_close.squeeze() > 0) & \
                  (u_c >= left) & (u_c < right) & \
                  (v_c >= top) & (v_c < bottom)
        
        # Data for warping
        # Source indices (Ref pixels) that are valid
        src_indices = torch.where(in_crop)[0]
        
        # Destination coordinates relative to crop
        dst_u = u_c[in_crop] - left
        dst_v = v_c[in_crop] - top
        
        # Viz Data: The Source Pixels (x_r, y_r) that passed the test
        if len(src_indices) > 0:
            perm = torch.randperm(len(src_indices))[:500]
            sel = src_indices[perm]
            viz_points_u = x_r.flatten()[sel].cpu().numpy()
            viz_points_v = y_r.flatten()[sel].cpu().numpy()

    # ==========================================
    # 5. Render / Crop
    # ==========================================
    
    output_pil = None
    lanczos_pil_out = None  # LANCZOS version when SR is used, for comparison viz

    if mode == 'warp':
        if method == 'backward':
            # Bilinear Sample from Ref (normalize against ref dimensions)
            norm_u = 2.0 * (u_sample / (W_ref - 1)) - 1.0
            norm_v = 2.0 * (v_sample / (H_ref - 1)) - 1.0
            grid_norm = torch.stack([norm_u, norm_v], dim=-1).view(1, target_h, target_w, 2)
            
            warped = F.grid_sample(img_ref, grid_norm, align_corners=True, padding_mode='zeros')
            warped = warped * valid_mask.view(1, 1, target_h, target_w).float()
            output_pil = tensor_to_pil(warped)
            
        elif method == 'forward':
            # Splat onto Crop Canvas
            canvas = torch.zeros((1, 3, target_h, target_w), device=device)
            canvas_flat = canvas.view(3, -1)
            
            # Destination indices in flattened crop
            dst_u_int = torch.round(dst_u).long()
            dst_v_int = torch.round(dst_v).long()
            
            # Safety clamp (rounding might push slightly out)
            valid_splat = (dst_u_int >= 0) & (dst_u_int < target_w) & \
                          (dst_v_int >= 0) & (dst_v_int < target_h)
            
            idx_dst = dst_v_int[valid_splat] * target_w + dst_u_int[valid_splat]
            
            # Source colors
            ref_flat = img_ref.view(3, -1)
            cols_src = ref_flat[:, src_indices[valid_splat]]
            
            # Splat (Simple overwriting)
            canvas_flat[:, idx_dst] = cols_src
            output_pil = tensor_to_pil(canvas)

    elif mode == 'crop':
        # Bounding Box Logic
        if method == 'backward':
            # u_sample contains Ref coordinates for every pixel in the crop
            # Filter invalid ones
            valid_mask_flat = valid_mask.squeeze()
            # Apply foreground filter: only keep input pixels that are foreground
            if fg_mask is not None:
                fg_flat = fg_mask[top:bottom, left:right].flatten()
                valid_mask_flat = valid_mask_flat & fg_flat
            valid_u = u_sample[valid_mask_flat]
            valid_v = v_sample[valid_mask_flat]
            
        elif method == 'forward':
            # src_indices contains indices of Ref pixels that landed in crop
            # Apply foreground filter: only keep ref pixels that land on foreground input pixels
            if fg_mask is not None and len(src_indices) > 0:
                # dst coords are in input pixel space (u_c, v_c already computed)
                dst_u_int = torch.round(u_c[src_indices]).long().clamp(0, W_close - 1)
                dst_v_int = torch.round(v_c[src_indices]).long().clamp(0, H_close - 1)
                fg_hit = fg_mask[dst_v_int, dst_u_int]
                fg_src_indices = src_indices[fg_hit]
            else:
                fg_src_indices = src_indices
            valid_u = x_r.flatten()[fg_src_indices]
            valid_v = y_r.flatten()[fg_src_indices]
            
        if len(valid_u) == 0:
            print("Warning: No overlap for crop.")
            # Return black or resize whole ref
            output_pil = img_ref_pil.resize((target_w, target_h))
            viz_bbox = None
        else:
            if crop_strategy == 'geometric' and bin_avg_depth is not None:
                # --- Geometric crop: distance-ratio scale ---
                # Compute 3D point at bin's median depth along input optical axis
                # Use image center as representative point
                center_uv = torch.tensor([W_close / 2.0, H_close / 2.0, 1.0], device=device, dtype=torch.float32)
                cam_pt = (K_close_inv @ center_uv) * bin_avg_depth  # (3,)
                ones_1 = torch.ones(1, device=device)
                world_pt = c2w_close @ torch.cat([cam_pt, ones_1])  # (4,)

                # Distance from input camera to 3D point
                d_input = bin_avg_depth  # by construction (z-depth along optical axis)

                # Distance from ref camera to same 3D point
                cam_ref_pt = w2c_ref @ world_pt  # (4,)
                d_ref = cam_ref_pt[2].item()  # z-depth in ref camera

                if d_ref <= 0 or d_input <= 0:
                    print(f"  [crop:geo] Invalid distances (d_in={d_input:.2f}, d_ref={d_ref:.2f}), no valid crop")
                    output_pil = img_ref_pil.resize((target_w, target_h))
                    viz_bbox = None
                else:
                    scale_ratio = d_input / d_ref  # <1 means ref is farther (crop smaller), >1 means ref is closer (crop larger)
                    crop_w = target_w * scale_ratio
                    crop_h = target_h * scale_ratio

                    # Guard degenerate
                    if crop_w > W_ref * 5 or crop_h > H_ref * 5 or crop_w < 1 or crop_h < 1:
                        print(f"  [crop:geo] Degenerate scale ratio={scale_ratio:.2f}, no valid crop")
                        output_pil = img_ref_pil.resize((target_w, target_h))
                        viz_bbox = None
                    else:
                        # Center at centroid of surviving warped points
                        crop_cx = valid_u.float().mean().item()
                        crop_cy = valid_v.float().mean().item()

                        min_x = crop_cx - crop_w / 2.0
                        min_y = crop_cy - crop_h / 2.0
                        max_x = min_x + crop_w
                        max_y = min_y + crop_h

                        viz_bbox = (min_x, min_y, max_x, max_y)
                        ix0, iy0 = int(round(min_x)), int(round(min_y))
                        ix1, iy1 = int(round(max_x)), int(round(max_y))
                        cw, ch = ix1 - ix0, iy1 - iy0
                        # print(f"  [crop:geo] d_in={d_input:.2f} d_ref={d_ref:.2f} ratio={scale_ratio:.3f} crop={cw}x{ch}")
                        if cw <= 0 or ch <= 0 or cw > 10000 or ch > 10000:
                            output_pil = img_ref_pil.resize((target_w, target_h))
                        else:
                            canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
                            src_x0, src_y0 = max(ix0, 0), max(iy0, 0)
                            src_x1, src_y1 = min(ix1, W_ref), min(iy1, H_ref)
                            if src_x1 > src_x0 and src_y1 > src_y0:
                                region = img_ref_pil.crop((src_x0, src_y0, src_x1, src_y1))
                                canvas.paste(region, (src_x0 - ix0, src_y0 - iy0))
                            if sr_model is not None:
                                output_pil, lanczos_pil_out = superres_pil(canvas, (target_w, target_h), sr_model, device)
                            else:
                                output_pil = canvas.resize((target_w, target_h), Image.LANCZOS)
            else:
                # --- Existing relative crop strategy ---
                proj_min_x, proj_max_x = valid_u.min().item(), valid_u.max().item()
                proj_min_y, proj_max_y = valid_v.min().item(), valid_v.max().item()
                proj_w = max(proj_max_x - proj_min_x, 1.0)
                proj_h = max(proj_max_y - proj_min_y, 1.0)
                proj_cx = (proj_min_x + proj_max_x) / 2.0
                proj_cy = (proj_min_y + proj_max_y) / 2.0

                if fg_mask is not None:
                    fg_ys, fg_xs = torch.where(fg_mask)
                    if len(fg_xs) > 0:
                        fg_u0 = fg_xs.min().item() / W_close
                        fg_u1 = fg_xs.max().item() / W_close
                        fg_v0 = fg_ys.min().item() / H_close
                        fg_v1 = fg_ys.max().item() / H_close
                        fg_cx_norm = (fg_u0 + fg_u1) / 2.0
                        fg_cy_norm = (fg_v0 + fg_v1) / 2.0
                        fg_frac_w = max(fg_u1 - fg_u0, 0.01)
                        fg_frac_h = max(fg_v1 - fg_v0, 0.01)
                    else:
                        fg_cx_norm, fg_cy_norm = 0.5, 0.5
                        fg_frac_w, fg_frac_h = 1.0, 1.0
                else:
                    fg_cx_norm, fg_cy_norm = 0.5, 0.5
                    fg_frac_w, fg_frac_h = 1.0, 1.0

                crop_w = proj_w / fg_frac_w
                crop_h = proj_h / fg_frac_h
                target_ratio = target_w / target_h
                if crop_w / crop_h < target_ratio:
                    crop_w = crop_h * target_ratio
                else:
                    crop_h = crop_w / target_ratio
                min_x = proj_cx - fg_cx_norm * crop_w
                min_y = proj_cy - fg_cy_norm * crop_h
                max_x = min_x + crop_w
                max_y = min_y + crop_h

                viz_bbox = (min_x, min_y, max_x, max_y)
                ix0, iy0 = int(round(min_x)), int(round(min_y))
                ix1, iy1 = int(round(max_x)), int(round(max_y))
                cw, ch = ix1 - ix0, iy1 - iy0
                if cw <= 0 or ch <= 0:
                    output_pil = img_ref_pil.resize((target_w, target_h))
                else:
                    canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
                    src_x0, src_y0 = max(ix0, 0), max(iy0, 0)
                    src_x1, src_y1 = min(ix1, W_ref), min(iy1, H_ref)
                    if src_x1 > src_x0 and src_y1 > src_y0:
                        region = img_ref_pil.crop((src_x0, src_y0, src_x1, src_y1))
                        canvas.paste(region, (src_x0 - ix0, src_y0 - iy0))
                    if sr_model is not None:
                        output_pil, lanczos_pil_out = superres_pil(canvas, (target_w, target_h), sr_model, device)
                    else:
                        output_pil = canvas.resize((target_w, target_h), Image.LANCZOS)

    # ==========================================
    # 6. Visualization Image Generation
    # ==========================================
    if return_heatmap and mode == 'crop' and 'valid_u' in dir() and len(valid_u) > 0:
        # Build density heatmap over reference image from projected points
        heatmap = np.zeros((H_ref, W_ref), dtype=np.float32)
        vu = valid_u.cpu().numpy()
        vv = valid_v.cpu().numpy()
        vu_int = np.clip(np.round(vu).astype(int), 0, W_ref - 1)
        vv_int = np.clip(np.round(vv).astype(int), 0, H_ref - 1)
        np.add.at(heatmap, (vv_int, vu_int), 1.0)

        # Gaussian blur for smooth visualization
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Apply colormap and blend with reference
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        ref_np = np.array(img_ref_pil.convert("RGB"))
        blended = (0.4 * ref_np + 0.6 * heatmap_color).astype(np.uint8)

        viz_pil = Image.fromarray(blended)
        draw = ImageDraw.Draw(viz_pil)
        if 'viz_bbox' in locals() and viz_bbox:
            draw.rectangle(viz_bbox, outline="lime", width=5)
    else:
        viz_pil = img_ref_pil.copy().convert("RGB")
        draw = ImageDraw.Draw(viz_pil)
        
        # Draw Bounding Box (if crop mode)
        if mode == 'crop' and 'viz_bbox' in locals() and viz_bbox:
            draw.rectangle(viz_bbox, outline="lime", width=5)
        
        # Draw Sample Points (Red)
        for x, y in zip(viz_points_u, viz_points_v):
            draw.ellipse((x-2, y-2, x+2, y+2), fill="red", outline="red")
        
    # Return crop bbox (pixel coords on ref image) as 3rd element, or None
    crop_bbox_out = None
    if mode == 'crop' and 'viz_bbox' in locals() and viz_bbox is not None:
        crop_bbox_out = viz_bbox  # (min_x, min_y, max_x, max_y) in ref pixel coords

    # Build warp map: per-input-pixel coords in cropped+resized ref model space
    warp_map_out = None
    if return_warp_map and mode == 'crop' and method == 'backward' and crop_bbox_out is not None:
        bx0, by0, bx1, by1 = crop_bbox_out
        cw = bx1 - bx0
        ch = by1 - by0
        if cw > 0 and ch > 0:
            # u_sample, v_sample: (H_close*W_close,) in full ref pixel coords
            # Transform to canvas coords (accounting for out-of-bounds black padding)
            u_canvas = (u_sample - bx0) / cw * target_w  # scale to model width
            v_canvas = (v_sample - by0) / ch * target_h  # scale to model height
            # Reshape to (H_close, W_close, 2), mark invalid as -1
            wm_u = torch.full((H_close, W_close), -1.0, device=device)
            wm_v = torch.full((H_close, W_close), -1.0, device=device)
            valid_flat = valid_mask.squeeze()
            wm_u.view(-1)[valid_flat] = u_canvas[valid_flat].float()
            wm_v.view(-1)[valid_flat] = v_canvas[valid_flat].float()
            warp_map_out = torch.stack([wm_u, wm_v], dim=-1)  # (H_close, W_close, 2)

    return output_pil, viz_pil, crop_bbox_out, warp_map_out, lanczos_pil_out

def tensor_to_pil(t):
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)