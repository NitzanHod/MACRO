"""Visualization helpers (debug dashboards, coverage/depth panels).

None of this affects the enhanced output or the reported metrics; it only runs
when visualization is enabled (the ENABLE_VIZ env var, or
config_params["enable_viz"]), dumping per-frame panels for inspection.
"""
import os
import json
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw

from imaging import resize_pil, concat_images_with_labels


def build_coverage_panel(input_pil, coverage_maps, display_size):
    """
    Draw colored dots on the input image showing where each reference covers.
    coverage_maps: list of K boolean ndarrays at coverage_resolution x coverage_resolution.
    display_size: (W, H) for the output panel.
    """
    # Start with the input image resized to display size
    panel = input_pil.resize(display_size, Image.LANCZOS).convert("RGB")
    panel_np = np.array(panel)
    disp_w, disp_h = display_size
    cov_h, cov_w = coverage_maps[0].shape

    for r, mask in enumerate(coverage_maps):
        color = REF_COLORS[r % len(REF_COLORS)]
        # Upscale mask to display resolution
        mask_up = cv2.resize(mask.astype(np.uint8), (disp_w, disp_h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        # Blend: where mask is True, mix color with existing pixel
        alpha = 0.5
        panel_np[mask_up] = (
            (1 - alpha) * panel_np[mask_up] + alpha * np.array(color)
        ).astype(np.uint8)

    return Image.fromarray(panel_np)


def depth_to_jet_pil(depth_np, vmin, vmax, size=None):
    """Convert a 2D depth array to a JET-colorized PIL image.
    depth_np: (H, W) float array. vmin/vmax: shared color scale.
    Returns PIL RGB image, optionally resized to (W, H) = size.
    """
    d = np.clip(depth_np, vmin, vmax)
    if vmax - vmin > 1e-6:
        d = (d - vmin) / (vmax - vmin)
    else:
        d = np.zeros_like(d)
    jet = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(jet_rgb)
    if size is not None:
        pil = pil.resize(size, Image.LANCZOS)
    return pil


def load_depth_for_viz(ref_basename, depth_source, mvs_depth_folder, gsplat_depth_folder):
    """Load a depth map for visualization. Returns (H, W) float numpy array or None."""
    depth_base = os.path.splitext(ref_basename)[0]
    if depth_source == 'mvs' and mvs_depth_folder is not None:
        path = os.path.join(mvs_depth_folder, ref_basename.replace('.png', '.npz'))
        if os.path.exists(path):
            d = np.load(path)['depth'].astype(np.float32)
            if d.ndim == 3:
                d = d[0]
            return d
    path = os.path.join(gsplat_depth_folder, f"{depth_base}_depth.tiff")
    if os.path.exists(path):
        from PIL import Image as PILImage
        return np.array(PILImage.open(path)).astype(np.float32)


def save_confidence_viz(coverage, input_pil, save_path, n_train=16):
    """Save confidence mask visualization: 4 distinct colors for 0, 1, 2, 3+ coverage."""
    H, W = coverage.shape
    cov_np = coverage.cpu().numpy().astype(np.int32)

    # 4 distinct colors: 0=red, 1=orange, 2=yellow, 3+=green
    colors = {
        0: (255, 0, 0),      # red — disoccluded
        1: (255, 160, 0),    # orange — barely covered
        2: (255, 255, 0),    # yellow — low coverage
        3: (0, 200, 0),      # green — well covered (3+)
    }
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for val, col in colors.items():
        if val < 3:
            mask = cov_np == val
        else:
            mask = cov_np >= val
        overlay[mask] = col

    # Blend with input
    inp_np = np.array(input_pil.convert("RGB").resize((W, H), Image.LANCZOS))
    blended = (0.4 * inp_np + 0.6 * overlay).astype(np.uint8)

    # Add legend
    legend_w = 80
    legend = np.zeros((H, legend_w, 3), dtype=np.uint8)
    labels = ["0: none", "1: one", "2: two", "3+: good"]
    band_h = H // 4
    for i, (lbl, col) in enumerate(zip(labels, colors.values())):
        y0 = i * band_h
        y1 = (i + 1) * band_h if i < 3 else H
        legend[y0:y1, :] = col
        legend_pil = Image.fromarray(legend)
    draw = ImageDraw.Draw(legend_pil)
    for i, lbl in enumerate(labels):
        y = i * band_h + band_h // 2 - 5
        draw.text((4, y), lbl, fill="black")
    legend_np = np.array(legend_pil)

    result = np.hstack([blended, legend_np])
    Image.fromarray(result).save(save_path)


def build_warp_debug_panel(ref_pil, input_pil, ref_name, input_name,
                           depth_ref_path, transforms_path, forward_params,
                           crop_bbox, device='cuda', input_depth_path=None,
                           warp_method='backward', fg_mask=None,
                           occlusion_depth_path=None):
    """
    Warp debug: project pixels between input and ref, draw red grid on ref.
    fg_mask: (H_in, W_in) bool tensor — only show pixels passing this filter.
    Returns a PIL image the same size as ref_pil.
    """
    import torch
    from warp import load_master_depth
    import torch.nn.functional as Fnn

    ref_np = np.array(ref_pil.convert("RGB"))
    H_ref, W_ref = ref_pil.height, ref_pil.width
    H_in, W_in = input_pil.height, input_pil.width

    # Load camera params
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    W_json, H_json = float(data['w']), float(data['h'])
    K_base = torch.tensor([
        [float(data['fl_x']), 0, float(data['cx'])],
        [0, float(data['fl_y']), float(data['cy'])],
        [0, 0, 1]], device=device, dtype=torch.float32)

    Fix_S = torch.diag(torch.tensor([1., -1., -1., 1.], device=device))

    def get_pose(fname):
        for fr in data['frames']:
            if fr['file_path'] == fname:
                return torch.tensor(fr['transform_matrix'], device=device, dtype=torch.float32)
        raise ValueError(f"{fname} not found")

    c2w_ref = get_pose(ref_name) @ Fix_S
    c2w_in = get_pose(input_name) @ Fix_S
    if forward_params is not None:
        sc = torch.tensor(forward_params['scene_center'], device=device, dtype=torch.float32)
        ratio = forward_params['forward_step_ratio']
        t_in = c2w_in[:3, 3]
        fwd_dir = c2w_in[:3, 2]
        fwd_dir = fwd_dir / fwd_dir.norm()
        step = ratio * (t_in - sc).norm()
        c2w_in[:3, 3] = t_in + step * fwd_dir

    w2c_ref = torch.linalg.inv(c2w_ref)

    K_ref = K_base.clone()
    K_ref[0, :] *= W_ref / W_json
    K_ref[1, :] *= H_ref / H_json

    K_in = K_base.clone()
    K_in[0, :] *= W_in / W_json
    K_in[1, :] *= H_in / H_json

    # Backward: unproject input pixels using input depth, project into ref
    if warp_method == 'backward' and input_depth_path is not None and os.path.exists(input_depth_path):
        depth_in = load_master_depth(input_depth_path, device=device)
        if depth_in.shape[-2:] != (H_in, W_in):
            depth_in = Fnn.interpolate(depth_in, size=(H_in, W_in), mode='nearest')
        K_in_inv = torch.linalg.inv(K_in)

        y_in, x_in = torch.meshgrid(torch.arange(H_in, device=device), torch.arange(W_in, device=device), indexing='ij')
        grid_homo = torch.stack([x_in.flatten().float(), y_in.flatten().float(), torch.ones(H_in*W_in, device=device)], dim=0)
        d_flat = depth_in.view(1, -1)

        cam_pts = (K_in_inv @ grid_homo) * d_flat
        ones = torch.ones((1, cam_pts.shape[1]), device=device)
        world_pts = c2w_in @ torch.cat([cam_pts, ones], dim=0)

        cam_ref = w2c_ref @ world_pts
        z_ref = cam_ref[2:3, :]
        uv_ref = K_ref @ cam_ref[:3, :]
        u_ref = uv_ref[0] / (uv_ref[2] + 1e-6)
        v_ref = uv_ref[1] / (uv_ref[2] + 1e-6)

        valid = (z_ref.squeeze() > 0) & (u_ref >= 0) & (u_ref < W_ref) & (v_ref >= 0) & (v_ref < H_ref)
        _n_geom_valid = valid.sum().item()

        # Apply foreground filter
        if fg_mask is not None:
            valid = valid & fg_mask.flatten()
            _n_after_fg = valid.sum().item()
        else:
            _n_after_fg = _n_geom_valid

        # Apply occlusion filter (same as geometric crop in warp.py)
        if occlusion_depth_path is not None and os.path.exists(occlusion_depth_path):
            _occ_d = load_master_depth(occlusion_depth_path, device=device)
            if _occ_d.shape[-2:] != (H_ref, W_ref):
                _occ_d = Fnn.interpolate(_occ_d, size=(H_ref, W_ref), mode='nearest')
            _occ_ref = _occ_d.squeeze()
            _u_int = u_ref.round().long().clamp(0, W_ref - 1)
            _v_int = v_ref.round().long().clamp(0, H_ref - 1)
            _ref_z = _occ_ref[_v_int, _u_int]
            _proj_z = z_ref.squeeze()
            _not_occ = (_ref_z == 0) | (_proj_z <= _ref_z * 1.05)
            valid = valid & _not_occ
            _n_after_occ = valid.sum().item()
        else:
            _n_after_occ = _n_after_fg

        # Draw on ref image
        overlay = ref_np.copy()
        # Subsample for grid look
        step = max(1, min(H_in, W_in) // 80)
        grid_2d = np.zeros((H_in, W_in), dtype=bool)
        grid_2d[::step, :] = True
        grid_2d[:, ::step] = True
        grid_mask_flat = grid_2d.flatten()
        valid_np = valid.cpu().numpy()
        show_mask = valid_np & grid_mask_flat
        n_valid_total = valid_np.sum()
        n_dots = show_mask.sum()
        u_show = u_ref.cpu().numpy()[show_mask].astype(int)
        v_show = v_ref.cpu().numpy()[show_mask].astype(int)
        u_show = np.clip(u_show, 0, W_ref - 1)
        v_show = np.clip(v_show, 0, H_ref - 1)
        overlay[v_show, u_show] = [255, 0, 0]
        if n_dots == 0:
            print(f"  [warp_dbg] 0 dots drawn: geom_valid={_n_geom_valid}, after_fg={_n_after_fg}, after_occ={_n_after_occ}, H_in={H_in}, W_in={W_in}")
    else:
        # Fallback: forward warp (ref depth -> input check)
        depth_ref = load_master_depth(depth_ref_path, device=device)
        if depth_ref.shape[-2:] != (H_ref, W_ref):
            depth_ref = Fnn.interpolate(depth_ref, size=(H_ref, W_ref), mode='nearest')
        K_ref_inv = torch.linalg.inv(K_ref)

        y_r, x_r = torch.meshgrid(torch.arange(H_ref, device=device), torch.arange(W_ref, device=device), indexing='ij')
        grid_homo = torch.stack([x_r.flatten().float(), y_r.flatten().float(), torch.ones(H_ref*W_ref, device=device)], dim=0)
        d_flat = depth_ref.view(1, -1)
        cam_pts = (K_ref_inv @ grid_homo) * d_flat
        ones = torch.ones((1, cam_pts.shape[1]), device=device)
        world_pts = c2w_ref @ torch.cat([cam_pts, ones], dim=0)
        w2c_in = torch.linalg.inv(c2w_in)
        cam_in = w2c_in @ world_pts
        z_in = cam_in[2:3, :]
        uv_in = K_in @ cam_in[:3, :]
        u_in = uv_in[0] / (uv_in[2] + 1e-6)
        v_in = uv_in[1] / (uv_in[2] + 1e-6)
        in_view = (z_in.squeeze() > 0) & (u_in >= 0) & (u_in < W_in) & (v_in >= 0) & (v_in < H_in)

        # Apply foreground filter: only keep ref pixels that land on foreground input pixels
        if fg_mask is not None and in_view.any():
            dst_u_int = torch.round(u_in).long().clamp(0, W_in - 1)
            dst_v_int = torch.round(v_in).long().clamp(0, H_in - 1)
            fg_hit = fg_mask[dst_v_int, dst_u_int]
            in_view = in_view & fg_hit

        in_view_2d = in_view.view(H_ref, W_ref).cpu().numpy()
        overlay = ref_np.copy()
        step = max(1, min(H_ref, W_ref) // 80)
        grid_mask = np.zeros((H_ref, W_ref), dtype=bool)
        grid_mask[::step, :] = True
        grid_mask[:, ::step] = True
        overlay[in_view_2d & grid_mask] = [255, 0, 0]

    # Green bbox
    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)
    if crop_bbox is not None:
        draw.rectangle(crop_bbox, outline="lime", width=3)

    return result
