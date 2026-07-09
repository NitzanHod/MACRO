"""Depth-plane decomposition and confidence/occlusion masks for MACRO.

`compute_disparity_bins` clusters a rendered depth map into M fronto-parallel
planes (k-means in disparity space); the reference crops are matched per plane,
and each close-up token is routed (in attention.py) to the reference tokens of
its own plane. `compute_occlusion_mask` / `compute_confidence_mask` build the
per-plane validity masks that gate that depth-aware cross-view attention.
"""
import os
import json
import torch


def compute_disparity_bins(depth, M=5, device='cuda'):
    """
    Bin a depth map into M layers using k-means clustering in disparity (1/z) space.
    K-means finds natural depth layers (object boundaries become bin boundaries).

    Args:
        depth: (H, W) torch tensor, z-depth values (>0 for valid pixels)
        M: number of bins
        device: torch device

    Returns:
        bin_map: (H, W) long tensor, values 0..M-1 for valid pixels, -1 for invalid
                 Ordered: bin 0 = farthest (lowest disparity), bin M-1 = closest
        bin_masks: list of M bool tensors (H, W), one per bin
        centers: (M,) tensor of cluster centers in disparity space (sorted ascending)
    """
    bin_map, bin_masks, centers, _ = _kmeans_disparity(depth, M=M, device=device)
    return bin_map, bin_masks, centers


def _kmeans_disparity(depth, M, device='cuda'):
    """Shared 1D k-means over disparity. Returns (bin_map, bin_masks, centers, wcss)
    where wcss = within-cluster sum of squares in disparity space (scalar float).
    Raises RuntimeError if any bin ends up empty."""
    valid = depth > 0
    if not valid.any():
        raise RuntimeError("No valid depth pixels for disparity binning")

    disp = torch.zeros_like(depth)
    disp[valid] = 1.0 / depth[valid]

    valid_disp = disp[valid].float()

    # Quantile init (deterministic).
    quantiles = torch.linspace(0, 1, M + 2, device=device)[1:-1]
    centers = torch.quantile(valid_disp, quantiles)

    for _ in range(20):
        dists = (valid_disp.unsqueeze(1) - centers.unsqueeze(0)).abs()
        labels = dists.argmin(dim=1)
        new_centers = torch.zeros(M, device=device)
        for m in range(M):
            mask_m = labels == m
            if mask_m.any():
                new_centers[m] = valid_disp[mask_m].mean()
            else:
                new_centers[m] = centers[m]
        if (new_centers - centers).abs().max() < 1e-6:
            break
        centers = new_centers

    # Sort centers (bin 0 = farthest = lowest disparity)
    sort_idx = centers.argsort()
    centers = centers[sort_idx]

    # Final assignment
    dists = (valid_disp.unsqueeze(1) - centers.unsqueeze(0)).abs()
    final_labels = dists.argmin(dim=1)

    # WCSS in disparity space (scalar). Squared-distance to assigned center.
    wcss_vec = (valid_disp - centers[final_labels]).pow(2)
    wcss = float(wcss_vec.sum().item())

    # Build bin_map
    bin_map = torch.full_like(depth, -1, dtype=torch.long)
    valid_indices = torch.where(valid.flatten())[0]
    bin_map_flat = bin_map.flatten()
    bin_map_flat[valid_indices] = final_labels
    bin_map = bin_map_flat.view(depth.shape)

    bin_masks = [(bin_map == m) for m in range(M)]
    for m, bm in enumerate(bin_masks):
        if not bm.any():
            raise RuntimeError(
                f"Disparity bin {m} is empty (center={centers[m]:.4f}, "
                f"depth={1.0/centers[m]:.3f})"
            )
    return bin_map, bin_masks, centers, wcss


def compute_occlusion_mask(input_depth, ref_depth_path, input_frame_name, ref_name,
                           transforms_path, forward_params=None, device='cuda',
                           tolerance=1.05):
    """
    Return an all-valid mask at reference resolution.

    The only invalidity macro's attention needs is the black-padding region
    introduced when a crop bbox extends outside the ref image, and that region
    is already zeroed out by the downstream crop logic (`_mask_canvas` is
    initialized to zeros in the bin-crop block). This function therefore just
    returns an all-True mask; the `occ_mask` config flag remains the master
    switch for the "block padding from attention" behavior.

    Args:
        input_depth: (H_in, W_in) tensor  (unused, kept for signature compat)
        ref_depth_path: path to ref depth (used ONLY to read H_ref / W_ref)
        ...
    Returns:
        occ_mask: (H_ref, W_ref) bool tensor, all True
    """
    from warp import load_master_depth
    # We only need H_ref / W_ref to size the mask correctly. Load the depth
    # tiff just for its shape.
    ref_depth = load_master_depth(ref_depth_path, device=device)
    H_ref, W_ref = ref_depth.shape[-2], ref_depth.shape[-1]
    # All-valid: the padding-aware invalidity is added by the cropping step.
    return torch.ones(H_ref, W_ref, dtype=torch.bool, device=device)


def compute_confidence_mask(train_frame_names, depth_folder, transforms_path,
                            input_frame_name, input_size, forward_params=None,
                            device='cuda'):
    """
    Compute per-pixel confidence mask for the input (closeup) view.
    Forward-projects each training view's depth into the input viewpoint.
    Returns (H, W) int tensor with values 0..K_train — count of training views
    that cover each pixel.
    """
    from warp import load_master_depth
    import torch.nn.functional as Fnn

    W_in, H_in = input_size

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

    # Input camera
    c2w_in = get_pose(input_frame_name) @ Fix_S
    if forward_params is not None:
        sc = torch.tensor(forward_params['scene_center'], device=device, dtype=torch.float32)
        ratio = forward_params['forward_step_ratio']
        t_in = c2w_in[:3, 3]
        fwd_dir = c2w_in[:3, 2]
        fwd_dir = fwd_dir / fwd_dir.norm()
        step = ratio * (t_in - sc).norm()
        c2w_in[:3, 3] = t_in + step * fwd_dir
    w2c_in = torch.linalg.inv(c2w_in)

    K_in = K_base.clone()
    K_in[0, :] *= W_in / W_json
    K_in[1, :] *= H_in / H_json

    coverage = torch.zeros(H_in, W_in, dtype=torch.int32, device=device)

    for ref_name in train_frame_names:
        ref_basename = os.path.basename(ref_name)
        depth_base = os.path.splitext(ref_basename)[0]
        depth_path = os.path.join(depth_folder, f"{depth_base}_depth.tiff")
        if not os.path.exists(depth_path):
            continue

        ref_d = load_master_depth(depth_path, device=device)
        H_r, W_r = ref_d.shape[-2], ref_d.shape[-1]
        ref_d = ref_d.squeeze()

        c2w_ref = get_pose(ref_name) @ Fix_S
        K_ref = K_base.clone()
        K_ref[0, :] *= W_r / W_json
        K_ref[1, :] *= H_r / H_json
        K_ref_inv = torch.linalg.inv(K_ref)

        # Unproject ref pixels
        y_r, x_r = torch.meshgrid(torch.arange(H_r, device=device),
                                   torch.arange(W_r, device=device), indexing='ij')
        grid_homo = torch.stack([x_r.flatten().float(), y_r.flatten().float(),
                                 torch.ones(H_r * W_r, device=device)], dim=0)
        d_flat = ref_d.flatten().unsqueeze(0)

        # Filter valid depth
        valid_d = d_flat.squeeze() > 0
        cam_pts = (K_ref_inv @ grid_homo) * d_flat
        ones = torch.ones((1, cam_pts.shape[1]), device=device)
        world_pts = c2w_ref @ torch.cat([cam_pts, ones], dim=0)

        # Project into input camera
        cam_in = w2c_in @ world_pts
        z_in = cam_in[2, :]
        uv_in = K_in @ cam_in[:3, :]
        u_in = uv_in[0] / (uv_in[2] + 1e-6)
        v_in = uv_in[1] / (uv_in[2] + 1e-6)

        # Valid projections
        valid = valid_d & (z_in > 0) & \
                (u_in >= 0) & (u_in < W_in) & (v_in >= 0) & (v_in < H_in)

        u_int = u_in[valid].round().long().clamp(0, W_in - 1)
        v_int = v_in[valid].round().long().clamp(0, H_in - 1)

        # Mark covered pixels
        coverage[v_int, u_int] += 1

    return coverage
