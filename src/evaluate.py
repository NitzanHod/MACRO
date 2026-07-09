"""
MACRO close-up evaluation driver.

Loads a pre-trained 3DGS checkpoint (stock gsplat format), renders each
close-up view + its depth from the Gaussians, runs the enhancement pass
(3dgs = none, difix = single-reference DiFix, macro = ours), and scores the
result against the ground-truth close-ups.

No training happens here — the checkpoint is produced upstream by the standard
gsplat trainer (see README).

Usage:
    python evaluate.py --scene-dir MobileClose-10/cactus --gpu 0 --configs macro
    python evaluate.py --scenes-dir <root of scene folders> --gpu 0 --configs 3dgs difix macro
"""

import copy
import json
import math
import os
import sys
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # parent of src/

# Compatibility: fix numpy 2.0 overflow in old pycolmap scene_manager.py
# np.uint64(-1) fails in numpy>=2.0; patch before importing.
import numpy as _np_compat
# The patch below temporarily shadows np.uint64 with a plain function. Several
# scipy/numpy C-extension submodules read np.uint64 as a *dtype* at their
# (lazy) import time and choke on the function ("Cannot interpret ... as a data
# type"). pycolmap pulls these in lazily, so force them to fully initialize
# BEFORE installing the patch. Guard each so a missing optional module is fine.
import numpy.random as _np_random_preload  # noqa: F401
_np_random_preload.SeedSequence(0).generate_state(1)
for _m in ("scipy.linalg", "scipy.sparse", "scipy.optimize", "scipy.spatial"):
    try:
        __import__(_m)
    except Exception:
        pass
_orig_uint64 = _np_compat.uint64
def _safe_uint64(val):
    if isinstance(val, int) and val < 0:
        return _orig_uint64(18446744073709551615)  # 2^64 - 1
    return _orig_uint64(val)
_np_compat.uint64 = _safe_uint64

try:
    from pycolmap import SceneManager
except ImportError:
    import pycolmap
    if hasattr(pycolmap, 'Reconstruction'):
        pycolmap.SceneManager = pycolmap.Reconstruction
    else:
        raise ImportError("pycolmap has neither SceneManager nor Reconstruction")
finally:
    _np_compat.uint64 = _orig_uint64  # restore

sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "gsplat"))

# HuggingFace's `datasets` package (often pulled in by diffusers/pyiqa) shadows
# our local `examples/gsplat/datasets/`. Register our local dir as a package
# under a unique name `_local_datasets` (no __init__.py — use ModuleType), then
# load colmap.py and normalize.py as submodules. This avoids the naming clash.
import importlib.util
import types
_pkg_path = os.path.join(REPO_ROOT, "examples", "gsplat", "datasets")

_local_datasets = types.ModuleType("_local_datasets")
_local_datasets.__path__ = [_pkg_path]
sys.modules["_local_datasets"] = _local_datasets

# Load dependent submodules first (colmap.py imports from .normalize etc.)
for _sub in ["normalize", "traj", "colmap"]:
    _src_path = os.path.join(_pkg_path, f"{_sub}.py")
    if os.path.exists(_src_path):
        _spec = importlib.util.spec_from_file_location(f"_local_datasets.{_sub}", _src_path)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[f"_local_datasets.{_sub}"] = _mod
        _spec.loader.exec_module(_mod)

Dataset = sys.modules["_local_datasets.colmap"].Dataset
Parser = sys.modules["_local_datasets.colmap"].Parser
from utils import knn, rgb_to_sh, set_random_seed
from gsplat.rendering import rasterization

sys.path.insert(0, SCRIPT_DIR)  # src/ directory
from difix_pipeline import DifixPipeline


def c2w_opencv_to_opengl(c2w):
    convert = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ convert


# -----------------------------------------------------------------------------
# Closeup-pose resolver that works for two scene layouts:
#   (1) closeup images are in the COLMAP sparse/0 model (and thus in
#       parser.image_names). The checkpoint was trained in the same frame.
#       We return parser.camtoworlds[idx] directly (already OpenCV, 4x4).
#   (2) closeup images were synthesized by forward-walking training cameras;
#       their poses live in the DL3DV/nerfstudio frame (OpenGL convention),
#       a pure rigid transform away from the COLMAP sparse/0 frame the ckpt
#       was trained in (plus an OpenCV/OpenGL axis flip). We fit a per-scene
#       similarity T_AB on the shared training cameras and return
#       `T_AB @ pose_gl @ diag(1,-1,-1,1)`.
# -----------------------------------------------------------------------------

_GL2CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _as4(m):
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (3, 4):
        m = np.vstack([m, [0, 0, 0, 1]])
    return m


def _umeyama_similarity(B, A, with_scale=True):
    """Find 4x4 similarity T such that A ≈ T @ B for point clouds."""
    muA = A.mean(0); muB = B.mean(0)
    A0 = A - muA; B0 = B - muB
    H = B0.T @ A0 / len(A)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / ((B0 ** 2).sum() / len(A)) if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = muA - s * R @ muB
    return T


_CLOSEUP_CACHE = {}


def build_closeup_pose_resolver(parser, split_info, scene_dir):
    """Return a function  resolve(fname) -> c2w_cv 4x4  for the given scene.

    Tries:
      (a) if fname is in parser.image_names, use parser.camtoworlds[idx]
      (b) else if DL3DV nerfstudio transforms.json exists alongside the scene's
          DL3DV backup or via scene['scene'] short-id, fit T_AB on shared
          training cams and apply T_AB @ pose_gl @ diag(1,-1,-1,1)
      (c) fallback to a raw OpenGL->OpenCV flip (pose_gl @ diag(1,-1,-1,1))
    """
    scene_key = os.path.abspath(scene_dir)
    if scene_key in _CLOSEUP_CACHE:
        return _CLOSEUP_CACHE[scene_key]

    parser_name_to_idx = {os.path.basename(n): i for i, n in enumerate(parser.image_names)}
    closeup_poses = split_info.get("closeup_poses", {})

    # Optionally fit T_AB using DL3DV nerfstudio transforms.json when a DL3DV
    # root is provided via the DL3DV_ROOT env var; otherwise fall back to the
    # poses in split.json (the standard path for the public release).
    T_AB = None
    short_id = str(split_info.get("scene", ""))
    dl3dv_root = Path(os.environ.get("DL3DV_ROOT", "")) if os.environ.get("DL3DV_ROOT") else None
    dl3dv_scene = None
    if short_id and dl3dv_root is not None and dl3dv_root.exists():
        for d in dl3dv_root.iterdir():
            if d.name.startswith(short_id):
                dl3dv_scene = d
                break
    if dl3dv_scene is not None:
        tj = dl3dv_scene / "nerfstudio" / "transforms.json"
        if not tj.exists():
            tj = dl3dv_scene / "transforms.json"
        if tj.exists():
            try:
                tdata = json.load(open(tj))
                B_by_name = {os.path.basename(f["file_path"]): _as4(f["transform_matrix"])
                             for f in tdata.get("frames", [])}
                A_poses = {n: _as4(parser.camtoworlds[i])
                           for n, i in parser_name_to_idx.items()}
                matched = [n for n in A_poses if n in B_by_name]
                if len(matched) >= 4:
                    A_c = np.stack([A_poses[n][:3, 3] for n in matched])
                    B_c = np.stack([B_by_name[n][:3, 3] for n in matched])
                    T_AB = _umeyama_similarity(B_c, A_c, with_scale=True)
                    # Sanity: max rotation residual after GL2CV flip
                    max_err = 0.0
                    for n in matched:
                        pred = T_AB @ B_by_name[n] @ _GL2CV_FLIP
                        max_err = max(max_err, float(np.linalg.norm(pred[:3, :3] - A_poses[n][:3, :3])))
                    if max_err > 1e-3:
                        print(f"  [{short_id}] WARNING: T_AB residual {max_err:.4g} — alignment may be off")
            except Exception as e:
                print(f"  [{short_id}] T_AB fit failed: {e}")

    def resolve(fname: str) -> np.ndarray:
        base = os.path.basename(fname)
        # (a) closeup image is in the parser (DS1 / DS3-v3 / anything with
        # closeups in COLMAP sparse/0). The parser holds (3,4) or (4,4).
        if base in parser_name_to_idx:
            c2w = _as4(parser.camtoworlds[parser_name_to_idx[base]])
            return c2w
        # (b) DS2-v3 / DL3DV-backed scenes with a fitted T_AB
        if fname in closeup_poses:
            pose_gl = _as4(closeup_poses[fname])
            if T_AB is not None:
                return T_AB @ pose_gl @ _GL2CV_FLIP
            # (c) fallback to the legacy behavior (raw GL→CV flip)
            return pose_gl @ _GL2CV_FLIP
        raise KeyError(f"No pose for closeup '{fname}' in parser or split.json")

    _CLOSEUP_CACHE[scene_key] = resolve
    return resolve


@torch.no_grad()
def render_depth(splats, camtoworlds, Ks, width, height, sh_degree=3, device="cuda:0"):
    """Render depth map from gsplat splats. Returns (H, W) numpy float32 array."""
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    # Use depth as "color" — 1 channel
    viewmats = torch.linalg.inv(camtoworlds)
    # Compute per-gaussian depth in camera space
    means_h = torch.cat([means, torch.ones(len(means), 1, device=means.device)], dim=-1)
    cam_means = (viewmats[0] @ means_h.T).T  # (N, 4)
    depths = cam_means[:, 2:3]  # (N, 1) z-depth
    # Render with depth as the color channel
    render_depths, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=depths.unsqueeze(0).expand(1, -1, -1),  # (1, N, 1)
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=None,  # raw colors, no SH
        rasterize_mode="classic",
    )
    depth_map = render_depths[0, ..., 0].cpu().numpy().astype(np.float32)
    # Zero out where alpha is low (no surface)
    alpha_map = render_alphas[0, ..., 0].cpu().numpy()
    depth_map[alpha_map < 0.5] = 0.0
    return depth_map


def build_transforms_json(parser, split_info, scene_dir):
    """
    Build a transforms.json compatible with our enhance.py pipeline
    from COLMAP parser data and split.json.
    """
    intrinsics = split_info["intrinsics"]
    # DS1/DS2-v3 use "training_frames", DS3-v3 uses "train_frames" — accept either.
    training_frames = split_info.get("training_frames") or split_info.get("train_frames") or []
    closeup_poses_gl = split_info.get("closeup_poses", {})

    frames = []

    # Training frames: poses from COLMAP parser (OpenCV) → convert to OpenGL
    for idx, img_name in enumerate(parser.image_names):
        c2w_cv = parser.camtoworlds[idx]  # (3, 4) or (4, 4) OpenCV
        if c2w_cv.shape[0] == 3:
            c2w_cv = np.vstack([c2w_cv, [0, 0, 0, 1]])
        c2w_gl = c2w_opencv_to_opengl(c2w_cv)
        frames.append({
            "file_path": img_name,
            "transform_matrix": c2w_gl.tolist(),
        })

    # Closeup frames: poses from split.json (already OpenGL)
    for fname, pose in closeup_poses_gl.items():
        # Check if already added (some closeup frames may also be training frames)
        existing = [f for f in frames if f["file_path"] == fname]
        if not existing:
            frames.append({
                "file_path": fname,
                "transform_matrix": pose if isinstance(pose, list) else np.array(pose).tolist(),
            })

    transforms = {
        "w": intrinsics["w"],
        "h": intrinsics["h"],
        "fl_x": intrinsics["fl_x"],
        "fl_y": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "frames": frames,
    }
    return transforms


@torch.no_grad()
def render_training_depths(splats, parser, output_dir, device="cuda:0"):
    """Render depth TIFFs for all training views. Returns the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    colmap_K = list(parser.Ks_dict.values())[0]
    colmap_w, colmap_h = list(parser.imsize_dict.values())[0]
    K_render = torch.from_numpy(colmap_K).float().to(device)

    for idx, img_name in enumerate(parser.image_names):
        c2w = parser.camtoworlds[idx]
        if c2w.shape[0] == 3:
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        c2w_t = torch.from_numpy(c2w).float().to(device).unsqueeze(0)
        Ks = K_render.unsqueeze(0)

        depth = render_depth(splats, c2w_t, Ks, colmap_w, colmap_h, device=device)
        depth_name = os.path.splitext(img_name)[0] + "_depth.tiff"
        imageio.imwrite(os.path.join(output_dir, depth_name), depth)
        # Also save the rendered RGB for reference
        renders, _, _ = rasterize_splats(splats, c2w_t, Ks, colmap_w, colmap_h)
        rgb = torch.clamp(renders[0, ..., :3], 0, 1)
        rgb_np = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, img_name), rgb_np)

    return output_dir


def create_splats_with_optimizers(
    parser, init_type="sfm", init_opacity=0.1, init_scale=1.0,
    scene_scale=1.0, sh_degree=3, batch_size=1, device="cuda",
):
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    else:
        raise ValueError("Only sfm init supported")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4 / 10 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3 / 5),
        ("quats", torch.nn.Parameter(quats), 1e-3 / 5),
        ("opacities", torch.nn.Parameter(opacities), 5e-2 / 5),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3 / 50),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20 / 50),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    BS = batch_size
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def rasterize_splats(splats, camtoworlds, Ks, width, height, sh_degree=3, **kwargs):
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        absgrad=True,
        rasterize_mode="classic",
        **kwargs,
    )
    return render_colors, render_alphas, info








# Config presets for each enhancer. These are the *canonical* configs used
# everywhere DiFix/macro are invoked in this script, ensuring consistency
# between per-round fixes and the final plus pass.
DIFIX_CONFIG = {
    "num_refs": 1,
    "num_bins": 1,
    "ref_selection": "greedy",
    "mask_mode": "none",
    "cross_attn_mode": "native",
    "ref_crop": False,
    "crop_strategy": "geometric",
}

MACRO_CONFIG = {
    "mask_mode": "all",
    "cross_attn_mode": "native",
    "ref_crop": True,
    "crop_method": "backward",
    "crop_layer": "bins",
    "num_bins": 3,
    "num_refs": 3,
    "ref_selection": "greedy",
    "crop_strategy": "geometric",
    # Use flex_attention kernel. Requires PyTorch 2.5+. Faster than 'split'
    # on the large (M×K) bin-crop attention pattern. May OOM at high M×K
    # counts — the ablation runner tracks a dynamic OOM threshold.
    "attention_mode": "flex",
    "hires_crop": True,
    "super_res": True,
    # Padding-aware ref mask: block attention to the black-padded regions
    # that appear when the crop bbox extends outside the ref image (common on
    # oblong aspect ratios like DS3's 1071x1428 iPhone captures). Routed
    # through the `occ_mask` machinery (compute_occlusion_mask now returns
    # an all-valid mask; the downstream crop step produces the padding-
    # invalid mask as a side-effect). See enhance.py.
    "occ_mask": True,
}









def load_baseline_checkpoint(scene_dir, splats, device="cuda:0"):
    """Load the pre-trained baseline 3DGS checkpoint."""
    ckpt_dir = os.path.join(scene_dir, "results", "ckpts")
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    if not ckpt_files:
        return None, 0
    ckpt_path = ckpt_files[-1]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    for k in splats.keys():
        if k in ckpt["splats"]:
            splats[k] = torch.nn.Parameter(ckpt["splats"][k].to(device))
    step = ckpt.get("step", 0)
    print(f"  Loaded baseline checkpoint: {ckpt_path} (step {step}, {len(splats['means'])} Gaussians)")
    return splats, step






@torch.no_grad()
def eval_closeup_plus(splats, split_info, parser, scene_dir, result_dir,
                      difix_pipe, enhancer="difix",
                      transforms_path=None, mvs_depth_folder=None,
                      train_depth_folder=None, hires_ref_folder=None,
                      device="cuda:0", sh_degree=3, metric_suite=None,
                      enhance_overrides=None, first_frame_only=False, pair_index=None):
    """
    Render each close-up from the baseline checkpoint and run one enhancement pass.

    Flow (same for both enhancers):
      1. Render RGB + depth for every valid closeup pose from the current splats.
      2. Build a merged transforms.json containing the novel closeup poses.
      3. Call main_video_pipeline once with DIFIX_CONFIG or MACRO_CONFIG
         (greedy ref selection for both, K=1 for difix, K=3 for macro).
      4. Load the enhanced outputs and compute PSNR/SSIM/LPIPS/DreamSim/DINOv2
         vs. the real GT (resized to GT native resolution).
    """
    from enhance import main_video_pipeline
    from metrics import MetricSuite, compute_all

    suite = metric_suite or MetricSuite(device=device)

    closeup_poses = split_info["closeup_poses"]
    closeup_pairs = split_info["closeup_eval_pairs"]
    colmap_K = list(parser.Ks_dict.values())[0]
    colmap_w, colmap_h = list(parser.imsize_dict.values())[0]
    K_render = torch.from_numpy(colmap_K).float().to(device)

    gt_dir = os.path.join(scene_dir, "closeup_gt")
    pred_dir = os.path.join(result_dir, "pred_plus")
    gt_out_dir = os.path.join(result_dir, "gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    # Filter to closeup pairs that have a pose AND a GT file on disk
    valid_pairs = []
    for i, pair in enumerate(closeup_pairs):
        fname = pair["closeup_frame"]
        if fname not in closeup_poses:
            continue
        if not os.path.exists(os.path.join(gt_dir, fname)):
            continue
        valid_pairs.append((i, pair))
    if not valid_pairs:
        return None
    config_name = f"{enhancer}_plus"
    # Optional: evaluate only the first closeup per scene (fast ablation mode
    # matching enhance.py's `skip_frames = set(test_frame_names[:1])`).
    if first_frame_only:
        valid_pairs = valid_pairs[:1]
    # Optional: evaluate only a specific pair index (targeted per-pair sweep).
    if pair_index is not None:
        if pair_index < 0 or pair_index >= len(valid_pairs):
            print(f"  [{config_name}] pair_index {pair_index} out of range [0, {len(valid_pairs)})")
            return None
        valid_pairs = [valid_pairs[pair_index]]

    # Choose config for this enhancer
    if enhancer == "macro":
        enhance_config = dict(MACRO_CONFIG)
    else:
        enhance_config = dict(DIFIX_CONFIG)
    if enhance_overrides:
        enhance_config.update(enhance_overrides)

    # 1. Render all closeup views (RGB + depth)
    render_dir = os.path.join(result_dir, "plus_renders")
    enhanced_dir = os.path.join(result_dir, "plus_enhanced")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    closeup_resolver = build_closeup_pose_resolver(parser, split_info, scene_dir)
    frame_names = []
    novel_poses_cv = []
    for i, pair in valid_pairs:
        closeup_fname = pair["closeup_frame"]
        pose_cv = closeup_resolver(closeup_fname)
        novel_poses_cv.append(pose_cv)
        camtoworld = torch.from_numpy(pose_cv).float().to(device).unsqueeze(0)
        Ks = K_render.unsqueeze(0)

        renders, _alphas, _ = rasterize_splats(
            splats, camtoworld, Ks, colmap_w, colmap_h,
            sh_degree=sh_degree, near_plane=0.01, far_plane=1e10,
        )
        pred = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
        pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
        frame_name = f"plus_{i:04d}.png"
        frame_names.append(frame_name)
        imageio.imwrite(os.path.join(render_dir, frame_name), pred_np)

        depth = render_depth(splats, camtoworld, Ks, colmap_w, colmap_h, device=device)
        depth_name = os.path.splitext(frame_name)[0] + "_depth.tiff"
        imageio.imwrite(os.path.join(render_dir, depth_name), depth)

    # 2. Merged transforms for the rendered closeups
    with open(transforms_path) as f:
        tf_data = json.load(f)
    merged_tf = dict(tf_data)
    merged_frames = list(tf_data["frames"])
    for slot, ((i, pair), pose_cv) in enumerate(zip(valid_pairs, novel_poses_cv)):
        c2w_gl = c2w_opencv_to_opengl(pose_cv)
        if c2w_gl.shape[0] == 3:
            c2w_gl = np.vstack([c2w_gl, [0, 0, 0, 1]])
        merged_frames.append({
            "file_path": frame_names[slot],
            "transform_matrix": c2w_gl.tolist(),
        })
    merged_tf["frames"] = merged_frames
    merged_path = os.path.join(result_dir, "_plus_merged_transforms.json")
    with open(merged_path, "w") as f:
        json.dump(merged_tf, f)

    # 3. Call main_video_pipeline on the rendered closeups.
    # REF_FOLDER stays pointing at the 1K folder so ref_images_pil (and therefore
    # model_size) stay at 1K; hires_ref_folder is passed separately and used
    # ONLY for the crop step when hires_crop=True.
    #
    # IMPORTANT: os.path.dirname(parser.image_paths[0]) points at the scene's
    # `images/` directory. For DS1 that's 1K (images_4). For DS2-v3 that's
    # 4K (images). We need 1K refs to keep model_size / attention within
    # memory budget. If hires_ref_folder was passed (DS2-v3 case), use its
    # sibling `images_4/` as the 1K ref folder.
    #
    # NOTE: For DS3, COLMAP registers BOTH training frames AND closeup frames
    # (because manual iPhone scenes need all 20 images to reconstruct). If we
    # feed parser.image_paths straight into train_frames_data, the greedy
    # reference selector can pick the closeup itself (its own depth covers
    # itself 100%). Filter using split_info["closeup_frames"] when present.
    _closeup_fname_set = set(split_info.get("closeup_frames", []))
    train_frame_names = [
        os.path.basename(p)
        for p in parser.image_paths
        if os.path.basename(p) not in _closeup_fname_set
    ]
    if _closeup_fname_set and len(train_frame_names) < len(parser.image_paths):
        print(f"  [ref-filter] Excluded {len(parser.image_paths) - len(train_frame_names)} "
              f"closeup frames from reference candidate pool")
    train_frames_data = {"train_frame_names": train_frame_names}
    ref_folder_base = os.path.dirname(parser.image_paths[0]) if parser.image_paths else ""
    if hires_ref_folder is not None:
        # hires_ref_folder points at the 4K `images/` dir under DL3DV.
        # Prefer its 1K sibling `images_4/` for model_size / attention, but
        # only if it's populated — some DL3DV scenes have an empty `images_4/`
        # directory, in which case we stay on the 4K `images/` (enhance.py
        # will clamp model_size downstream).
        _lowres_sibling = os.path.join(os.path.dirname(hires_ref_folder), "images_4")
        if os.path.isdir(_lowres_sibling):
            try:
                _has_files = any(
                    f.lower().endswith((".png", ".jpg", ".jpeg"))
                    for f in os.listdir(_lowres_sibling)
                )
            except OSError:
                _has_files = False
            if _has_files:
                ref_folder_base = _lowres_sibling
                print(f"  [REF_FOLDER] Using 1K sibling for attention: {_lowres_sibling}")
            else:
                print(f"  [REF_FOLDER] 1K sibling {_lowres_sibling} is empty — staying on 4K (model_size clamp will kick in)")
    viz_dir = os.path.join(result_dir, "plus_viz")

    def _run_enhance_pass(pass_image_folder, pass_save_dir, pass_config_name, pass_viz_dir):
        """Thin wrapper around main_video_pipeline for a single enhancement pass."""
        # Reuse the preloaded DifixPipeline for speed (difix / macro).
        pipe_to_use = difix_pipe
        try:
            main_video_pipeline(
                IMAGE_FOLDER=pass_image_folder,
                TRANSFORMS_FILE=merged_path,
                train_frames_data=train_frames_data,
                VIZ_FOLDER=pass_viz_dir,
                REF_FOLDER=ref_folder_base,
                DEPTH_FOLDER=train_depth_folder,
                config_name=pass_config_name,
                config_params=enhance_config,
                forward_params=None,
                forward_poses=None,
                coverage_data=None,
                mvs_depth_folder=mvs_depth_folder,
                device=device,
                skip=False,
                gt_folder=None,
                test_frame_filter=set(frame_names),
                pipe=pipe_to_use,
                save_outputs_dir=pass_save_dir,
                hires_ref_folder=hires_ref_folder,
            )
            return True
        except Exception as e:
            print(f"    [{pass_config_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    skip_enhance_flag = enhance_config.get("skip_enhance", False)

    if skip_enhance_flag:
        # --- Skip enhancement: use the raw 3DGS render as the "enhanced" output.
        # Used by the 3dgs no-enhancement baseline config. ---
        for fname in frame_names:
            src = os.path.join(render_dir, fname)
            dst = os.path.join(enhanced_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
    else:
        if not _run_enhance_pass(render_dir, enhanced_dir, config_name, viz_dir):
            return None

    # 4. Load enhanced outputs and compute all metrics at GT native resolution
    metrics = defaultdict(list)
    pair_results = []
    for (i, pair), frame_name in zip(valid_pairs, frame_names):
        closeup_fname = pair["closeup_frame"]
        training_fname = pair["training_frame"]

        enhanced_path = os.path.join(enhanced_dir, frame_name)
        if not os.path.exists(enhanced_path):
            print(f"    [{config_name}] Missing enhanced output: {frame_name}")
            continue

        gt_pil = Image.open(os.path.join(gt_dir, closeup_fname)).convert("RGB")
        enhanced_pil = Image.open(enhanced_path).convert("RGB")
        if enhanced_pil.size != gt_pil.size:
            enhanced_pil = enhanced_pil.resize(gt_pil.size, Image.LANCZOS)

        enhanced_pil.save(os.path.join(pred_dir, f"{i:04d}_{closeup_fname}"))
        gt_pil.save(os.path.join(gt_out_dir, f"{i:04d}_{closeup_fname}"))

        m = compute_all(enhanced_pil, gt_pil, suite)
        for k, v in m.items():
            metrics[k].append(v)

        pair_results.append({
            "training_frame": training_fname,
            "closeup_frame": closeup_fname,
            "depth_ratio": pair["depth_ratio"],
            "iou": pair["iou"],
            **{k: round(v, 4) for k, v in m.items()},
        })

    if not metrics["psnr"]:
        return None

    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    return {
        "scene": split_info["scene"],
        "num_pairs": len(pair_results),
        **{f"avg_{k}": round(v, 4) for k, v in avg.items()},
        "num_gaussians": len(splats["means"]),
        "per_pair": pair_results,
    }


def eval_scene(scene_dir, difix_pipe, device="cuda:0",
               force=False, enhancer="difix",
               dl3dv_root=None, results_base=None, metric_suite=None,
               enhance_overrides=None, result_tag_override=None,
               first_frame_only=False, pair_index=None,
               depth_ratio=None):
    import time as _time
    from metrics import MetricSuite
    scene_id = os.path.basename(scene_dir)
    split_path = os.path.join(scene_dir, "split.json")

    if not os.path.exists(split_path):
        print(f"  [{scene_id[:20]}] SKIP: no split.json")
        return None

    with open(split_path) as f:
        split_info = json.load(f)

    # Optional: restrict evaluation to a single closeup depth_ratio
    # (e.g. 0.5 = x2, 0.8 = x5, 0.9 = x10) instead of all closeup_eval_pairs.
    if depth_ratio is not None:
        _orig_pairs = split_info.get("closeup_eval_pairs", [])
        _kept = [p for p in _orig_pairs
                 if abs(float(p.get("depth_ratio", -1)) - float(depth_ratio)) < 1e-6]
        if not _kept:
            print(f"  [{scene_id[:20]}] SKIP: no pairs match depth_ratio={depth_ratio}")
            return None
        print(f"  [{scene_id[:20]}] depth_ratio filter: "
              f"{len(_orig_pairs)} -> {len(_kept)} pairs (r={depth_ratio})")
        split_info = dict(split_info)
        split_info["closeup_eval_pairs"] = _kept
        # Also restrict closeup_poses to only those pairs so downstream ref
        # pool / target pose resolution doesn't include other-ratio poses.
        _kept_fnames = {p["closeup_frame"] for p in _kept}
        if "closeup_poses" in split_info:
            split_info["closeup_poses"] = {
                k: v for k, v in split_info["closeup_poses"].items()
                if k in _kept_fnames
            }
        if "closeup_frames" in split_info:
            split_info["closeup_frames"] = [
                f for f in split_info.get("closeup_frames", [])
                if f in _kept_fnames
            ]

    # Check for COLMAP data
    if not os.path.exists(os.path.join(scene_dir, "sparse", "0", "cameras.bin")):
        print(f"  [{scene_id[:20]}] SKIP: no COLMAP data")
        return None

    # Result directory based on enhancer: render-from-baseline + enhance (no distillation).
    if enhancer == "macro":
        result_tag = "results_macro"
    else:
        result_tag = "results_difix"
    if result_tag_override:
        result_tag = result_tag_override
    if results_base:
        result_dir = os.path.join(results_base, scene_id[:8], result_tag)
    else:
        result_dir = os.path.join(scene_dir, result_tag)
    eval_path = os.path.join(result_dir, "closeup_eval.json")

    if not force and os.path.exists(eval_path):
        print(f"  [{scene_id[:20]}] Already evaluated ({result_tag}), skipping.")
        with open(eval_path) as f:
            return json.load(f)

    os.makedirs(result_dir, exist_ok=True)

    closeup_pairs = split_info["closeup_eval_pairs"]
    if not closeup_pairs:
        print(f"  [{scene_id[:20]}] No close-up pairs")
        return None

    parser = Parser(data_dir=scene_dir, factor=1, normalize=False, test_every=0)

    # Check for baseline checkpoint
    ckpt_dir = os.path.join(scene_dir, "results", "ckpts")
    if not glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")):
        print(f"  [{scene_id[:20]}] SKIP: no baseline checkpoint (run train_and_eval.py first)")
        return None

    print(f"  [{scene_id[:20]}] No-distill eval ({enhancer}) from baseline checkpoint...")

    # Prepare paths needed by main_video_pipeline (used by both difix and macro now
    # that DiFix is also routed through it for fair K=1 greedy ref selection).
    #
    # We always build transforms.json + train_depth_folder (required by the
    # enhance pipeline). mvs_depth_folder and hires_ref_folder are optional;
    # they're populated from dl3dv_root when DL3DV data exists (DS1, DS2-v3).
    # DS3 has neither a DL3DV equivalent nor 4K hires refs, so it passes
    # dl3dv_root=None and those stay None — the enhance pipeline degrades
    # gracefully (no MVS occlusion masking; hires_crop becomes an identity op).
    transforms_path = None
    mvs_depth_folder = None
    train_depth_folder = None
    hires_ref_folder = None

    if dl3dv_root:
        dl3dv_scene = os.path.join(dl3dv_root, scene_id, "nerfstudio")
        mvs_depth_folder = os.path.join(dl3dv_scene, "mvsanywhere_predictions")
        hires_ref_folder = os.path.join(dl3dv_scene, "images")

    # Always build transforms.json from Parser's COLMAP poses + split closeup
    # poses, regardless of whether DL3DV is present. Enhance pipeline needs it
    # to look up per-frame c2w for warping and ref selection.
    transforms_data = build_transforms_json(parser, split_info, scene_dir)
    transforms_path = os.path.join(result_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(transforms_data, f, indent=2)
    # Always render training view depths (gsplat-rendered). Used by ref selector
    # and for bin-crop backward warping.
    train_depth_folder = os.path.join(result_dir, "train_depth")
    if not os.path.exists(train_depth_folder) or force:
        splats_for_depth, _ = create_splats_with_optimizers(
            parser, init_type="sfm", init_opacity=0.1, init_scale=1.0,
            scene_scale=parser.scene_scale * 1.1, sh_degree=3, device=device,
        )
        splats_for_depth, _ = load_baseline_checkpoint(scene_dir, splats_for_depth, device)
        if splats_for_depth is not None:
            print(f"  [{scene_id[:20]}] Rendering training view depths...")
            render_training_depths(splats_for_depth, parser, train_depth_folder, device)
            del splats_for_depth
            torch.cuda.empty_cache()

    # No distillation — load baseline checkpoint, keep fixed
    splats, _optimizers = create_splats_with_optimizers(
        parser, init_type="sfm", init_opacity=0.1, init_scale=1.0,
        scene_scale=parser.scene_scale * 1.1, sh_degree=3, device=device,
    )
    splats, _ = load_baseline_checkpoint(scene_dir, splats, device)
    if splats is None:
        return None

    if metric_suite is None:
        metric_suite = MetricSuite(device=device)

    t_enhance_start = _time.time()

    # Render the close-ups from the baseline checkpoint and enhance them.
    print(f"  [{scene_id[:20]}] Running enhancement pass...")
    results = eval_closeup_plus(
        splats, split_info, parser, scene_dir, result_dir,
        difix_pipe=difix_pipe, enhancer=enhancer,
        transforms_path=transforms_path, mvs_depth_folder=mvs_depth_folder,
        train_depth_folder=train_depth_folder, hires_ref_folder=hires_ref_folder,
        device=device, metric_suite=metric_suite,
        enhance_overrides=enhance_overrides,
        first_frame_only=first_frame_only,
        pair_index=pair_index,
    )
    t_enhance_end = _time.time()

    if results:
        # Store timing info
        results["enhance_time_s"] = round(t_enhance_end - t_enhance_start, 2)
        results["total_time_s"] = results["enhance_time_s"]

        tag = enhancer.capitalize()
        print(f"  [{scene_id[:20]}] {tag} ({results['num_pairs']} pairs): "
              f"PSNR={results['avg_psnr']:.2f} SSIM={results['avg_ssim']:.4f} "
              f"LPIPS={results['avg_lpips']:.4f} "
              f"DreamSim={results.get('avg_dreamsim', 0):.4f} "
              f"DINOv2={results.get('avg_dinov2', 0):.4f} "
              f"total={results['total_time_s']:.1f}s")
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ==============================================================================
# Configuration definitions
# ==============================================================================

PUBLIC_CONFIGS = {
    # Public release config set. enhancer: which pipeline (difix=vanilla K=1,
    # macro=depth-plane cross-view attention K=3 M=3).
    "3dgs": {
        "enhancer": "difix",  # ignored when skip_enhance is on
        "enhance_overrides": {"skip_enhance": True},
        "result_tag_override": "results_3dgs",
    },
    "difix": {"enhancer": "difix"},
    "macro": {"enhancer": "macro"},
    "macro_unmasked": {  # K=3, M=3, mask_mode=none: ablation of the depth-aware mask
        "enhancer": "macro",
        "enhance_overrides": {"mask_mode": "none"},
        "result_tag_override": "results_macro_unmasked",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-dir", default=None,
                        help="Parent directory of scene folders (batch mode). Use --scene-dir for a single scene.")
    parser.add_argument("--scene-dir", default=None, help="Single scene to evaluate")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if results already exist")
    parser.add_argument("--dl3dv-root", default=None,
                        help="Optional path to DL3DV data (MVSAnywhere depth / 4K images) for DS1 "
                             "occlusion masking. Leave unset for DS3 or to use gsplat-rendered depth only.")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names to run (default: all — 3dgs difix macro macro_unmasked)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to save aggregate results (default: results/ in repo root)")
    parser.add_argument("--first-frame-only", action="store_true",
                        help="Evaluate only the first closeup per scene (fast ablation mode, "
                             "matches enhance.py's `skip_frames = set(test_frame_names[:1])`)")
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Evaluate only a specific closeup_eval_pairs index (0-based). "
                             "Use this for targeted per-pair sweeps. Mutually exclusive with --first-frame-only.")
    parser.add_argument("--depth-ratio", type=float, default=None,
                        help="Restrict evaluation to one closeup depth_ratio "
                             "(e.g. 0.5 / 0.8 / 0.9 for x2 / x5 / x10). When None, all ratios in "
                             "split.json['closeup_eval_pairs'] are evaluated.")
    parser.add_argument("--num-refs-override", type=int, default=None,
                        help="Override num_refs (K) in the enhance config. Used for Macro K×M sweep.")
    parser.add_argument("--num-bins-override", type=int, default=None,
                        help="Override num_bins (M) in the enhance config. Used for Macro K×M sweep.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    set_random_seed(42)

    # Select configs to run
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",")]
        configs = {k: PUBLIC_CONFIGS[k] for k in config_names if k in PUBLIC_CONFIGS}
    else:
        configs = PUBLIC_CONFIGS

    print(f"Configs to run: {list(configs.keys())}")

    # Discover scenes
    if args.scene_dir:
        scene_dirs = [args.scene_dir]
    else:
        scene_dirs = sorted([d.rstrip("/") for d in glob.glob(os.path.join(args.scenes_dir, "*/")) if os.path.isdir(d)])
    print(f"Scenes: {len(scene_dirs)}\n")

    print("Loading Difix pipeline (nvidia/difix_ref)...")
    difix_pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    difix_pipe.set_progress_bar_config(disable=True)
    difix_pipe.to(device)
    print("Pipeline loaded.\n")

    # Results: {config_name: [scene_result, ...]}
    all_config_results = {name: [] for name in configs}

    # Unify output root for per-scene dirs AND aggregate JSON so we never
    # fall back to scene_dir (which may be read-only external data).
    results_base = args.results_dir or os.path.join(REPO_ROOT, "results")
    os.makedirs(results_base, exist_ok=True)
    print(f"Results base: {results_base}\n")

    # Instantiate metric suite once and reuse across all configs/scenes.
    # Heavy models (DreamSim, DINOv2) are class-level cached so this also
    # avoids re-loading them for each invocation.
    from metrics import MetricSuite
    metric_suite = MetricSuite(device=device)

    for conf_name, conf_params in configs.items():
        enhancer = conf_params["enhancer"]

        print(f"\n{'#'*60}")
        print(f"CONFIG: {conf_name} (enhancer={enhancer})")
        print(f"{'#'*60}")

        for scene_dir in scene_dirs:
            # Merge CLI K/M overrides onto the per-config enhance_overrides
            # (used by the Macro sweep driver).
            base_ovr = dict(conf_params.get("enhance_overrides") or {})
            if args.num_refs_override is not None:
                base_ovr["num_refs"] = args.num_refs_override
            if args.num_bins_override is not None:
                base_ovr["num_bins"] = args.num_bins_override
            effective_ovr = base_ovr if base_ovr else None

            r = eval_scene(
                scene_dir, difix_pipe, device,
                force=args.force,
                enhancer=enhancer,
                dl3dv_root=args.dl3dv_root,
                results_base=results_base,
                metric_suite=metric_suite,
                enhance_overrides=effective_ovr,
                result_tag_override=conf_params.get("result_tag_override"),
                first_frame_only=args.first_frame_only,
                pair_index=args.pair_index,
                depth_ratio=args.depth_ratio,
            )
            if r:
                all_config_results[conf_name].append(r)

    # Print unified results table with all metrics and wall-time
    print(f"\n{'='*110}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*110}")
    print(f"{'Config':<22} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'DreamSim':>9} {'DINOv2':>8} "
          f"{'Time(s)':>9} {'Scenes':>7}")
    print(f"{'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*9} {'-'*7}")

    summary = {}
    for conf_name, results in all_config_results.items():
        if not results:
            print(f"{conf_name:<22} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>9} {'N/A':>8} "
                  f"{'N/A':>9} {0:>7}")
            continue
        avg = {}
        for key in ["psnr", "ssim", "lpips", "dreamsim", "dinov2"]:
            vals = [r.get(f"avg_{key}") for r in results if r.get(f"avg_{key}") is not None]
            avg[key] = float(np.mean(vals)) if vals else float("nan")
        time_vals = [r.get("total_time_s", 0.0) for r in results]
        avg["total_time_s"] = float(np.mean(time_vals)) if time_vals else 0.0
        summary[conf_name] = avg
        print(f"{conf_name:<22} {avg['psnr']:>7.2f} {avg['ssim']:>7.4f} {avg['lpips']:>7.4f} "
              f"{avg['dreamsim']:>9.4f} {avg['dinov2']:>8.4f} "
              f"{avg['total_time_s']:>9.1f} {len(results):>7}")
    print(f"{'='*110}")

    # Save aggregate JSON — merge with any existing aggregate so running one
    # config at a time doesn't wipe previous results for other configs.
    agg_path = os.path.join(results_base, "aggregate_results_progressive.json")
    existing = {"summary": {}, "per_config": {}}
    if os.path.exists(agg_path):
        try:
            with open(agg_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    # Merge in this invocation's results (overwriting only the configs we ran)
    for k, v in summary.items():
        existing.setdefault("summary", {})[k] = v
    for k, v in all_config_results.items():
        if v:
            existing.setdefault("per_config", {})[k] = v
    with open(agg_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {agg_path}")


if __name__ == "__main__":
    main()
print(1)