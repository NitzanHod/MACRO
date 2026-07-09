"""
MACRO enhancement pipeline.

`main_video_pipeline` is the core enhancer called by evaluate.py for each
scene. Per close-up it: selects reference views (references.py), decomposes
the rendered depth into planes and crops scale-matched reference patches
(depth_planes.py + warp.py), super-resolves the crops (sr_worker.py), and runs
a single DiFix step (difix_pipeline.py) with depth-aware epipolar cross-view
attention (attention.py + epipolar.py) to paint reference detail into the
close-up. `3dgs`/`difix`/`macro` differ only by the config dict passed in.
Visualization helpers live in viz.py; small image utilities in imaging.py.
"""
import os
import math
import json
import time
import sys
import threading
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from diffusers.utils import load_image
from torchvision import transforms
from token_grid import build_S_to_HW
from difix_pipeline import DifixPipeline
from attention import EpipolarMixingBlock
from epipolar import get_epipolar_cache
from references import select_references
from warp import warp_reference_to_closeup
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from imaging import resize_pil, concat_images_with_labels
from depth_planes import (
    compute_disparity_bins,
    compute_occlusion_mask,
    compute_confidence_mask,
)
from viz import (
    build_coverage_panel,
    depth_to_jet_pil,
    load_depth_for_viz,
    save_confidence_viz,
    build_warp_debug_panel,
)

# ------------------------------------------------------------------------------
# Module-level runtime config (used by main_video_pipeline). These were part of
# the internal standalone runner; kept here for the public entry point
# (src/evaluate.py), which imports main_video_pipeline().
# ------------------------------------------------------------------------------
NUM_ENHANCE_PASSES = 1                          # DiFix passes per closeup (1 = single pass)
ENABLE_VIZ = bool(int(os.environ.get("ENABLE_VIZ", "0")))  # save per-frame ECA viz dashboards
SR_GPU_ID = os.environ.get("SR_GPU_ID", "1")               # GPU dedicated to PFT-SR subprocess

# ==============================================================================
# 1. Utility Functions
# ==============================================================================


REF_COLORS = [
    (255, 0, 0),      # red - ref0
    (0, 200, 0),      # green - ref1
    (0, 100, 255),    # blue - ref2
    (255, 200, 0),    # yellow - ref3
    (255, 0, 255),    # magenta - ref4
    (0, 255, 255),    # cyan - ref5
]


def frame_to_render_name(frame_name):
    """Convert frame_00040.png -> val_step29999_0039.png (1-indexed to 0-indexed)."""
    base = os.path.splitext(frame_name)[0]
    frame_idx = int(base.split('_')[-1])
    return f"val_step29999_{(frame_idx - 1):04d}"


def compute_image_metrics(out_pil, gt_pil, metric_fns, device='cuda'):
    """Compute PSNR, SSIM, LPIPS between two PIL images. Returns dict of float values."""
    out_t = TF.to_tensor(out_pil).unsqueeze(0).to(device)  # [1, 3, H, W] in [0,1]
    gt_t = TF.to_tensor(gt_pil).unsqueeze(0).to(device)
    psnr_val = metric_fns['psnr'](out_t, gt_t).item()
    ssim_val = metric_fns['ssim'](out_t, gt_t).item()
    lpips_val = 0.0
    if metric_fns.get('lpips') is not None:
        with torch.no_grad():
            lpips_val = metric_fns['lpips'](out_t.cpu() * 2 - 1, gt_t.cpu() * 2 - 1).item()
    metric_fns['psnr'].reset()
    metric_fns['ssim'].reset()
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val,
    }


# ==============================================================================
# 2. Main Inference Logic
# ==============================================================================

def main_video_pipeline(
    IMAGE_FOLDER, TRANSFORMS_FILE, train_frames_data, VIZ_FOLDER,
    REF_FOLDER, DEPTH_FOLDER,
    config_name, config_params,
    forward_params=None,
    forward_poses=None,
    coverage_data=None,
    mvs_depth_folder=None,
    device="cuda", skip=False,
    gt_folder=None,
    test_frame_filter=None,
    pipe=None,
    save_outputs_dir=None,
    hires_ref_folder=None,
):
    K = config_params.get("num_refs", 4)
    use_gt = config_params.get("use_gt", False)
    mask_mode = config_params.get("mask_mode", "all")
    cross_attn_mode = config_params.get("cross_attn_mode", "native")
    ref_crop = config_params.get("ref_crop", False)
    ref_warp = config_params.get("ref_warp", False)  # if True, synthesize ref via backward warp (mode='warp') instead of cropping; forces hires_crop/SR off downstream
    depth_source = config_params.get("depth_source", "gsplat")  # 'gsplat' or 'mvs'
    crop_method = config_params.get("crop_method", "backward")  # 'backward' or 'forward'
    crop_layer = config_params.get("crop_layer", "fg")  # 'fg', 'bg', or 'bins'
    ref_selection = config_params.get("ref_selection", "greedy")  # 'coverage' (JSON) or 'greedy' (depth-based)
    gt_ref_swap = (ref_selection == 'gt')  # swap ref tensors with GT at enhancement time
    if ref_selection == 'gt' and not ref_crop:
        ref_crop = False  # GT refs without bins — skip cropping
    if ref_selection == 'gt' and ref_crop:
        # Bins + GT: run full pipeline with greedy refs, swap tensors later
        ref_selection = 'greedy'
    occ_mask_enabled = config_params.get("occ_mask", False)  # enable occlusion attention masking
    conf_mask_enabled = config_params.get("conf_mask", False)  # enable confidence masking (block uncovered input tokens)
    enhance_scale = config_params.get("enhance_scale", 1.0)  # scale factor for inference resolution
    use_super_res = config_params.get("super_res", False)  # use PFT-SR for crop upscaling
    warp_patch = config_params.get("warp_patch", 32)  # warp masking patch size in pixels (full res)
    warp_forced = config_params.get("warp_forced", False)  # teacher-force warp locations (identity for GT refs)
    input_local_patch = config_params.get("input_local_patch", 0)  # restrict input self-attention to local patch (0=disabled)
    ref_boost = config_params.get("ref_boost", 0.0)  # additive bias to ref columns for input queries (0=disabled)
    block_ref_to_input = config_params.get("block_ref_to_input", False)  # ablation: block ref→input attention (Rule 4)
    attention_mode = config_params.get("attention_mode", "full")  # 'full' (current) or 'split' (per-bin ref SA + masked input queries)
    num_enhance_passes = config_params.get("num_enhance_passes", NUM_ENHANCE_PASSES)  # per-config override for enhancement passes
    flat_occ_masks = None  # initialized here; populated during bin flattening if bins + occ_mask
    flat_warp_maps = None  # initialized here; populated during bin flattening for warp masking
    skip_enhance = config_params.get("skip_enhance", False)
    enhancer = config_params.get("enhancer", "difix")  # 'difix' or 'macro' (both use the DiFix pipeline)
    # do_viz can be overridden per-run via the ENABLE_VIZ env var (truthy),
    # via config_params["enable_viz"], or defaults to the module-level ENABLE_VIZ.
    _env_viz = os.environ.get("ENABLE_VIZ", "").lower() in ("1", "true", "yes", "on")
    do_viz = bool(config_params.get("enable_viz", ENABLE_VIZ or _env_viz))

    print(f"\n>>> Running Pipeline: {config_name} (K={K}, gt={use_gt}, mask={mask_mode}, ca={cross_attn_mode}, crop={ref_crop}{' (WARP)' if ref_warp else ''}, attn={attention_mode}, T={num_enhance_passes}, enhancer={enhancer})")
    print(f"    VIZ Folder: {VIZ_FOLDER}")

    # 1. Setup Pipeline (reuse if provided; skip for baseline)
    if not skip_enhance:
        if pipe is None:
            pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
            pipe.to(device)
        pipe.no_ref = False
    os.makedirs(VIZ_FOLDER, exist_ok=True)

    # Load PFT-SR model for super-resolution crop upscaling
    sr_model = None
    if use_super_res and not ref_warp:
        sr_model = True  # flag — actual SR runs in subprocess via sr_worker.py
        print(f"  [SR] PFT-SR enabled (subprocess mode)")
    elif use_super_res and ref_warp:
        print(f"  [SR] PFT-SR disabled (ref_warp overrides super_res — warp output is full-frame)")

    # Per-frame diagnostic metrics (printed only; the scored metrics come from
    # the eval driver's MetricSuite). LPIPS runs on CPU.
    import lpips as lpips_pkg
    _lpips_fn = lpips_pkg.LPIPS(net='alex')
    _lpips_fn.eval()
    metric_fns = {
        'psnr': PeakSignalNoiseRatio(data_range=1.0).to(device),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        'lpips': _lpips_fn,
    }
    frame_metrics = []  # list of per-frame metric dicts

    # 2. Load Metadata
    with open(TRANSFORMS_FILE, 'r') as f:
        meta = json.load(f)

    frame_lookup = {}
    for fr in meta['frames']:
        fname = os.path.basename(fr['file_path'])
        frame_lookup[fname] = fr

    # Override input poses with pre-computed forwarded poses (depth-based forwarding)
    # forward_transforms.json poses are in OpenCV convention (Fix_S already applied)
    # We store them with a marker so downstream code skips Fix_S
    if forward_poses is not None:
        for fname, fwd_matrix in forward_poses.items():
            if fname in frame_lookup:
                frame_lookup[fname] = dict(frame_lookup[fname])
                frame_lookup[fname]['transform_matrix'] = fwd_matrix
        forward_params = None
        # Write a merged transforms file so warp.py/epipolar.py/etc. use forwarded poses
        _merged_tf = dict(meta)
        _merged_frames = []
        for fr in meta['frames']:
            fn = os.path.basename(fr['file_path'])
            if fn in forward_poses:
                fr_copy = dict(fr)
                fr_copy['transform_matrix'] = forward_poses[fn]
                _merged_frames.append(fr_copy)
            else:
                _merged_frames.append(fr)
        _merged_tf['frames'] = _merged_frames
        _merged_path = os.path.join(VIZ_FOLDER, '_merged_transforms.json')
        with open(_merged_path, 'w') as f:
            json.dump(_merged_tf, f)
        TRANSFORMS_FILE = _merged_path
        print(f"  [fwd] Using {len(forward_poses)} pre-computed forwarded poses, merged transforms at {_merged_path}")

    train_frame_names_set = set(train_frames_data['train_frame_names'])
    # Full list of train frame file_path values (for reference selection)
    train_frame_names_list = []
    for tfn in train_frames_data['train_frame_names']:
        if tfn in frame_lookup:
            train_frame_names_list.append(frame_lookup[tfn]['file_path'])

    test_frame_names = sorted([fn for fn in frame_lookup.keys() if fn not in train_frame_names_set])

    # Optional filter: only process specific test frames (e.g. closeup frames in disparity mode)
    if test_frame_filter is not None:
        test_frame_names = [fn for fn in test_frame_names if fn in test_frame_filter]
        # print(f"  [filter] {len(test_frame_names)} test frames after filter")

    # --- Depth round-trip sanity check (Test A) ---
    if ref_crop and train_frames_data['train_frame_names']:
        _test_ref = train_frames_data['train_frame_names'][0]
        _test_fp = frame_lookup.get(_test_ref, {}).get('file_path')
        if _test_fp:
            _depth_base = os.path.splitext(_test_ref)[0]
            if depth_source == 'mvs' and mvs_depth_folder is not None:
                _dpath = os.path.join(mvs_depth_folder, _test_ref.replace('.png', '.npz'))
            else:
                _dpath = os.path.join(DEPTH_FOLDER, f"{_depth_base}_depth.tiff")
            if os.path.exists(_dpath):
                from warp import load_master_depth
                import torch.nn.functional as Fnn
                _W_json, _H_json = float(meta['w']), float(meta['h'])
                _K_base = torch.tensor([
                    [float(meta['fl_x']), 0, float(meta['cx'])],
                    [0, float(meta['fl_y']), float(meta['cy'])],
                    [0, 0, 1]], device=device, dtype=torch.float32)
                _Fix_S = torch.diag(torch.tensor([1., -1., -1., 1.], device=device))
                _c2w = torch.tensor(frame_lookup[_test_ref]['transform_matrix'], device=device, dtype=torch.float32) @ _Fix_S
                _w2c = torch.linalg.inv(_c2w)
                _d = load_master_depth(_dpath, device=device)
                _ref_img = Image.open(os.path.join(REF_FOLDER, _test_ref))
                _W_ref, _H_ref = _ref_img.size
                if _d.shape[-2:] != (_H_ref, _W_ref):
                    _d = Fnn.interpolate(_d, size=(_H_ref, _W_ref), mode='nearest')
                _K_ref = _K_base.clone()
                _K_ref[0, :] *= _W_ref / _W_json
                _K_ref[1, :] *= _H_ref / _H_json
                _K_inv = torch.linalg.inv(_K_ref)
                # Test 5 pixels: center + 4 quadrants
                _test_pts = [(int(_W_ref*x), int(_H_ref*y)) for x, y in
                             [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]]
                # print(f"  [depth round-trip] ref={_test_ref} img={_W_ref}x{_H_ref} depth={_d.shape[-1]}x{_d.shape[-2]}")
                for _px, _py in _test_pts:
                    _z = _d[0, 0, _py, _px].item()
                    _uv1 = torch.tensor([_px, _py, 1.0], device=device, dtype=torch.float32)
                    _cam = (_K_inv @ _uv1) * _z
                    _world = _c2w @ torch.cat([_cam, torch.ones(1, device=device)])
                    _cam2 = _w2c @ _world
                    _proj = _K_ref @ _cam2[:3]
                    _u2 = (_proj[0] / (_proj[2] + 1e-8)).item()
                    _v2 = (_proj[1] / (_proj[2] + 1e-8)).item()
                    # print(f"    ({_px},{_py}) z={_z:.3f} -> ({_u2:.1f},{_v2:.1f}) err=({_u2-_px:.2f},{_v2-_py:.2f})")

    # Buffer State
    frame_buffer = []
    depth_buffer = []
    warp_debug_buffer = []

    # 3. Processing loop
    skip_frames = set(test_frame_names[:1])  # first test frame only (used when skip=True)

    # Apply skip filter to get actual frames to process
    if skip:
        frames_to_process = [f for f in test_frame_names if f in skip_frames]
    else:
        frames_to_process = list(test_frame_names)

    for test_fname in tqdm(frames_to_process, desc=f"Config {config_name}"):

        input_frame_name = frame_lookup[test_fname]['file_path']

        # --- Load Input ---
        _t0 = time.time()
        if use_gt:
            gt_path = os.path.join(REF_FOLDER, test_fname)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"GT image not found: {gt_path}")
            input_full = load_image(gt_path).convert("RGB")
        else:
            # Try direct frame name first (new renders), fall back to old naming
            full_path = os.path.join(IMAGE_FOLDER, test_fname)
            if not os.path.exists(full_path):
                render_base = frame_to_render_name(test_fname)
                full_path = os.path.join(IMAGE_FOLDER, f"{render_base}.png")
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Render not found: {full_path}")
            input_full = load_image(full_path).convert("RGB")

        # --- Load GT for display ---
        _gt_dir = gt_folder if gt_folder is not None else REF_FOLDER
        gt_path = os.path.join(_gt_dir, test_fname)
        _gt_exists = os.path.exists(gt_path)
        gt_full = load_image(gt_path).convert("RGB") if _gt_exists else input_full

        # --- Baseline mode: skip enhancement, use input as output ---
        if skip_enhance:
            _t8 = time.time()
            out = input_full.copy()
            if _gt_exists:
                out_for_metrics = out.resize(gt_full.size, Image.LANCZOS)
                m = compute_image_metrics(out_for_metrics, gt_full, metric_fns, device)
                frame_metrics.append(m)
                _t9 = time.time()
                print(f"  [METRICS] {test_fname}: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} (baseline)")
            else:
                _t9 = time.time()
            if save_outputs_dir is not None:
                os.makedirs(save_outputs_dir, exist_ok=True)
                out.save(os.path.join(save_outputs_dir, test_fname))
            if not do_viz:
                continue
            # Viz for baseline
            model_size = (input_full.size[0] // 8 * 8, input_full.size[1] // 8 * 8)
            input_disp = resize_pil(input_full, model_size)
            gt_disp = resize_pil(gt_full, model_size)
            out_disp = resize_pil(out, model_size)
            compact_pil = concat_images_with_labels(
                images=[input_disp, out_disp, gt_disp],
                labels=["Input", "Output", "GT"],
                path="", colors=["red", "red", "red"],
            )
            compact_bgr = cv2.cvtColor(np.array(compact_pil), cv2.COLOR_RGB2BGR)
            frame_buffer.append(compact_bgr)
            if not hasattr(main_video_pipeline, '_compact_buffer'):
                main_video_pipeline._compact_buffer = []
            main_video_pipeline._compact_buffer.append(compact_bgr)
            continue

        # --- Select K references ---
        _t1 = time.time()
        coverage_maps_selected = None
        ref_coverages_selected = None
        if use_gt:
            selected_ref_names = [input_frame_name] * K
        elif ref_selection == 'gt':
            # Use GT image as reference K times (oracle test)
            _gt_dir = gt_folder if gt_folder is not None else REF_FOLDER
            _gt_basename = os.path.basename(input_frame_name)
            _gt_ref_path = os.path.join(_gt_dir, _gt_basename)
            if not os.path.exists(_gt_ref_path):
                raise FileNotFoundError(f"GT ref not found: {_gt_ref_path}")
            selected_ref_names = [input_frame_name] * K
            print(f"  [gt] Using GT image as reference: {_gt_ref_path}")
        elif ref_selection == 'align':
            # Select K refs whose viewing direction most aligns with input
            Fix_S = np.diag([1., -1., -1., 1.])
            input_c2w = np.array(frame_lookup[test_fname]['transform_matrix']) @ Fix_S
            input_dir = input_c2w[:3, 2]  # optical axis (OpenCV z-forward after Fix_S)
            input_dir = input_dir / (np.linalg.norm(input_dir) + 1e-8)
            scored = []
            for tfp in train_frame_names_list:
                tb = os.path.basename(tfp)
                if tb not in frame_lookup:
                    continue
                ref_c2w = np.array(frame_lookup[tb]['transform_matrix']) @ Fix_S
                ref_dir = ref_c2w[:3, 2]
                ref_dir = ref_dir / (np.linalg.norm(ref_dir) + 1e-8)
                dot = float(np.dot(input_dir, ref_dir))
                scored.append((tfp, dot))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_ref_names = [s[0] for s in scored[:K]]
            dots = [s[1] for s in scored[:K]]
            print(f"  [align] Selected {K} refs by viewing direction alignment: dots={[f'{d:.3f}' for d in dots]}")
        elif ref_selection == 'greedy':
            # Depth-based greedy set cover (occlusion-aware). When the user has
            # supplied an oracle depth folder, use it for ref coverage too.
            _oracle_dir = config_params.get("oracle_depth_folder")
            _effective_depth_folder = _oracle_dir if _oracle_dir else DEPTH_FOLDER
            selected_ref_names, coverage_maps_selected, coverage = select_references(
                input_frame_name=input_frame_name,
                train_frame_names=train_frame_names_list,
                frame_lookup=frame_lookup,
                transforms_path=TRANSFORMS_FILE,
                depth_folder=_effective_depth_folder,
                depth_mode='master',
                forward_params=forward_params,
                K=K,
                device=device,
                mvs_depth_folder=mvs_depth_folder,
                depth_source=depth_source,
            )
        else:
            selected_ref_names, coverage_maps_selected, coverage = select_references(
                input_frame_name=input_frame_name,
                train_frame_names=train_frame_names_list,
                frame_lookup=frame_lookup,
                transforms_path=TRANSFORMS_FILE,
                depth_folder=DEPTH_FOLDER,
                depth_mode='master',
                forward_params=forward_params,
                K=K,
                device=device,
            )

        # --- Load K reference images ---
        _t2 = time.time()
        ref_images_pil = []
        for ref_name in selected_ref_names:
            ref_basename = os.path.basename(ref_name)
            if ref_selection == 'gt' and gt_folder is not None:
                ref_path = os.path.join(gt_folder, ref_basename)
            else:
                ref_path = os.path.join(REF_FOLDER, ref_basename)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference image not found: {ref_path}")
            ref_images_pil.append(load_image(ref_path).convert("RGB"))

        # --- Model Size ---
        _t3 = time.time()
        down_factor = 1
        ref_w = (ref_images_pil[0].size[0] // down_factor) // 8 * 8
        ref_h = (ref_images_pil[0].size[1] // down_factor) // 8 * 8
        # Clamp model_size to ~1K on the larger side. At ref resolutions above
        # ~1000px per side, both (a) PFT-SR subprocess and (b) main attention
        # OOM with tens of GiB allocations. DS1 runs at 960x540 (1K); DS2-v3
        # runs at 960x540 (1K sibling patched in evaluate);
        # DS3 has intrinsic ~1071x1428 which is too tall — clamp here.
        _MAX_MODEL_SIDE = 960
        _mx = max(ref_w, ref_h)
        if _mx > _MAX_MODEL_SIDE:
            _scale = _MAX_MODEL_SIDE / _mx
            ref_w = (int(round(ref_w * _scale)) // 8) * 8
            ref_h = (int(round(ref_h * _scale)) // 8) * 8
            print(f"  [model_size] Clamped from {ref_images_pil[0].size} to ({ref_w}, {ref_h}) (max={_MAX_MODEL_SIDE})")
        ref_size = (ref_w, ref_h)
        model_size = ref_size  # input resized to match ref resolution

        # --- Confidence mask: count training view coverage per input pixel ---
        confidence_mask = compute_confidence_mask(
            train_frame_names=train_frame_names_list,
            depth_folder=DEPTH_FOLDER,
            transforms_path=TRANSFORMS_FILE,
            input_frame_name=input_frame_name,
            input_size=input_full.size,
            forward_params=forward_params,
            device=device,
        )
        n_uncovered = (confidence_mask == 0).sum().item()
        n_total_px = confidence_mask.numel()
        print(f"  [confidence] {n_uncovered}/{n_total_px} uncovered ({n_uncovered/n_total_px*100:.1f}%), max={confidence_mask.max().item()}")
        if do_viz:
            save_confidence_viz(confidence_mask, input_full,
                os.path.join(VIZ_FOLDER, f"confidence_{os.path.splitext(test_fname)[0]}.jpg"),
                n_train=len(train_frame_names_list))

        # --- Crop references if ref_crop enabled ---
        ref_crop_uvs = [None] * K  # normalized [0,1] crop bboxes, None if no crop
        input_depth_path = None
        crop_method_frame = crop_method
        ref_eliminated = [False] * K
        bin_map = None  # (H, W) int tensor for disparity bins, set if crop_layer='bins'
        bin_crops = None  # M x K structure for multi-bin mode
        fg_mask = None  # foreground mask for single-mask crop or warp debug

        if ref_crop and not use_gt:
            # Resolve input depth for backward warping and depth filtering.
            # Oracle override: when config_params["oracle_depth_folder"] is set,
            # prefer <oracle>/<stem>_depth.tiff. Falls back to the gsplat
            # render's depth in IMAGE_FOLDER when no oracle exists.
            input_depth_base = os.path.splitext(test_fname)[0]
            oracle_dir = config_params.get("oracle_depth_folder")

            def _ref_depth_path_oracle_aware(depth_base):
                """Return <oracle>/<base>_depth.tiff if it exists, else the
                default DEPTH_FOLDER path. Used for per-ref depth lookups
                in the macro path below."""
                if oracle_dir:
                    cand = os.path.join(oracle_dir, f"{depth_base}_depth.tiff")
                    if os.path.exists(cand):
                        return cand
                return os.path.join(DEPTH_FOLDER, f"{depth_base}_depth.tiff")

            if oracle_dir:
                oracle_path = os.path.join(oracle_dir, f"{input_depth_base}_depth.tiff")
                if os.path.exists(oracle_path):
                    input_depth_path = oracle_path
                    print(f"  [oracle-depth] using {oracle_path}")
                else:
                    input_depth_path = os.path.join(IMAGE_FOLDER, f"{input_depth_base}_depth.tiff")
                    print(f"  [oracle-depth] WARN no oracle for {input_depth_base}, falling back to gsplat")
            else:
                input_depth_path = os.path.join(IMAGE_FOLDER, f"{input_depth_base}_depth.tiff")
            if crop_method == 'backward' and not os.path.exists(input_depth_path):
                print(f"  [crop] Input depth not found: {input_depth_path}, falling back to forward crop")
                crop_method_frame = 'forward'
            else:
                crop_method_frame = crop_method

            # Load input depth
            _input_depth_tensor = None
            if os.path.exists(input_depth_path):
                from warp import load_master_depth
                import torch.nn.functional as Fnn
                _input_depth_tensor = load_master_depth(input_depth_path, device=device)
                _H_in, _W_in = input_full.size[1], input_full.size[0]
                if _input_depth_tensor.shape[-2:] != (_H_in, _W_in):
                    _input_depth_tensor = Fnn.interpolate(_input_depth_tensor, size=(_H_in, _W_in), mode='nearest')

            if crop_layer == 'bins' and _input_depth_tensor is not None:
                # --- Multi-bin disparity crop ---
                _d = _input_depth_tensor.squeeze()
                M = config_params.get("num_bins", 5)
                bin_map, bin_masks, bin_centers = compute_disparity_bins(_d, M=M, device=device)
                # print(f"  [bins] M={M} centers (disp): {[f'{c:.4f}' for c in bin_centers.tolist()]}")
                for m in range(M):
                    pct = bin_masks[m].float().mean().item() * 100
                    d_center = 1.0 / bin_centers[m] if bin_centers[m] > 0 else float('inf')
                    # print(f"    bin{m}: {pct:.1f}% pixels, depth~{d_center:.2f}")

                # Per-bin, per-ref cropping
                bin_crops = [[None] * K for _ in range(M)]
                bin_ref_eliminated = [[False] * K for _ in range(M)]
                bin_occ_masks = [[None] * K for _ in range(M)]  # cropped occlusion masks per bin×ref
                bin_warp_maps = [[None] * K for _ in range(M)]  # per-input-pixel warp coords in cropped ref space
                bin_sr_viz = [[None] * K for _ in range(M)]  # (SR_pil, LANCZOS_pil) per bin×ref for viz
                ref_occlusion_masks = [None] * K  # per-ref occlusion masks at ref resolution

                for r_idx in range(K):
                    ref_name = selected_ref_names[r_idx]
                    ref_pil = ref_images_pil[r_idx]
                    ref_basename = os.path.basename(ref_name)
                    depth_base = os.path.splitext(ref_basename)[0]

                    if depth_source == 'mvs' and mvs_depth_folder is not None:
                        depth_ref_path = os.path.join(mvs_depth_folder, ref_basename.replace('.png', '.npz'))
                    else:
                        depth_ref_path = _ref_depth_path_oracle_aware(depth_base)

                    if not os.path.exists(depth_ref_path):
                        print(f"  [bins] Ref depth not found for {ref_basename}, eliminating all bins")
                        for m in range(M):
                            bin_ref_eliminated[m][r_idx] = True
                        continue

                    # Compute per-ref occlusion mask (once for all bins)
                    _occ_depth_for_mask = depth_ref_path
                    if mvs_depth_folder is not None:
                        _mvs_cand = os.path.join(mvs_depth_folder, ref_basename.replace('.png', '.npz'))
                        if os.path.exists(_mvs_cand):
                            _occ_depth_for_mask = _mvs_cand
                    try:
                        ref_occlusion_masks[r_idx] = compute_occlusion_mask(
                            input_depth=_d,
                            ref_depth_path=_occ_depth_for_mask,
                            input_frame_name=input_frame_name,
                            ref_name=ref_name,
                            transforms_path=TRANSFORMS_FILE,
                            forward_params=forward_params,
                            device=device,
                        )
                        n_occ = (~ref_occlusion_masks[r_idx]).sum().item()
                        n_total = ref_occlusion_masks[r_idx].numel()
                        # print(f"  [occ] ref{r_idx}: {n_occ}/{n_total} pixels occluded ({n_occ/n_total*100:.1f}%)")
                    except Exception as e:
                        print(f"  [occ] Failed for ref{r_idx}: {e}")

                    for m in range(M):
                        try:
                            _bin_depths = _d[bin_masks[m]]
                            _bin_avg_d = _bin_depths.median().item() if len(_bin_depths) > 0 else None
                            # Resolve MVS depth for occlusion filtering
                            _occ_depth_path = None
                            if mvs_depth_folder is not None:
                                _occ_candidate = os.path.join(mvs_depth_folder, ref_basename.replace('.png', '.npz'))
                                if os.path.exists(_occ_candidate):
                                    _occ_depth_path = _occ_candidate
                            cropped_pil, viz_pil, crop_bbox, warp_map, lanczos_pil = warp_reference_to_closeup(
                                img_ref_pil=ref_pil,
                                img_closeup_pil=input_full,
                                depth_ref_path=depth_ref_path,
                                depth_closeup_path=input_depth_path if crop_method_frame == 'backward' else None,
                                img_ref_name=ref_name,
                                img_closeup_name=input_frame_name,
                                transforms_path=TRANSFORMS_FILE,
                                device=device,
                                method=crop_method_frame,
                                mode='warp' if ref_warp else 'crop',
                                depth_mode='master',
                                forward_params=forward_params,
                                return_heatmap=True,
                                fg_mask=bin_masks[m],
                                crop_strategy=config_params.get('crop_strategy', 'relative'),
                                bin_avg_depth=_bin_avg_d,
                                occlusion_depth_path=_occ_depth_path,
                                return_warp_map=True,
                                sr_model=None if ref_warp else sr_model,
                            )
                            if ref_warp:
                                # Warp mode returns a synthesized full-frame image at closeup
                                # resolution and no crop_bbox. Downstream expects a bbox for
                                # the elimination check and for hires_crop/SR. Stub a full-ref
                                # bbox so the ref is kept; hires_crop/SR branches below will
                                # short-circuit because ref_warp disables them.
                                W_ref_img, H_ref_img = ref_pil.size
                                crop_bbox = (0.0, 0.0, float(W_ref_img), float(H_ref_img))
                                warp_map = None
                                lanczos_pil = None
                                # Resize warped pil to the ref's 1K canonical size so it can
                                # flow through the same tensorization path as a normal crop.
                                cropped_pil = cropped_pil.resize((W_ref_img, H_ref_img), Image.LANCZOS)
                        except Exception as e:
                            import traceback
                            print(f"  [bins] Failed bin{m} ref{r_idx}: {e}")
                            traceback.print_exc()
                            bin_ref_eliminated[m][r_idx] = True
                            continue

                        if crop_bbox is None:
                            bin_ref_eliminated[m][r_idx] = True
                            continue

                        W_ref_full, H_ref_full = ref_pil.size

                        # Hi-res crop: re-crop from 4K image for better quality.
                        # Disabled under ref_warp — the warp output is a full-frame
                        # synthesis, not a real crop bbox to re-sample in 4K.
                        hires_crop = config_params.get("hires_crop", False) and not ref_warp
                        if hires_crop and crop_bbox is not None:
                            # Prefer explicit hires_ref_folder if caller provided one
                            # (DS2-v3 / DS3 route). Otherwise fall back to the DS1
                            # convention of naming REF_FOLDER as `.../images_4/` and
                            # its 4K sibling as `.../images/`.
                            if hires_ref_folder is not None:
                                hires_folder = hires_ref_folder
                            else:
                                hires_folder = REF_FOLDER.replace('images_4', 'images')
                            hires_path = os.path.join(hires_folder, ref_basename)
                            if os.path.exists(hires_path):
                                hires_pil = load_image(hires_path).convert("RGB")
                                W_hi, H_hi = hires_pil.size
                                # Scale crop bbox from 1K to 4K coords
                                sx = W_hi / W_ref_full
                                sy = H_hi / H_ref_full
                                bx0 = crop_bbox[0] * sx
                                by0 = crop_bbox[1] * sy
                                bx1 = crop_bbox[2] * sx
                                by1 = crop_bbox[3] * sy
                                ix0, iy0 = int(round(bx0)), int(round(by0))
                                ix1, iy1 = int(round(bx1)), int(round(by1))
                                cw_hi, ch_hi = ix1 - ix0, iy1 - iy0
                                if cw_hi > 0 and ch_hi > 0:
                                    canvas_hi = Image.new("RGB", (cw_hi, ch_hi), (0, 0, 0))
                                    src_x0 = max(ix0, 0)
                                    src_y0 = max(iy0, 0)
                                    src_x1 = min(ix1, W_hi)
                                    src_y1 = min(iy1, H_hi)
                                    if src_x1 > src_x0 and src_y1 > src_y0:
                                        region = hires_pil.crop((src_x0, src_y0, src_x1, src_y1))
                                        canvas_hi.paste(region, (src_x0 - ix0, src_y0 - iy0))
                                    if sr_model is not None:
                                        # SR active: keep raw 4K canvas for SR to upscale
                                        # (only beneficial if canvas < target; SR worker handles this)
                                        cropped_pil = canvas_hi
                                        print(f"  [hires+sr] b{m}r{r_idx}: 4K canvas {cw_hi}x{ch_hi} (raw for SR)")
                                    else:
                                        # No SR: LANCZOS resize to model resolution
                                        cropped_pil = canvas_hi.resize(ref_pil.size, Image.LANCZOS)

                        crop_uv = (crop_bbox[0] / W_ref_full, crop_bbox[1] / H_ref_full,
                                   crop_bbox[2] / W_ref_full, crop_bbox[3] / H_ref_full)
                        bin_crops[m][r_idx] = (cropped_pil, crop_uv, ref_name)
                        bin_warp_maps[m][r_idx] = warp_map  # (H_in, W_in, 2) or None
                        if lanczos_pil is not None:
                            bin_sr_viz[m][r_idx] = (cropped_pil, lanczos_pil)  # (SR, LANCZOS) at ref_pil.size

                        # Crop occlusion mask to match (accounting for out-of-bounds padding)
                        if ref_occlusion_masks[r_idx] is not None and crop_bbox is not None:
                            _om = ref_occlusion_masks[r_idx]
                            _oh, _ow = _om.shape
                            # crop_bbox is in ref image pixel coords; mask may be at different resolution
                            W_ref_img, H_ref_img = ref_pil.size
                            _scale_x = _ow / W_ref_img
                            _scale_y = _oh / H_ref_img
                            _fx0 = int(round(crop_bbox[0] * _scale_x))
                            _fy0 = int(round(crop_bbox[1] * _scale_y))
                            _fx1 = int(round(crop_bbox[2] * _scale_x))
                            _fy1 = int(round(crop_bbox[3] * _scale_y))
                            _fw = _fx1 - _fx0
                            _fh = _fy1 - _fy0
                            if _fw > 0 and _fh > 0:
                                # Canvas matches the crop canvas in warp.py (same size, same padding)
                                _mask_canvas = np.zeros((_fh, _fw), dtype=np.uint8)
                                # Overlap with ref bounds
                                _sx0 = max(_fx0, 0)
                                _sy0 = max(_fy0, 0)
                                _sx1 = min(_fx1, _ow)
                                _sy1 = min(_fy1, _oh)
                                if _sx1 > _sx0 and _sy1 > _sy0:
                                    _valid_region = _om[_sy0:_sy1, _sx0:_sx1].cpu().numpy().astype(np.uint8)
                                    _paste_x = _sx0 - _fx0
                                    _paste_y = _sy0 - _fy0
                                    _mask_canvas[_paste_y:_paste_y+_valid_region.shape[0],
                                                 _paste_x:_paste_x+_valid_region.shape[1]] = _valid_region
                                # Resize to SAME target as the image crop (target_w × target_h from warp.py)
                                # ref_pil.size = original ref size = target for resize
                                _cm_resized = cv2.resize(_mask_canvas, ref_pil.size, interpolation=cv2.INTER_NEAREST)
                                bin_occ_masks[m][r_idx] = torch.from_numpy(_cm_resized).bool().to(device)

                # Eliminate bin×ref combos where occlusion mask is mostly False
                occ_elim_threshold = config_params.get("occ_elim_threshold", 0.0)  # 0.0 = only fully occluded
                for m in range(M):
                    for r_idx in range(K):
                        if bin_crops[m][r_idx] is not None and bin_occ_masks[m][r_idx] is not None:
                            valid_frac = bin_occ_masks[m][r_idx].float().mean().item()
                            if valid_frac <= occ_elim_threshold:
                                print(f"  [occ] Eliminating b{m}r{r_idx}: {valid_frac*100:.0f}% valid (threshold={occ_elim_threshold*100:.0f}%)")
                                bin_crops[m][r_idx] = None
                                bin_ref_eliminated[m][r_idx] = True

                # Top-N selection: keep only the best crops per bin by occlusion coverage
                max_crops_per_bin = config_params.get("max_crops_per_bin", 0)  # 0 = no limit
                if max_crops_per_bin > 0:
                    for m in range(M):
                        # Score surviving crops by valid fraction
                        scored = []
                        for r_idx in range(K):
                            if bin_crops[m][r_idx] is None:
                                continue
                            if bin_occ_masks[m][r_idx] is not None:
                                score = bin_occ_masks[m][r_idx].float().mean().item()
                            else:
                                score = 1.0  # no occlusion data = assume fully valid
                            scored.append((r_idx, score))
                        # Sort by score descending, keep top N
                        scored.sort(key=lambda x: x[1], reverse=True)
                        if len(scored) > max_crops_per_bin:
                            kept = set(r_idx for r_idx, _ in scored[:max_crops_per_bin])
                            for r_idx, score in scored[max_crops_per_bin:]:
                                bin_crops[m][r_idx] = None
                                bin_ref_eliminated[m][r_idx] = True
                            n_elim = len(scored) - max_crops_per_bin
                            print(f"  [topN] bin{m}: kept {max_crops_per_bin}/{len(scored)} crops (eliminated {n_elim}, min_kept_score={scored[max_crops_per_bin-1][1]*100:.0f}%)")

                # Log summary
                total_surviving = 0
                for m in range(M):
                    n_surv = sum(1 for x in bin_crops[m] if x is not None)
                    total_surviving += n_surv
                print(f"  [bins] {total_surviving} total crops surviving (M={M}, K={K})")

                # --- Batch SR: super-resolve all surviving crops via subprocess ---
                # Disabled under ref_warp — warp produces a full-frame synthesis at
                # closeup resolution, SR has nothing meaningful to upscale.
                if use_super_res and not ref_warp:
                    import tempfile
                    import subprocess as sp
                    import filelock

                    # Per-GPU SR lock: each SR_GPU gets its own lock file so
                    # workers using different SR_GPUs don't contend. Single-host
                    # default ('/tmp/.difix_sr_lock') is preserved when SR_GPU_ID
                    # is the legacy default, keeping backward compat for callers
                    # that share one SR_GPU.
                    sr_lock_path = f'/tmp/.difix_sr_lock_gpu{SR_GPU_ID}'
                    sr_lock = filelock.FileLock(sr_lock_path, timeout=600)

                    # Collect all surviving crops into a temp dir
                    tmp_dir = tempfile.mkdtemp(prefix='sr_crops_')
                    crop_map = {}  # filename -> (m, r_idx)
                    for m in range(M):
                        for r_idx in range(K):
                            entry = bin_crops[m][r_idx]
                            if entry is None:
                                continue
                            cropped_pil, crop_uv, ref_name = entry
                            # Save the ORIGINAL crop (before resize to ref_pil.size)
                            # We need the raw canvas — but bin_crops stores the resized version
                            # So we save the resized version and SR will upscale from there
                            fname = f"b{m}_r{r_idx}.png"
                            cropped_pil.save(os.path.join(tmp_dir, fname))
                            crop_map[fname] = (m, r_idx)

                    if crop_map:
                        tmp_out_dir = tmp_dir + '_sr'
                        # SR target = model working resolution. DO NOT use
                        # ref_images_pil[0].size here — when hires_crop=True that's
                        # the full 4K ref, and targeting 4K forces SR to upscale
                        # crops way beyond what's needed, OOMing the subprocess
                        # (37+ GiB attention allocations inside PFT-SR).
                        ref_w, ref_h = model_size
                        sr_script = os.path.join(os.path.dirname(__file__), 'sr_worker.py')
                        sr_gpu = SR_GPU_ID
                        sr_backend = config_params.get("sr_backend", "pft")

                        with sr_lock:
                            result = sp.run([
                                sys.executable, sr_script,
                                '--input_dir', tmp_dir,
                                '--output_dir', tmp_out_dir,
                                '--target_w', str(ref_w),
                                '--target_h', str(ref_h),
                                '--backend', sr_backend,
                            ], capture_output=True, text=True,
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': sr_gpu})

                        # Log SR subprocess output to file
                        sr_log_dir = os.path.join(VIZ_FOLDER, '..', '..', '..', 'sr_logs')
                        os.makedirs(sr_log_dir, exist_ok=True)
                        sr_log_path = os.path.join(sr_log_dir, f"sr_{os.path.splitext(test_fname)[0]}.log")
                        with open(sr_log_path, 'w') as _lf:
                            _lf.write(f"returncode: {result.returncode}\n")
                            _lf.write(f"--- stdout ---\n{result.stdout}\n")
                            _lf.write(f"--- stderr ---\n{result.stderr}\n")

                        if result.returncode == 0:
                            n_replaced = 0
                            sr_viz_panels = []  # for SR vs LANCZOS comparison
                            for fname, (m, r_idx) in crop_map.items():
                                sr_path = os.path.join(tmp_out_dir, fname)
                                if os.path.exists(sr_path):
                                    sr_pil = Image.open(sr_path).convert('RGB')
                                    entry = bin_crops[m][r_idx]
                                    orig_pil = entry[0]
                                    _, crop_uv, ref_name = entry
                                    # Save comparison viz: LANCZOS (orig) | SR
                                    if do_viz and n_replaced < 20:  # limit to first 20 for viz
                                        lanczos_resized = orig_pil.resize(sr_pil.size, Image.LANCZOS) if orig_pil.size != sr_pil.size else orig_pil
                                        sr_viz_panels.append((f"b{m}r{r_idx}", lanczos_resized, sr_pil))
                                    if sr_pil.size != orig_pil.size:
                                        print(f"    [SR] b{m}r{r_idx}: {orig_pil.size} -> {sr_pil.size}")
                                    bin_crops[m][r_idx] = (sr_pil, crop_uv, ref_name)
                                    n_replaced += 1
                            print(f"  [SR] Batch SR done: {n_replaced}/{len(crop_map)} crops replaced")
                            # Save SR comparison grid
                            if do_viz and sr_viz_panels:
                                _panel_w, _panel_h = 256, 256
                                _n_panels = len(sr_viz_panels)
                                _grid_w = _panel_w * 2 + 10  # LANCZOS | SR side by side
                                _grid_h = _panel_h * _n_panels
                                _grid = Image.new("RGB", (_grid_w, _grid_h), (0, 0, 0))
                                for pi, (label, lanczos_img, sr_img) in enumerate(sr_viz_panels):
                                    _l = lanczos_img.resize((_panel_w, _panel_h), Image.LANCZOS)
                                    _s = sr_img.resize((_panel_w, _panel_h), Image.LANCZOS)
                                    _grid.paste(_l, (0, pi * _panel_h))
                                    _grid.paste(_s, (_panel_w + 10, pi * _panel_h))
                                _sr_viz_path = os.path.join(VIZ_FOLDER, f"sr_comparison_{os.path.splitext(test_fname)[0]}.jpg")
                                _grid.save(_sr_viz_path)
                                print(f"  [SR] Comparison viz saved: {_sr_viz_path}")
                        else:
                            # Fallback: LANCZOS-resize the raw 4K canvases to model resolution
                            # so downstream attention matrices stay within memory budget.
                            # Without this, the raw 4K canvas stays in bin_crops and
                            # downstream tensorization builds enormous ref tensors that OOM.
                            # Use (ref_w, ref_h) — the same target that sr_worker was aiming at —
                            # NOT ref_images_pil[0].size which may be 4K when hires_crop=True.
                            #
                            # IMPORTANT: this degrades the reference crops (plain LANCZOS instead
                            # of PFT-SR), so the resulting MACRO metrics are NOT comparable to the
                            # paper. The most common cause is a CUDA OOM in the SR subprocess when
                            # the GPU is shared — free up memory (or set SR_GPU_ID to an idle GPU)
                            # and re-run. Warn loudly so this is never mistaken for a valid result.
                            fallback_size = (ref_w, ref_h)
                            n_fallback = 0
                            for fname, (m, r_idx) in crop_map.items():
                                entry = bin_crops[m][r_idx]
                                if entry is None:
                                    continue
                                cropped_pil, crop_uv, ref_name = entry
                                if cropped_pil.size != fallback_size:
                                    cropped_pil = cropped_pil.resize(fallback_size, Image.LANCZOS)
                                    bin_crops[m][r_idx] = (cropped_pil, crop_uv, ref_name)
                                    n_fallback += 1
                            print("  " + "!" * 76)
                            print(f"  [SR] *** PFT-SR FAILED (rc={result.returncode}) — FELL BACK TO LANCZOS ***")
                            print(f"  [SR] {n_fallback} crops degraded. METRICS ARE NOT PAPER-COMPARABLE.")
                            print(f"  [SR] Usually a CUDA OOM on a shared GPU. Free GPU memory / set")
                            print(f"  [SR] SR_GPU_ID to an idle GPU and re-run. Log: {sr_log_path}")
                            print("  " + "!" * 76)

                        # Cleanup
                        import shutil
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        shutil.rmtree(tmp_out_dir, ignore_errors=True)

                # --- A. Cluster overlay + B. Bin dashboard + Occ mask viz ---
                if do_viz:
                    _bm_np = bin_map.cpu().numpy()
                    _bm_norm = np.zeros_like(_bm_np, dtype=np.float32)
                    _bm_valid = _bm_np >= 0
                    if M > 1:
                        _bm_norm[_bm_valid] = _bm_np[_bm_valid].astype(np.float32) / (M - 1)
                    _bm_jet = cv2.applyColorMap((_bm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    _bm_jet = cv2.cvtColor(_bm_jet, cv2.COLOR_BGR2RGB)
                    _inp_np = np.array(input_full.convert("RGB"))
                    if _inp_np.shape[:2] != _bm_jet.shape[:2]:
                        _bm_jet = cv2.resize(_bm_jet, (_inp_np.shape[1], _inp_np.shape[0]))
                    _cluster_overlay = (0.4 * _inp_np + 0.6 * _bm_jet).astype(np.uint8)
                    _cluster_overlay[~_bm_valid] = _inp_np[~_bm_valid]
                    Image.fromarray(_cluster_overlay).save(os.path.join(VIZ_FOLDER, f"bins_{os.path.splitext(test_fname)[0]}.jpg"))

                    _bin_jet_colors = []
                    for m in range(M):
                        val = int(m / max(M - 1, 1) * 255)
                        c = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                        _bin_jet_colors.append((int(c[2]), int(c[1]), int(c[0])))

                    _dash_panel_w = 320
                    _dash_panel_h = int(_dash_panel_w * input_full.size[1] / input_full.size[0])
                    _dash_size = (_dash_panel_w, _dash_panel_h)
                    dash_rows = []
                    for m in range(M):
                        row_panels = []
                        bin_color = _bin_jet_colors[m]
                        for r_idx in range(K):
                            ref_name = selected_ref_names[r_idx]
                            rb = os.path.basename(ref_name)
                            entry = bin_crops[m][r_idx]
                            orig_ref_path = os.path.join(REF_FOLDER, rb)
                            if os.path.exists(orig_ref_path):
                                ref_panel = load_image(orig_ref_path).convert("RGB")
                                ref_panel = resize_pil(ref_panel, _dash_size)
                            else:
                                ref_panel = Image.new("RGB", _dash_size, (0, 0, 0))
                            if entry is None:
                                draw = ImageDraw.Draw(ref_panel)
                                draw.line([(0, 0), _dash_size], fill="red", width=3)
                                draw.line([(_dash_size[0], 0), (0, _dash_size[1])], fill="red", width=3)
                            else:
                                _, crop_uv, _ = entry
                                draw = ImageDraw.Draw(ref_panel)
                                bx0 = int(crop_uv[0] * _dash_size[0])
                                by0 = int(crop_uv[1] * _dash_size[1])
                                bx1 = int(crop_uv[2] * _dash_size[0])
                                by1 = int(crop_uv[3] * _dash_size[1])
                                draw.rectangle([(bx0, by0), (bx1, by1)], outline="lime", width=2)
                                depth_base_r = os.path.splitext(rb)[0]
                                if depth_source == 'mvs' and mvs_depth_folder is not None:
                                    _drp = os.path.join(mvs_depth_folder, rb.replace('.png', '.npz'))
                                else:
                                    _drp = _ref_depth_path_oracle_aware(depth_base_r)
                                if os.path.exists(_drp):
                                    try:
                                        _bin_mask_np = bin_masks[m].cpu().numpy().astype(np.uint8)
                                        _bin_mask_dash = cv2.resize(_bin_mask_np, _dash_size, interpolation=cv2.INTER_NEAREST)
                                        _bin_mask_dash_t = torch.from_numpy(_bin_mask_dash).bool().to(device)
                                        _occ_path = None
                                        if mvs_depth_folder is not None:
                                            _occ_cand = os.path.join(mvs_depth_folder, rb.replace('.png', '.npz'))
                                            if os.path.exists(_occ_cand):
                                                _occ_path = _occ_cand
                                        _full_ref_w, _full_ref_h = load_image(orig_ref_path).size
                                        _full_bbox = (crop_uv[0]*_full_ref_w, crop_uv[1]*_full_ref_h,
                                                      crop_uv[2]*_full_ref_w, crop_uv[3]*_full_ref_h) if crop_uv else None
                                        _warp_panel = build_warp_debug_panel(
                                            load_image(orig_ref_path).convert("RGB"),
                                            input_full, ref_name, input_frame_name,
                                            _drp, TRANSFORMS_FILE, None,
                                            _full_bbox, device,
                                            input_depth_path=input_depth_path,
                                            warp_method=crop_method_frame,
                                            fg_mask=bin_masks[m])
                                        ref_panel = resize_pil(_warp_panel, _dash_size)
                                    except Exception as _e:
                                        print(f"  [dash] b{m}r{r_idx} warp failed: {_e}")
                            draw = ImageDraw.Draw(ref_panel)
                            draw.text((4, 4), f"b{m}r{r_idx}", fill=bin_color)
                            row_panels.append(ref_panel)
                        _highlight = _inp_np.copy()
                        _this_bin = bin_masks[m].cpu().numpy()
                        _other = _bm_valid & ~_this_bin
                        _highlight[_other] = (_highlight[_other] * 0.3).astype(np.uint8)
                        _tint = np.array(bin_color, dtype=np.float32)
                        _highlight[_this_bin] = (0.5 * _highlight[_this_bin] + 0.5 * _tint).astype(np.uint8)
                        _hl_pil = Image.fromarray(_highlight)
                        if _hl_pil.size != _dash_size:
                            _hl_pil = resize_pil(_hl_pil, _dash_size)
                        draw = ImageDraw.Draw(_hl_pil)
                        d_center = 1.0 / bin_centers[m].item() if bin_centers[m] > 0 else float('inf')
                        draw.text((4, 4), f"bin{m} d~{d_center:.1f}", fill="white")
                        row_panels.append(_hl_pil)
                        row_img = Image.new("RGB", (_dash_size[0] * (K + 1), _dash_size[1]))
                        for ci, p in enumerate(row_panels):
                            row_img.paste(p, (ci * _dash_size[0], 0))
                        dash_rows.append(np.array(row_img))
                    dash_img = np.vstack(dash_rows)
                    cv2.imwrite(os.path.join(VIZ_FOLDER, f"bins_dashboard_{os.path.splitext(test_fname)[0]}.jpg"),
                        cv2.cvtColor(dash_img, cv2.COLOR_RGB2BGR))

                    if any(m is not None for m in ref_occlusion_masks):
                        occ_rows = []
                        for m in range(M):
                            row_panels = []
                            for r_idx in range(K):
                                mask = bin_occ_masks[m][r_idx]
                                if mask is not None:
                                    entry = bin_crops[m][r_idx]
                                    if entry is not None:
                                        crop_pil, _, _ = entry
                                        _ref_np = np.array(resize_pil(crop_pil, _dash_size).convert("L"))
                                        _ref_rgb = np.stack([_ref_np]*3, axis=-1)
                                    else:
                                        _ref_rgb = np.zeros((_dash_size[1], _dash_size[0], 3), dtype=np.uint8)
                                    _m_np = mask.cpu().numpy().astype(np.uint8)
                                    _m_resized = cv2.resize(_m_np, _dash_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                                    _ref_rgb[~_m_resized] = [255, 60, 60]
                                    panel = Image.fromarray(_ref_rgb)
                                else:
                                    panel = Image.new("RGB", _dash_size, (0, 0, 0))
                                    if bin_crops[m][r_idx] is None:
                                        draw = ImageDraw.Draw(panel)
                                        draw.line([(0,0), _dash_size], fill="red", width=3)
                                        draw.line([(_dash_size[0],0), (0,_dash_size[1])], fill="red", width=3)
                                draw = ImageDraw.Draw(panel)
                                draw.text((4, 4), f"b{m}r{r_idx}", fill="white")
                                row_panels.append(panel)
                            row_panels.append(Image.new("RGB", _dash_size, (40, 40, 40)))
                            row_img = Image.new("RGB", (_dash_size[0] * (K + 1), _dash_size[1]))
                            for ci, p in enumerate(row_panels):
                                row_img.paste(p, (ci * _dash_size[0], 0))
                            occ_rows.append(np.array(row_img))
                        occ_grid = np.vstack(occ_rows)
                        cv2.imwrite(os.path.join(VIZ_FOLDER, f"occ_masks_{os.path.splitext(test_fname)[0]}.jpg"),
                            cv2.cvtColor(occ_grid, cv2.COLOR_RGB2BGR))

                    # --- SR comparison visualization: 3 grids (SR, LANCZOS, diff) ---
                    if any(bin_sr_viz[m][r] is not None for m in range(M) for r in range(K)):
                        for grid_type in ['sr', 'lanczos', 'diff']:
                            sr_rows = []
                            for m in range(M):
                                row_panels = []
                                for r_idx in range(K):
                                    entry = bin_sr_viz[m][r_idx]
                                    if entry is not None:
                                        sr_pil, lz_pil = entry
                                        sr_np = np.array(resize_pil(sr_pil, _dash_size))
                                        lz_np = np.array(resize_pil(lz_pil, _dash_size))
                                        if grid_type == 'sr':
                                            panel = sr_np
                                        elif grid_type == 'lanczos':
                                            panel = lz_np
                                        else:
                                            diff = np.abs(sr_np.astype(float) - lz_np.astype(float)).mean(axis=2)
                                            vmax = max(diff.max(), 1.0)
                                            diff_jet = cv2.applyColorMap((diff / vmax * 255).astype(np.uint8), cv2.COLORMAP_JET)
                                            panel = cv2.cvtColor(diff_jet, cv2.COLOR_BGR2RGB)
                                            cv2.putText(panel, f"max={vmax:.0f}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                                    else:
                                        panel = np.zeros((_dash_size[1], _dash_size[0], 3), dtype=np.uint8)
                                    draw = ImageDraw.Draw(Image.fromarray(panel))
                                    draw.text((4, 4), f"b{m}r{r_idx}", fill="white")
                                    row_panels.append(np.array(Image.fromarray(panel)))
                                sr_rows.append(np.hstack(row_panels))
                            grid = np.vstack(sr_rows)
                            cv2.imwrite(os.path.join(VIZ_FOLDER, f"sr_{grid_type}_{os.path.splitext(test_fname)[0]}.jpg"),
                                cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            else:
                # --- Single-mask crop (fg/bg mode) ---
                fg_mask = None
                if _input_depth_tensor is not None:
                    _d = _input_depth_tensor.squeeze()
                    _valid = _d > 0
                    if _valid.any():
                        _median = _d[_valid].median()
                        if crop_layer == 'bg':
                            fg_mask = _valid & (_d > _median)
                        else:
                            fg_mask = _valid & (_d <= _median)
                        fg_pct = fg_mask.float().mean().item() * 100
                        # print(f"  [crop] {crop_layer.upper()} mask: median_depth={_median:.3f}, pixels={fg_pct:.1f}%")
                    else:
                        print(f"  [crop] No valid depth pixels, skipping filter")

                for r_idx in range(K):
                    ref_name = selected_ref_names[r_idx]
                    ref_pil = ref_images_pil[r_idx]
                    ref_basename = os.path.basename(ref_name)
                    depth_base = os.path.splitext(ref_basename)[0]

                    if depth_source == 'mvs' and mvs_depth_folder is not None:
                        depth_ref_path = os.path.join(mvs_depth_folder, ref_basename.replace('.png', '.npz'))
                    else:
                        depth_ref_path = _ref_depth_path_oracle_aware(depth_base)

                    if not os.path.exists(depth_ref_path):
                        print(f"  [crop] Ref depth not found for {ref_basename}, eliminating")
                        ref_eliminated[r_idx] = True
                        continue

                    try:
                        cropped_pil, viz_pil, crop_bbox, _, _ = warp_reference_to_closeup(
                            img_ref_pil=ref_pil,
                            img_closeup_pil=input_full,
                            depth_ref_path=depth_ref_path,
                            depth_closeup_path=input_depth_path if crop_method_frame == 'backward' else None,
                            img_ref_name=ref_name,
                            img_closeup_name=input_frame_name,
                            transforms_path=TRANSFORMS_FILE,
                            device=device,
                            method=crop_method_frame,
                            mode='warp' if ref_warp else 'crop',
                            depth_mode='master',
                            forward_params=forward_params,
                            return_heatmap=True,
                            fg_mask=fg_mask,
                            sr_model=None if ref_warp else sr_model,
                        )
                        if ref_warp:
                            W_ref_img, H_ref_img = ref_pil.size
                            crop_bbox = (0.0, 0.0, float(W_ref_img), float(H_ref_img))
                            cropped_pil = cropped_pil.resize((W_ref_img, H_ref_img), Image.LANCZOS)
                    except Exception as e:
                        print(f"  [crop] Failed for ref{r_idx} ({ref_basename}): {e}")
                        ref_eliminated[r_idx] = True
                        continue

                    if crop_bbox is None:
                        print(f"  [crop] No overlap for ref{r_idx} ({ref_basename}), eliminating")
                        ref_eliminated[r_idx] = True
                        continue

                    W_ref_full, H_ref_full = ref_pil.size
                    ref_crop_uv = (crop_bbox[0] / W_ref_full, crop_bbox[1] / H_ref_full,
                                   crop_bbox[2] / W_ref_full, crop_bbox[3] / H_ref_full)
                    crop_area_frac = (ref_crop_uv[2] - ref_crop_uv[0]) * (ref_crop_uv[3] - ref_crop_uv[1])

                    ref_images_pil[r_idx] = cropped_pil.resize(ref_pil.size, Image.LANCZOS)
                    ref_crop_uvs[r_idx] = ref_crop_uv
                    # print(f"  [crop] ref{r_idx}: bbox=({crop_bbox[0]:.0f},{crop_bbox[1]:.0f},{crop_bbox[2]:.0f},{crop_bbox[3]:.0f}) "
                    #       f"uv=({ref_crop_uv[0]:.3f},{ref_crop_uv[1]:.3f},{ref_crop_uv[2]:.3f},{ref_crop_uv[3]:.3f}) "
                    #       f"area={crop_area_frac*100:.1f}%")

                    if do_viz and viz_pil is not None:
                        viz_pil.save(os.path.join(VIZ_FOLDER, f"crop_viz_ref{r_idx}_{os.path.splitext(test_fname)[0]}.jpg"))

        # --- Keep original lists for viz, filter eliminated refs for model ---
        _t4 = time.time()
        all_ref_images_pil = list(ref_images_pil)  # full list for viz (including eliminated)
        all_selected_ref_names = list(selected_ref_names)
        all_ref_crop_uvs = list(ref_crop_uvs)
        all_ref_eliminated = list(ref_eliminated)  # save before bin flattening replaces it

        # ref_to_bin: list parallel to ref_images_pil, mapping each ref to its bin index (or -1 for no bin)
        ref_to_bin = [-1] * len(ref_images_pil)
        latent_bin_map = None  # (H_lat, W_lat) long tensor, set for bins mode

        if ref_crop and crop_layer == 'bins' and bin_crops is not None:
            # Flatten M×K bin_crops into ref lists for the model
            M = len(bin_crops)
            ref_images_pil = []
            selected_ref_names = []
            ref_crop_uvs = []
            ref_to_bin = []
            ref_eliminated = []
            flat_occ_masks = []  # per-ref occlusion masks at model resolution
            flat_warp_maps = []  # per-ref warp maps (H_in, W_in, 2)

            for m in range(M):
                for r_idx in range(K):
                    entry = bin_crops[m][r_idx]
                    if entry is None:
                        continue  # eliminated
                    cropped_pil, crop_uv, ref_name = entry
                    ref_images_pil.append(cropped_pil)
                    selected_ref_names.append(ref_name)
                    ref_crop_uvs.append(crop_uv)
                    ref_to_bin.append(m)
                    ref_eliminated.append(False)
                    flat_occ_masks.append(bin_occ_masks[m][r_idx])  # may be None
                    flat_warp_maps.append(bin_warp_maps[m][r_idx])  # may be None

            n_total = len(ref_images_pil)
            # print(f"  [bins] Flattened to {n_total} ref crops (M={M} × K={K}, after elimination)")

            # Compute latent-resolution bin_map (majority vote per 8×8 patch)
            _bm = bin_map.clone()  # (H, W) long tensor
            H_img, W_img = _bm.shape
            H_lat, W_lat = H_img // 8, W_img // 8
            # Reshape into (H_lat, 8, W_lat, 8) patches, take mode
            _bm_padded = _bm[:H_lat * 8, :W_lat * 8].view(H_lat, 8, W_lat, 8)
            _bm_patches = _bm_padded.permute(0, 2, 1, 3).reshape(H_lat, W_lat, 64)
            # Majority vote: for each patch, find the most common bin (ignoring -1)
            latent_bin_map = torch.zeros(H_lat, W_lat, dtype=torch.long, device=device)
            for ly in range(H_lat):
                for lx in range(W_lat):
                    patch = _bm_patches[ly, lx]
                    valid_patch = patch[patch >= 0]
                    if len(valid_patch) > 0:
                        latent_bin_map[ly, lx] = valid_patch.mode().values
                    else:
                        latent_bin_map[ly, lx] = 0

        elif ref_crop and any(ref_eliminated):
            n_elim = sum(ref_eliminated)
            print(f"  [crop] Eliminated {n_elim}/{K} refs, using {K - n_elim}")
            # Filter to surviving refs only for model
            ref_images_pil = [r for r, e in zip(ref_images_pil, ref_eliminated) if not e]
            selected_ref_names = [r for r, e in zip(selected_ref_names, ref_eliminated) if not e]
            ref_crop_uvs = [r for r, e in zip(ref_crop_uvs, ref_eliminated) if not e]

        # --- Prepare K ref tensors (0-1 range, pipeline preprocessor normalizes to [-1,1]) ---
        # GT swap: replace ref images with GT for enhancement (masks stay from real refs)
        if gt_ref_swap and gt_folder is not None:
            _gt_basename = os.path.basename(input_frame_name)
            _gt_path = os.path.join(gt_folder, _gt_basename)
            if os.path.exists(_gt_path):
                _gt_pil = load_image(_gt_path).convert("RGB")
                ref_images_pil = [_gt_pil] * len(ref_images_pil)
                print(f"  [gt_swap] Replaced {len(ref_images_pil)} ref tensors with GT")
        ref_tensors = []
        for ref_pil in ref_images_pil:
            ref_t = TF.to_tensor(resize_pil(ref_pil, ref_size))
            ref_tensors.append(ref_t)

        # --- Input Preparation (0-1 range) ---
        input_for_model = TF.to_tensor(resize_pil(input_full, model_size))

        # --- GT resized to model_size for metric comparison ---
        gt_resized = resize_pil(gt_full, model_size)
        gt_for_disp = TF.to_tensor(gt_resized)

        # --- Build K epipolar caches ---
        _t5 = time.time()
        epipolar_caches = []
        if len(ref_tensors) == 0:
            print(f"  [WARN] No refs survived for {test_fname}, skipping")
            continue
        for r_idx, (ref_name, ref_pil) in enumerate(zip(selected_ref_names, ref_images_pil)):
            cache = get_epipolar_cache(
                input_full, ref_pil, input_frame_name, ref_name,
                TRANSFORMS_FILE, device, VIZ_FOLDER, 0,
                zoom=1.0, forward_params=forward_params,
                ref_crop_bbox=ref_crop_uvs[r_idx]
            )
            cache.img_model_input_pil = resize_pil(input_full, model_size)
            # For crop mode, img_target_pil should be the cropped ref (already in ref_images_pil)
            if ref_crop_uvs[r_idx] is not None:
                cache.img_target_pil = resize_pil(ref_pil, model_size)
            cache.frame_name = os.path.splitext(test_fname)[0]
            epipolar_caches.append(cache)

        # --- ECA / Warp Metadata Setup (v2: batch-of-(K+1), no tiling) ---
        _t6 = time.time()
        import types
        import importlib
        _src_mv_unet = importlib.import_module('mv_unet') if 'mv_unet' not in sys.modules or \
            (hasattr(sys.modules['mv_unet'], '__file__') and 'See3D' in (sys.modules['mv_unet'].__file__ or '')) \
            else sys.modules['mv_unet']
        if hasattr(_src_mv_unet, '__file__') and 'See3D' in (_src_mv_unet.__file__ or ''):
            # See3D's mv_unet is cached — force reload from src/
            import importlib.util
            _spec = importlib.util.spec_from_file_location('mv_unet_difix', os.path.join(os.path.dirname(__file__), 'mv_unet.py'))
            _src_mv_unet = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_src_mv_unet)
        new_forward = _src_mv_unet.new_forward
        from diffusers.models.attention import BasicTransformerBlock

        H_full, W_full = input_for_model.shape[-2:]
        warp_metadata = dict()
        warp_metadata['S_to_HW'] = build_S_to_HW(H_full=H_full, W_full=W_full)
        # Same dims for refs (model_size = ref_size, no tiling)
        warp_metadata['S_to_HW_REF'] = warp_metadata['S_to_HW']
        warp_metadata['epipolar_caches'] = epipolar_caches
        warp_metadata['num_refs'] = len(ref_tensors)
        warp_metadata['ref_to_bin'] = ref_to_bin  # list: ref batch idx -> bin idx (-1 = no bin)
        warp_metadata['latent_bin_map'] = latent_bin_map  # (H_lat, W_lat) or None
        warp_metadata['K_orig'] = K  # original number of refs (before bin flattening)
        warp_metadata['occ_masks'] = flat_occ_masks if (ref_crop and crop_layer == 'bins' and flat_occ_masks is not None) else None
        warp_metadata['occ_mask_enabled'] = occ_mask_enabled
        warp_metadata['warp_maps'] = flat_warp_maps if (ref_crop and crop_layer == 'bins' and flat_warp_maps is not None) else None
        warp_metadata['warp_patch'] = warp_patch
        warp_metadata['warp_forced'] = warp_forced
        warp_metadata['input_local_patch'] = input_local_patch
        warp_metadata['ref_boost'] = ref_boost
        warp_metadata['confidence_mask'] = confidence_mask  # (H_in, W_in) int tensor, 0..16
        warp_metadata['conf_mask_enabled'] = conf_mask_enabled
        warp_metadata['block_ref_to_input'] = block_ref_to_input

        # ECA wrapping: wrap each DiFix BasicTransformerBlock's self-attention
        # (attn1) with an EpipolarMixingBlock so cross-view reference attention
        # is routed through the depth-aware mask.
        for name, module in pipe.unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                module.warp_metadata = warp_metadata
                module.epipolar_cache = epipolar_caches[0] if epipolar_caches else None
                if not isinstance(module.attn1, EpipolarMixingBlock):
                    mixing_block = EpipolarMixingBlock(module.attn1, threshold=0.03, viz_folder=VIZ_FOLDER if do_viz else None)
                    module.attn1 = mixing_block
                module.attn1.mask_mode = mask_mode
                module.attn1.cross_attn_mode = cross_attn_mode
                module.attn1.attention_mode = attention_mode
                module.attn1.is_down = 'down' in name
                if ref_crop:
                    module.attn1._block_mask_cache.clear()
                module.forward = types.MethodType(new_forward, module)
                module.name = name

        # --- Inference ---
        _t7 = time.time()

        # Apply enhance_scale: resize all tensors for inference, undo on output
        original_model_size = model_size  # (W, H) for undoing scale
        if enhance_scale != 1.0:
            sw = int(round(model_size[0] * enhance_scale / 8)) * 8
            sh = int(round(model_size[1] * enhance_scale / 8)) * 8
            scaled_size = (sw, sh)
            input_for_model = TF.resize(input_for_model, [sh, sw], antialias=True)
            ref_tensors = [TF.resize(r, [sh, sw], antialias=True) for r in ref_tensors]
            # Update warp_metadata spatial dims for ECA
            warp_metadata['S_to_HW'] = build_S_to_HW(H_full=sh, W_full=sw)
            warp_metadata['S_to_HW_REF'] = warp_metadata['S_to_HW']
            # print(f"  [scale] {model_size} -> {scaled_size} (S={enhance_scale})")

        # print(f"  Input: {input_for_model.shape}, Refs: {len(ref_tensors)}x{ref_tensors[0].shape}")

        # --- DiFix inference path ---
        with torch.no_grad():
            current_input = input_for_model
            for pass_idx in range(num_enhance_passes):
                out = pipe(
                    "remove degradation",
                    image=current_input,
                    ref_images=ref_tensors,
                    num_inference_steps=1,
                    timesteps=[199],
                    guidance_scale=0.0
                ).images[0]
                if pass_idx < num_enhance_passes - 1:
                    # Convert output PIL back to tensor for next pass (0-1 range)
                    current_input = TF.to_tensor(out.resize(
                        (current_input.shape[-1], current_input.shape[-2]), Image.LANCZOS))
                if num_enhance_passes > 1:
                    print(f"    pass {pass_idx+1}/{num_enhance_passes} done")

        # Undo scale: resize output back to original model_size
        if enhance_scale != 1.0:
            out = out.resize(original_model_size, Image.LANCZOS)

        # Optional post-processing sharpen via ESRGAN: upsample once with
        # Real-ESRGAN then immediately downsample back to `out`'s resolution.
        # Activated by config_params["sharpen_output"] == True. Uses the SR
        # subprocess so the heavy model stays on SR_GPU_ID, same as the crop
        # SR path. Runs once per frame.
        if config_params.get("sharpen_output", False):
            import tempfile
            import subprocess as sp
            import filelock as _fl
            _sw, _sh = out.size
            with tempfile.TemporaryDirectory() as _sd:
                _in_dir = os.path.join(_sd, 'in');  os.makedirs(_in_dir)
                _out_dir = os.path.join(_sd, 'out'); os.makedirs(_out_dir)
                out.save(os.path.join(_in_dir, 'frame.png'))
                _sharp_script = os.path.join(os.path.dirname(__file__), 'sr_worker.py')
                _sharp_gpu = SR_GPU_ID
                _sharp_lock = _fl.FileLock(f'/tmp/sr_sharpen.lock_gpu{SR_GPU_ID}')
                with _sharp_lock:
                    _res = sp.run([
                        sys.executable, _sharp_script,
                        '--input_dir', _in_dir,
                        '--output_dir', _out_dir,
                        '--target_w', str(_sw),
                        '--target_h', str(_sh),
                        '--backend', 'esrgan',
                        '--force-run',
                    ], capture_output=True, text=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': _sharp_gpu})
                _sharp_out = os.path.join(_out_dir, 'frame.png')
                if _res.returncode == 0 and os.path.exists(_sharp_out):
                    out = Image.open(_sharp_out).convert('RGB')
                else:
                    print(f"  [SHARPEN] WARN failed (rc={_res.returncode}); keeping unsharpened output")

        # Compute metrics: resize output to GT size, compare
        _t8 = time.time()
        if _gt_exists:
            out_for_metrics = out.resize(gt_full.size, Image.LANCZOS)
            m = compute_image_metrics(out_for_metrics, gt_full, metric_fns, device)
            frame_metrics.append(m)
            _t9 = time.time()
            print(f"  [METRICS] {test_fname}: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f}")
        else:
            _t9 = time.time()
            # No on-disk GT for this frame — skip metrics to avoid misleading
            # self-vs-input comparisons.
        print(f"  [TIME] load={_t1-_t0:.1f}s ref_sel={_t2-_t1:.1f}s load_refs={_t3-_t2:.1f}s crop={_t4-_t3:.1f}s prep={_t5-_t4:.1f}s epi={_t6-_t5:.1f}s eca={_t7-_t6:.1f}s infer={_t8-_t7:.1f}s metrics={_t9-_t8:.1f}s total={_t9-_t0:.1f}s")

        # Save enhanced output if requested.
        if save_outputs_dir is not None:
            os.makedirs(save_outputs_dir, exist_ok=True)
            out.save(os.path.join(save_outputs_dir, test_fname))

        # --- Visualization ---
        # HQ export for presentation is allowed even when do_viz is False
        # (it's a small, self-contained set of pngs; see block near end of loop).
        _save_hq = config_params.get("save_hq_pngs", False)
        if not do_viz and not _save_hq:
            continue
        print(f"  [DBG] viz start")
        to_pil = transforms.ToPILImage()

        # Use original model_size for all viz panels (not scaled)
        _disp_size = original_model_size if enhance_scale != 1.0 else model_size
        input_disp = resize_pil(input_full, _disp_size)
        gt_disp = resize_pil(gt_full, _disp_size)
        out_disp = out.resize(_disp_size, Image.LANCZOS) if out.size != _disp_size else out
        ref_disps = []
        for r_idx, ref_pil in enumerate(all_ref_images_pil):
            disp = resize_pil(ref_pil, input_disp.size)
            if all_ref_eliminated[r_idx]:
                # Draw red X on eliminated refs
                draw = ImageDraw.Draw(disp)
                w_d, h_d = disp.size
                draw.line([(0, 0), (w_d, h_d)], fill="red", width=4)
                draw.line([(w_d, 0), (0, h_d)], fill="red", width=4)
            ref_disps.append(disp)
        print(f"  [DBG] summary panels built")

        # Build coverage panel
        if coverage_maps_selected is not None:
            coverage_panel = build_coverage_panel(input_full, coverage_maps_selected, input_disp.size)
        else:
            coverage_panel = input_disp.copy()

        # Order: Ref0..RefK | Input | Output | GT | Coverage
        all_images = ref_disps + [input_disp, out_disp, gt_disp, coverage_panel]
        if coverage_maps_selected is not None:
            def _hull_coverage(mask):
                """Fraction of image inside convex hull of True pixels."""
                ys, xs = np.where(mask)
                if len(xs) < 3:
                    return mask.mean()
                pts = np.stack([xs, ys], axis=1)
                hull = cv2.convexHull(pts)
                hull_mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.fillConvexPoly(hull_mask, hull, 1)
                return hull_mask.astype(bool).mean()

            ref_labels = [f"Ref{i} ({_hull_coverage(coverage_maps_selected[i])*100:.0f}%)"
                          for i in range(len(ref_disps))]
        elif ref_coverages_selected is not None:
            ref_labels = [f"Ref{i} ({ref_coverages_selected[i]*100:.0f}%)"
                          for i in range(len(ref_disps))]
        else:
            ref_labels = [f"Ref{i}" for i in range(len(ref_disps))]
        # Mark eliminated refs
        for i in range(len(ref_labels)):
            if all_ref_eliminated[i]:
                ref_labels[i] += " ELIM"
        all_labels = ref_labels + ["Input", "Output", "GT", "Coverage"]
        all_colors = [REF_COLORS[i % len(REF_COLORS)] for i in range(len(ref_disps))] + ["red"] * 4

        frame_save_path = os.path.join(VIZ_FOLDER, f"{test_fname}")
        combined_pil = concat_images_with_labels(
            images=all_images,
            labels=all_labels,
            path=frame_save_path,
            colors=all_colors,
        )

        frame_bgr = cv2.cvtColor(np.array(combined_pil), cv2.COLOR_RGB2BGR)
        print(f"  [DBG] summary image built")

        # Compact summary: Input | Output | GT only
        compact_pil = concat_images_with_labels(
            images=[input_disp, out_disp, gt_disp],
            labels=["Input", "Output", "GT"],
            path="",
            colors=["red", "red", "red"],
        )
        compact_bgr = cv2.cvtColor(np.array(compact_pil), cv2.COLOR_RGB2BGR)

        frame_buffer.append(frame_bgr)
        if not hasattr(main_video_pipeline, '_compact_buffer'):
            main_video_pipeline._compact_buffer = []
        main_video_pipeline._compact_buffer.append(compact_bgr)
        print(f"  [DBG] frame_buffer appended, starting depth stack")

        # --- Depth visualization stack ---
        disp_size = input_disp.size  # (W, H)
        ref_depths = []
        all_depth_vals = []
        for r_idx, ref_name in enumerate(all_selected_ref_names):
            rb = os.path.basename(ref_name)
            d = load_depth_for_viz(rb, depth_source, mvs_depth_folder, DEPTH_FOLDER)
            # Apply same crop as ref images
            if d is not None and all_ref_crop_uvs[r_idx] is not None:
                h_d, w_d = d.shape
                u0, v0, u1, v1 = all_ref_crop_uvs[r_idx]
                y0 = max(0, int(v0 * h_d))
                y1 = min(h_d, int(v1 * h_d))
                x0 = max(0, int(u0 * w_d))
                x1 = min(w_d, int(u1 * w_d))
                if y1 > y0 and x1 > x0:
                    d = d[y0:y1, x0:x1]
                else:
                    d = None
            ref_depths.append(d)
            if d is not None:
                valid = d[d > 0]
                if len(valid) > 0:
                    all_depth_vals.append(valid)

        if all_depth_vals:
            all_vals = np.concatenate(all_depth_vals)
            vmin = float(np.percentile(all_vals, 2))
            vmax = float(np.percentile(all_vals, 98))

            depth_panels = []
            for r_idx, d in enumerate(ref_depths):
                rb = os.path.basename(all_selected_ref_names[r_idx])
                if d is not None:
                    jet = depth_to_jet_pil(d, vmin, vmax, size=disp_size)
                    med = float(np.median(d[d > 0])) if (d > 0).any() else 0
                    lbl = f"Ref{r_idx} d:[{d.min():.2f}-{d.max():.2f}] med={med:.2f}"
                else:
                    jet = Image.new("RGB", disp_size, (0, 0, 0))
                    lbl = f"Ref{r_idx} NO DEPTH"
                depth_panels.append((jet, lbl))

            # Fill input panel with input depth (JET), rest blank for alignment
            blank = Image.new("RGB", disp_size, (40, 40, 40))
            if input_depth_path and os.path.exists(input_depth_path):
                _id = load_depth_for_viz(os.path.basename(input_depth_path).replace('_depth.tiff', '.png'),
                                         'gsplat', None, os.path.dirname(input_depth_path))
                if _id is None:
                    # Try loading directly
                    _id = np.array(Image.open(input_depth_path)).astype(np.float32)
                input_depth_panel = depth_to_jet_pil(_id, vmin, vmax, size=disp_size)
            else:
                input_depth_panel = blank
            depth_images = [p[0] for p in depth_panels] + [input_depth_panel, blank, blank, blank]
            depth_labels = [p[1] for p in depth_panels] + ["Input depth", f"range [{vmin:.2f}, {vmax:.2f}]", "", ""]
            depth_colors = [REF_COLORS[i % len(REF_COLORS)] for i in range(len(depth_panels))] + ["white"] * 4

            depth_combined = concat_images_with_labels(depth_images, depth_labels, "", colors=depth_colors)
            depth_buffer.append(cv2.cvtColor(np.array(depth_combined), cv2.COLOR_RGB2BGR))
            print(f"  [DBG] depth stack done")

        # --- MVS depth stack (same layout, no input column) ---
        if mvs_depth_folder is not None:
            mvs_depths = []
            mvs_depth_vals = []
            for r_idx, ref_name in enumerate(all_selected_ref_names):
                rb = os.path.basename(ref_name)
                d = load_depth_for_viz(rb, 'mvs', mvs_depth_folder, DEPTH_FOLDER)
                if d is not None and all_ref_crop_uvs[r_idx] is not None:
                    h_d, w_d = d.shape
                    u0, v0, u1, v1 = all_ref_crop_uvs[r_idx]
                    y0 = max(0, int(v0 * h_d))
                    y1 = min(h_d, int(v1 * h_d))
                    x0 = max(0, int(u0 * w_d))
                    x1 = min(w_d, int(u1 * w_d))
                    if y1 > y0 and x1 > x0:
                        d = d[y0:y1, x0:x1]
                    else:
                        d = None
                mvs_depths.append(d)
                if d is not None:
                    valid = d[d > 0]
                    if len(valid) > 0:
                        mvs_depth_vals.append(valid)

            if mvs_depth_vals:
                all_vals = np.concatenate(mvs_depth_vals)
                vmin_m = float(np.percentile(all_vals, 2))
                vmax_m = float(np.percentile(all_vals, 98))
                mvs_panels = []
                for r_idx, d in enumerate(mvs_depths):
                    rb = os.path.basename(all_selected_ref_names[r_idx])
                    if d is not None:
                        jet = depth_to_jet_pil(d, vmin_m, vmax_m, size=disp_size)
                        med = float(np.median(d[d > 0])) if (d > 0).any() else 0
                        lbl = f"Ref{r_idx} MVS [{d.min():.2f}-{d.max():.2f}]"
                    else:
                        jet = Image.new("RGB", disp_size, (0, 0, 0))
                        lbl = f"Ref{r_idx} NO MVS"
                    mvs_panels.append((jet, lbl))
                blank = Image.new("RGB", disp_size, (40, 40, 40))
                mvs_images = [p[0] for p in mvs_panels] + [blank] * 4
                mvs_labels = [p[1] for p in mvs_panels] + [f"MVS [{vmin_m:.2f},{vmax_m:.2f}]", "", "", ""]
                mvs_colors = [REF_COLORS[i % len(REF_COLORS)] for i in range(len(mvs_panels))] + ["white"] * 4
                mvs_combined = concat_images_with_labels(mvs_images, mvs_labels, "", colors=mvs_colors)
                if not hasattr(main_video_pipeline, '_mvs_depth_buffer'):
                    main_video_pipeline._mvs_depth_buffer = []
                main_video_pipeline._mvs_depth_buffer.append(cv2.cvtColor(np.array(mvs_combined), cv2.COLOR_RGB2BGR))

        # --- Warp debug stack: show where input FOV maps on each full ref ---
        if ref_crop:
            print(f"  [DBG] starting warp debug stack")
            warp_panels = []
            for r_idx, ref_name in enumerate(all_selected_ref_names):
                rb = os.path.basename(ref_name)
                depth_base = os.path.splitext(rb)[0]
                if depth_source == 'mvs' and mvs_depth_folder is not None:
                    drp = os.path.join(mvs_depth_folder, rb.replace('.png', '.npz'))
                else:
                    drp = os.path.join(DEPTH_FOLDER, f"{depth_base}_depth.tiff")

                # Load original (uncropped) ref for overlay
                orig_ref_path = os.path.join(REF_FOLDER, rb)
                if os.path.exists(orig_ref_path) and os.path.exists(drp):
                    orig_ref = load_image(orig_ref_path).convert("RGB")
                    cb = all_ref_crop_uvs[r_idx]
                    # Convert UV crop to pixel bbox for green rect
                    if cb is not None:
                        ow, oh = orig_ref.size
                        pix_bbox = (cb[0]*ow, cb[1]*oh, cb[2]*ow, cb[3]*oh)
                    else:
                        pix_bbox = None
                    panel = build_warp_debug_panel(
                        orig_ref, input_full, ref_name, input_frame_name,
                        drp, TRANSFORMS_FILE, forward_params, pix_bbox, device,
                        input_depth_path=input_depth_path,
                        warp_method=crop_method_frame,
                        fg_mask=fg_mask)
                    print(f"  [DBG] warp debug ref{r_idx} done")
                    panel = resize_pil(panel, disp_size)
                else:
                    panel = Image.new("RGB", disp_size, (0, 0, 0))
                warp_panels.append(panel)

            blank = Image.new("RGB", disp_size, (40, 40, 40))
            # Input panel: show fg_mask overlay on input image
            if fg_mask is not None:
                _inp_np = np.array(resize_pil(input_full, disp_size).convert("RGB"))
                _fg_up = cv2.resize(fg_mask.cpu().numpy().astype(np.uint8),
                                    (disp_size[0], disp_size[1]), interpolation=cv2.INTER_NEAREST).astype(bool)
                # Darken background, keep foreground bright
                _inp_np[~_fg_up] = (_inp_np[~_fg_up] * 0.3).astype(np.uint8)
                fg_panel = Image.fromarray(_inp_np)
            else:
                fg_panel = blank
            warp_images = warp_panels + [fg_panel, blank, blank, blank]
            warp_labels = [f"Ref{i} warp" for i in range(len(warp_panels))] + ["FG mask", "", "", ""]
            warp_colors = [REF_COLORS[i % len(REF_COLORS)] for i in range(len(warp_panels))] + ["white"] * 4
            warp_combined = concat_images_with_labels(warp_images, warp_labels, "", colors=warp_colors)
            warp_debug_buffer.append(cv2.cvtColor(np.array(warp_combined), cv2.COLOR_RGB2BGR))
            print(f"  [DBG] warp debug stack done")

        # --- High-quality PNG export for presentation ---
        if config_params.get("save_hq_pngs", False) and bin_crops is not None:
            hq_dir = os.path.join(VIZ_FOLDER, f"hq_{os.path.splitext(test_fname)[0]}")
            os.makedirs(hq_dir, exist_ok=True)

            # [1] M×K crops
            M_bins = len(bin_crops)
            for m in range(M_bins):
                for r_idx in range(K):
                    entry = bin_crops[m][r_idx]
                    if entry is not None:
                        crop_pil, _, _ = entry
                        crop_pil.save(os.path.join(hq_dir, f"crop_b{m}_r{r_idx}.png"))

            # [2] K full references
            for r_idx, ref_pil in enumerate(all_ref_images_pil):
                ref_pil.save(os.path.join(hq_dir, f"ref_{r_idx}.png"))

            # [3] Input and output
            input_disp.save(os.path.join(hq_dir, "input.png"))
            out.save(os.path.join(hq_dir, "output.png"))
            gt_disp.save(os.path.join(hq_dir, "gt.png"))

            # [4] Bin masks with JET colors overlaid on input.
            # The overlay keeps non-bin pixels at full brightness (looks nicer
            # in the paper figure) — only the bin pixels are tinted.
            if bin_map is not None:
                _disp_w, _disp_h = input_disp.size
                _inp_arr = np.array(input_disp.convert("RGB"))
                _bm_np = bin_map.cpu().numpy()
                # Resize bin_map to match display size
                _bm_np = cv2.resize(_bm_np.astype(np.float32), (_disp_w, _disp_h), interpolation=cv2.INTER_NEAREST).astype(int)
                for m in range(M_bins):
                    _this = (_bm_np == m)
                    panel = _inp_arr.copy()
                    # Tint bin pixels with JET color (50% blend).
                    val = int(m / max(M_bins - 1, 1) * 255)
                    jc = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                    tint = np.array([int(jc[2]), int(jc[1]), int(jc[0])], dtype=np.float32)
                    panel[_this] = (0.5 * panel[_this] + 0.5 * tint).astype(np.uint8)
                    Image.fromarray(panel).save(os.path.join(hq_dir, f"bin_{m}.png"))

            print(f"  [HQ] Saved to {hq_dir}")

    # Cleanup
    if do_viz:
        print(f"  [DBG] loop finished, frame_buffer={len(frame_buffer)}, depth_buffer={len(depth_buffer)}, warp_debug_buffer={len(warp_debug_buffer)}")
        if frame_buffer:
            stack_path = os.path.join(VIZ_FOLDER, "summary_stack.png")
            cv2.imwrite(stack_path, np.vstack(frame_buffer))
            print(f"Stack saved to {stack_path}")
        if depth_buffer:
            depth_stack_path = os.path.join(VIZ_FOLDER, "depth_stack.png")
            cv2.imwrite(depth_stack_path, np.vstack(depth_buffer))
            print(f"Depth stack saved to {depth_stack_path}")
        if warp_debug_buffer:
            warp_stack_path = os.path.join(VIZ_FOLDER, "warp_debug_stack.png")
            cv2.imwrite(warp_stack_path, np.vstack(warp_debug_buffer))
            print(f"Warp debug stack saved to {warp_stack_path}")
        if hasattr(main_video_pipeline, '_mvs_depth_buffer') and main_video_pipeline._mvs_depth_buffer:
            mvs_stack_path = os.path.join(VIZ_FOLDER, "depth_stack_mvs.png")
            cv2.imwrite(mvs_stack_path, np.vstack(main_video_pipeline._mvs_depth_buffer))
            print(f"MVS depth stack saved to {mvs_stack_path}")
            main_video_pipeline._mvs_depth_buffer = []
        if hasattr(main_video_pipeline, '_compact_buffer') and main_video_pipeline._compact_buffer:
            compact_path = os.path.join(VIZ_FOLDER, "compact_stack.png")
            cv2.imwrite(compact_path, np.vstack(main_video_pipeline._compact_buffer))
            print(f"Compact stack saved to {compact_path}")
            main_video_pipeline._compact_buffer = []

    # Per-scene metric averages
    scene_avg = {}
    if frame_metrics:
        for key in frame_metrics[0]:
            scene_avg[key] = sum(m[key] for m in frame_metrics) / len(frame_metrics)
        print(f"  [SCENE AVG] {len(frame_metrics)} frames: PSNR={scene_avg['psnr']:.2f} SSIM={scene_avg['ssim']:.4f} LPIPS={scene_avg['lpips']:.4f}")
    return scene_avg, pipe


# NOTE: the standalone __main__ batch runner from the internal repo was removed
# for the public release. Use src/evaluate.py as the entry point;
# it imports main_video_pipeline() from this module.
