"""
Unified image-similarity metric suite for evaluation scripts.

Provides PSNR, SSIM, LPIPS, DreamSim, and DINOv2 cls-token cosine distance.
All pred/GT pairs are assumed already matched in resolution.

Usage:
    from metrics import MetricSuite, compute_all
    suite = MetricSuite(device="cuda:0")
    metrics = compute_all(pred_pil, gt_pil, suite)  # dict of floats

Notes:
- PSNR, SSIM, LPIPS, DreamSim, and DINOv2 all run on `device`.
- DINOv2 model is `vit_small_patch14_dinov2` via timm (384-dim cls).
- Metric is `1 - cos_sim` so larger = more dissimilar (consistent with LPIPS / DreamSim).
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class MetricSuite:
    """Holds metric model instances. Create once per scene; reuse across frames."""

    # Shared class-level singletons for heavy models — one load per process
    _dreamsim_cached = None
    _dinov2_cached = None

    def __init__(self, device: str = "cuda:0", dreamsim_cache_dir: str | None = None,
                 eager: bool = True):
        self.device = device
        # Lightweight metrics — instantiate per-suite
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

        # DreamSim (lazy, cache per-process)
        self._dreamsim_cache_dir = (
            dreamsim_cache_dir
            or os.environ.get("DREAMSIM_CACHE")
            or os.path.expanduser("~/.cache/dreamsim")
        )

        # Eager-load heavy models so any sys.path shenanigans happen here
        # and not mid-evaluation.
        if eager:
            self._get_dreamsim()
            self._get_dinov2()

    # ------------- lazy heavy models -------------

    def _get_dreamsim(self):
        if MetricSuite._dreamsim_cached is None:
            # DINO's hub code does `from utils import trunc_normal_` — this
            # collides with our local `examples/gsplat/utils.py` when it's on
            # sys.path. Temporarily strip our gsplat paths before loading so
            # the DINO hub `utils.py` resolves first.
            import sys, os as _os
            # Strip this repo's examples/gsplat and src dirs (which each contain a
            # `utils.py`) from sys.path so DINO's hub `from utils import ...` resolves
            # to its own bundled utils rather than ours. Repo-relative, not hardcoded.
            _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            _bad_prefixes = (
                _os.path.join(_repo_root, "examples"),
                _os.path.join(_repo_root, "src"),
            )
            _saved_path = list(sys.path)
            sys.path = [p for p in sys.path if not any(_os.path.abspath(p).startswith(b) for b in _bad_prefixes)]
            _utils_mod = sys.modules.pop("utils", None)
            try:
                from dreamsim import dreamsim
                model, preprocess = dreamsim(
                    pretrained=True,
                    cache_dir=self._dreamsim_cache_dir,
                    device=self.device,
                )
            finally:
                sys.path = _saved_path
                if _utils_mod is not None:
                    sys.modules["utils"] = _utils_mod
            MetricSuite._dreamsim_cached = (model, preprocess)
        return MetricSuite._dreamsim_cached

    def _get_dinov2(self):
        if MetricSuite._dinov2_cached is None:
            import timm
            m = timm.create_model(
                "vit_small_patch14_dinov2", pretrained=True, num_classes=0
            )
            m.eval().to(self.device)
            # DINOv2 ViT-S uses 14x14 patches and expects ImageNet normalization
            # on 518x518 inputs (224/16 * 14 = 196 tokens plus cls). Any multiple
            # of 14 works; we'll resize to 518x518 which is the canonical size.
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            MetricSuite._dinov2_cached = (m, mean, std)
        return MetricSuite._dinov2_cached


def _pil_to_t(pil: Image.Image, device: str) -> torch.Tensor:
    """PIL → (1, 3, H, W) float in [0, 1] on device."""
    a = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


@torch.no_grad()
def _dinov2_cls_feature(img_t: torch.Tensor, suite: MetricSuite) -> torch.Tensor:
    """Return DINOv2 ViT-S cls-token feature (1, 384) for a (1, 3, H, W) tensor in [0,1]."""
    model, mean, std = suite._get_dinov2()
    # Resize to 518x518 (canonical DINOv2 input; 518 = 37 * 14)
    x = F.interpolate(img_t, size=(518, 518), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    feat = model(x)  # (1, 384) when num_classes=0 (cls token pre-head)
    return feat


@torch.no_grad()
def compute_all(
    pred_pil: Image.Image,
    gt_pil: Image.Image,
    suite: MetricSuite,
) -> dict:
    """Compute the full metric set. Pred and GT must match in size."""
    device = suite.device
    # Make sure both are at GT resolution — caller should resize beforehand,
    # but do it here for robustness.
    if pred_pil.size != gt_pil.size:
        pred_pil = pred_pil.resize(gt_pil.size, Image.LANCZOS)

    pred_t = _pil_to_t(pred_pil, device)  # (1,3,H,W) [0,1]
    gt_t = _pil_to_t(gt_pil, device)

    # PSNR / SSIM (GPU)
    psnr_val = float(suite.psnr(pred_t, gt_t).item())
    ssim_val = float(suite.ssim(pred_t, gt_t).item())
    suite.psnr.reset()
    suite.ssim.reset()

    # LPIPS on GPU (if the metric instance is there) — normalize=True expects [0,1]
    lpips_val = float(suite.lpips(pred_t, gt_t).item())

    # DreamSim — its preprocess expects a PIL, but a tensor [0,1] path works when we bypass it.
    ds_model, ds_preprocess = suite._get_dreamsim()
    pred_ds = ds_preprocess(pred_pil).to(device)
    gt_ds = ds_preprocess(gt_pil).to(device)
    dreamsim_val = float(ds_model(pred_ds, gt_ds).item())

    # DINOv2 cls-cosine distance (1 - cos_sim); both features L2-normalized internally.
    f_pred = _dinov2_cls_feature(pred_t, suite)
    f_gt = _dinov2_cls_feature(gt_t, suite)
    cos = F.cosine_similarity(f_pred, f_gt, dim=1).item()
    dinov2_cos_dist = float(1.0 - cos)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
        "dreamsim": dreamsim_val,
        "dinov2": dinov2_cos_dist,
    }


# Display helpers — keep the table formatting consistent across scripts
METRIC_ORDER = ["psnr", "ssim", "lpips", "dreamsim", "dinov2"]
METRIC_HEADERS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "dreamsim": "DreamSim",
    "dinov2": "DINOv2",
}
METRIC_FORMATS = {
    "psnr": "{:.2f}",
    "ssim": "{:.4f}",
    "lpips": "{:.4f}",
    "dreamsim": "{:.4f}",
    "dinov2": "{:.4f}",
}
