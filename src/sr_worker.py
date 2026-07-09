"""Batch SR worker — runs either PFT-SR (default) or Real-ESRGAN on a
dedicated GPU for all crops in a directory.

CLI:
  python sr_worker.py --input_dir <dir> --output_dir <dir>
                      --target_w W --target_h H
                      [--backend pft|esrgan]

For backend=esrgan we use RealESRGANer from the `realesrgan` pip package with
the default RealESRGAN_x4plus 4× RRDBNet. The model supports a float
`outscale` arg that resizes internally, so there is no need for the
"iterate x4/x2 until we exceed target" loop that PFT-SR required.

We pick `outscale = max(target_w/w, target_h/h)` so the upscaled image
is at least as big as the target, then LANCZOS to the exact target size.
"""
import argparse
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


# ============================================================
# PFT-SR backend (legacy default)
# ============================================================

def _load_pft_models():
    from basicsr.archs.pft_arch import PFT
    # PFT-SR weights (103_PFT_light_SRx4_finetune.pth, 101_PFT_light_SRx2_scratch.pth).
    # Download from the release Google Drive and point PFT_SR_WEIGHTS at the folder,
    # or place them under ./pft_sr_weights/ at the repo root.
    PFT_SR_PATH = os.environ.get(
        "PFT_SR_WEIGHTS",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pft_sr_weights"),
    )
    if not os.path.isdir(PFT_SR_PATH):
        raise FileNotFoundError(
            f"PFT-SR weights folder not found at '{PFT_SR_PATH}'. "
            f"Set the PFT_SR_WEIGHTS env var to the folder containing "
            f"103_PFT_light_SRx4_finetune.pth and 101_PFT_light_SRx2_scratch.pth."
        )
    cfg = dict(
        embed_dim=52, depths=[2, 4, 6, 6, 6], num_heads=4,
        num_topk=[1024, 1024, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128,
                  64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32],
        window_size=32, convffn_kernel_size=7, mlp_ratio=1,
        upsampler='pixelshuffledirect', use_checkpoint=False,
    )
    model_x4 = PFT(upscale=4, **cfg).cuda()
    sd = torch.load(f'{PFT_SR_PATH}/103_PFT_light_SRx4_finetune.pth', map_location='cuda')['params_ema']
    model_x4.load_state_dict(sd, strict=True); model_x4.eval()

    model_x2 = PFT(upscale=2, **cfg).cuda()
    sd2 = torch.load(f'{PFT_SR_PATH}/101_PFT_light_SRx2_scratch.pth', map_location='cuda')['params_ema']
    model_x2.load_state_dict(sd2, strict=True); model_x2.eval()
    return model_x4, model_x2


def superres_pft(img_pil, target_w, target_h, model_x4, model_x2):
    """Iterative x4/x2 upsample until image >= target, then LANCZOS-resize."""
    crop_np = np.array(img_pil)
    valid_mask = (crop_np.sum(axis=2) > 0).astype(np.uint8)
    img_t = TF.to_tensor(img_pil).unsqueeze(0).cuda()
    mask_t = torch.from_numpy(valid_mask).float().unsqueeze(0).unsqueeze(0).cuda()

    MAX_FOR_X4 = 512
    MAX_SR_SIDE = 1500

    with torch.no_grad():
        while img_t.shape[-1] < target_w or img_t.shape[-2] < target_h:
            h, w = img_t.shape[-2], img_t.shape[-1]
            if max(h, w) > MAX_SR_SIDE:
                break
            if h <= MAX_FOR_X4 and w <= MAX_FOR_X4:
                img_t = model_x4(img_t).clamp(0.0, 1.0)
            else:
                img_t = model_x2(img_t).clamp(0.0, 1.0)
            mask_t = F.interpolate(mask_t, size=(img_t.shape[-2], img_t.shape[-1]), mode='nearest')

    img_t = img_t * (mask_t > 0.5).float()
    sr_pil = TF.to_pil_image(img_t.squeeze(0).cpu())
    return sr_pil.resize((target_w, target_h), Image.LANCZOS)


# ============================================================
# Real-ESRGAN backend
# ============================================================

def _load_esrgan_model(half: bool = False):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # RealESRGAN_x4plus: 4× RRDBNet, 23 blocks, 64 feat, 32 grow_ch.
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    url = ('https://github.com/xinntao/Real-ESRGAN/releases/download/'
           'v0.1.0/RealESRGAN_x4plus.pth')
    upsampler = RealESRGANer(
        scale=4, model_path=url, model=model,
        tile=0, tile_pad=10, pre_pad=10,
        half=half, device='cuda',
    )
    return upsampler


def superres_esrgan(img_pil, target_w, target_h, upsampler, force_run: bool = False):
    """Run RealESRGAN with a float outscale that guarantees >= target,
    then LANCZOS-resize to exactly (target_w, target_h).

    When `force_run=True`, ESRGAN always runs (even if the input is already
    at or above target) — used for the post-processing sharpen pass that
    upscales then downsamples to restore detail.
    """
    w0, h0 = img_pil.size
    if (w0 >= target_w and h0 >= target_h) and not force_run:
        # Already at or above target; just downsample with LANCZOS.
        return img_pil.resize((target_w, target_h), Image.LANCZOS)

    if force_run:
        # Net's native 4× scale; downsample back to target after.
        outscale = 4.0
    else:
        outscale = max(target_w / w0, target_h / h0)
    img_bgr = np.array(img_pil)[:, :, ::-1].copy()   # RGB→BGR uint8
    out_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale)
    out_rgb = out_bgr[:, :, ::-1]                     # BGR→RGB
    sr_pil = Image.fromarray(out_rgb)
    return sr_pil.resize((target_w, target_h), Image.LANCZOS)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with crop PNGs')
    parser.add_argument('--output_dir', required=True, help='Directory for SR output PNGs')
    parser.add_argument('--target_w', type=int, required=True)
    parser.add_argument('--target_h', type=int, required=True)
    parser.add_argument('--backend', choices=['pft', 'esrgan'], default='pft',
                        help='SR backend. Default: pft (legacy PFT-SR).')
    parser.add_argument('--force-run', action='store_true',
                        help='For esrgan backend: always run the model even if '
                             'the input is already >= target (used for the '
                             'sharpen-then-downsample post-pass).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.backend == 'pft':
        model_x4, model_x2 = _load_pft_models()
    else:
        upsampler = _load_esrgan_model()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.png')])
    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        crop_pil = Image.open(in_path).convert('RGB')
        if args.backend == 'pft':
            sr_pil = superres_pft(crop_pil, args.target_w, args.target_h, model_x4, model_x2)
        else:
            sr_pil = superres_esrgan(crop_pil, args.target_w, args.target_h,
                                      upsampler, force_run=args.force_run)
        sr_pil.save(out_path)
        torch.cuda.empty_cache()

    print(f"SR done ({args.backend}): {len(files)} crops processed")


if __name__ == '__main__':
    main()
