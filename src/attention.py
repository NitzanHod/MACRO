"""
Depth-aware cross-view attention (the core of MACRO / "ECA").

`EpipolarMixingBlock` wraps a DiFix self-attention module and adds a mask that
lets each close-up token attend to the reference-crop tokens of *its own depth
plane*. The mask is built (in `_build_input_query_mask` / `_forward_split`)
from the per-token bin assignment `latent_bin_map` / `ref_to_bin` plus an
occlusion mask — this is what injects scale-matched reference detail into the
close-up render during the single DiFix step.

Legacy note: the module also contains an epipolar-line masking path
(`need_epipolar`, using the fundamental matrices `F_effs` / `get_effective_F`)
for `mask_mode="epipolar"`. That mode is NOT in the public config set (macro
uses `mask_mode="all"`), so on the macro path F is only consumed by the
visualization dashboards, not by the enhancement output.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2
import numpy as np
import os
import copy


# Module-level dedup: track which (frame_tag, resolution) combos we've already
# viz'd, so that across the MANY EpipolarMixingBlock instances that make up the
# UNet (one per attention layer), each frame is viz'd exactly ONCE at the
# preferred resolution. Dashboarding is expensive (several seconds) and without
# this gate each frame would be re-viz'd at every Unet layer.
_ECA_VIZ_FRAMES_DONE = set()

# Resolution we prefer to viz at (the highest-resolution non-downsampled layer
# that the UNet produces). Picking a fixed target avoids non-deterministic
# first-come-first-serve behavior where a small layer wins the dedup race.
_ECA_VIZ_PREFERRED_H = None  # auto-discovered on first call (max H seen)
_ECA_VIZ_CANDIDATE_FRAMES = {}  # frame_tag -> (max_H_seen, True/False written)


# Optional custom query-point override. If env ATTN_VIZ_POINTS_JSON is set to
# a JSON path, the dashboard will use those UV coords instead of auto-sampled
# points for any frame whose stem appears in the JSON. Schema:
#   {"<frame_stem>": {"points": [{"u": 0.23, "v": 0.47}, ...], ...}, ...}
_ATTN_VIZ_POINTS = None
_ATTN_VIZ_POINTS_LOADED = False


def _load_viz_points():
    global _ATTN_VIZ_POINTS, _ATTN_VIZ_POINTS_LOADED
    if _ATTN_VIZ_POINTS_LOADED:
        return _ATTN_VIZ_POINTS
    _ATTN_VIZ_POINTS_LOADED = True
    p = os.environ.get("ATTN_VIZ_POINTS_JSON")
    if not p or not os.path.exists(p):
        return None
    try:
        import json as _json
        with open(p) as _f:
            _ATTN_VIZ_POINTS = _json.load(_f)
        print(f"[ECA VIZ] loaded custom points: {p}  frames={list(_ATTN_VIZ_POINTS.keys())}")
    except Exception as _e:
        print(f"[ECA VIZ] failed to load {p}: {_e}")
        _ATTN_VIZ_POINTS = None
    return _ATTN_VIZ_POINTS

# Optional flex_attention (PyTorch 2.5+). Falls back to masked SDPA if unavailable.
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX_ATTENTION = True
    # Compile flex_attention to get the Triton/inductor kernel (block-sparse, fast).
    # Without compilation, flex_attention dispatches to a dense math reference that
    # materializes the full (B, H, Q, KV) score tensor → OOM on our workloads.
    # dynamic=True because Q_LEN/KV_LEN vary across UNet layers (68x120, 34x60, ...).
    _flex_attention_compiled = torch.compile(flex_attention, dynamic=True)
except ImportError:
    _HAS_FLEX_ATTENTION = False
    _flex_attention_compiled = None


def get_effective_F(F_norm, zoom, ref_crop_bbox=None):
    F_eff = F_norm
    if zoom != 1.0:
        inv_z = 1.0 / zoom
        offset = 0.5 - 0.5 * inv_z
        T_zoom = torch.tensor([[inv_z, 0., offset], [0., inv_z, offset], [0., 0., 1.]],
                               device=F_norm.device, dtype=F_norm.dtype)
        F_eff = F_eff @ T_zoom
    if ref_crop_bbox is not None:
        u_min, v_min, u_max, v_max = ref_crop_bbox
        du = u_max - u_min
        dv = v_max - v_min
        T_ref = torch.tensor([[du, 0., u_min], [0., dv, v_min], [0., 0., 1.]],
                              device=F_norm.device, dtype=F_norm.dtype)
        F_eff = T_ref.T @ F_eff
    return F_eff


class EpipolarMixingBlock(nn.Module):
    def __init__(self, source_attn_module, threshold=0.03, viz_folder=None):
        super().__init__()
        self.to_q = copy.deepcopy(source_attn_module.to_q)
        self.to_k = copy.deepcopy(source_attn_module.to_k)
        self.to_v = copy.deepcopy(source_attn_module.to_v)
        self.to_out = copy.deepcopy(source_attn_module.to_out)
        self.orig_attn1 = source_attn_module
        self.heads = source_attn_module.heads
        self.scale = getattr(source_attn_module, 'scale', 1.0 / (getattr(source_attn_module, 'dim_head', 64) ** 0.5))

        self.threshold = threshold
        self.viz_folder = viz_folder
        self.debug_counter = 0
        self._block_mask_cache = {}

        # --- Control parameters ---
        # cross_attn_mode:
        #   'off'    → per-view self-attention, no cross-view info
        #   'native' → symmetric SA over all (K+1)*N tokens (original DiFix rearrange)
        #   <float>  → symmetric flex_attention with additive score bias on cross-view tokens
        self.cross_attn_mode = 'native'
        self.mask_mode = 'epipolar'  # 'epipolar' or 'all'
        self.attention_mode = 'full'  # 'full' (chunked SDPA with full mask), 'split' (per-bin ref SA + masked input SDPA), or 'flex' (per-bin ref SA + flex_attention input)

        self._last_viz_frame = None
        self.is_down = False

    def _get_beta(self):
        """Resolve cross_attn_mode to a numeric beta value (or None for 'off')."""
        if self.cross_attn_mode == 'off':
            return None
        if self.cross_attn_mode in ('native', 'dampen'):
            return 0.0
        return float(self.cross_attn_mode)

    def forward(self, hidden_states, epipolar_cache, warp_metadata,
                encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        v2 batch-of-(K+1) forward.
        hidden_states: (K+1, N, D) — batch 0 = input, batch 1..K = refs.

        cross_attn_mode controls behavior:
          'off'    → per-view self-attention, no cross-view info
          'native' → symmetric SA over all (K+1)*N tokens (original DiFix rearrange)
                     mask_mode='all': fast path via orig_attn1 rearrange
                     mask_mode='epipolar': chunked SDPA with explicit epipolar attn_mask
          <float>  → symmetric chunked SDPA with additive score bias on cross-view tokens
                     and optional epipolar masking
        """
        B, N, D = hidden_states.shape
        device = hidden_states.device
        K = B - 1

        beta = self._get_beta()

        # --- No refs or CA disabled: per-view self-attention for all ---
        if K == 0 or beta is None:
            return self.orig_attn1(hidden_states, encoder_hidden_states=None,
                                   attention_mask=None, **kwargs)

        # --- Native mode ---
        if self.cross_attn_mode in ('native', 'dampen'):
            has_bins = warp_metadata.get('latent_bin_map', None) is not None
            if self.mask_mode in ('all', 'none') and not has_bins and self.cross_attn_mode != 'dampen':
                # Full symmetric SA over all (K+1)*N tokens (original DiFix rearrange)
                merged = rearrange(hidden_states, "b n d -> 1 (b n) d")
                # --- Optional viz before returning (for DiFix no-bin path) ---
                if self.viz_folder is not None and epipolar_cache is not None:
                    _frame_tag = getattr(epipolar_cache, 'frame_name', None)
                    # Compute H_in, W_in from warp_metadata like the chunked path.
                    try:
                        _H_in, _W_in = warp_metadata['S_to_HW'][N * 2]
                        _H_in, _W_in = int(_H_in), int(_W_in)
                    except Exception:
                        _H_in, _W_in = 0, 0
                    if os.environ.get("ECA_VIZ_DEBUG"):
                        print(f"[ECA VIZ GATE fast-path] ft={_frame_tag} H={_H_in} is_down={self.is_down} done={_frame_tag in _ECA_VIZ_FRAMES_DONE}")
                    _all_layers = os.environ.get("ECA_VIZ_CACHE_ALL_LAYERS", "").strip() not in ("", "0", "false")
                    _include_down = os.environ.get("ECA_VIZ_ALL_LAYERS_INCLUDE_DOWN", "").strip() not in ("", "0", "false")
                    _gate_ok = (
                        _frame_tag is not None and _H_in > 8
                        and (_include_down or not self.is_down)
                        and (_all_layers or _frame_tag not in _ECA_VIZ_FRAMES_DONE)
                    )
                    if _gate_ok:
                        if not _all_layers:
                            _ECA_VIZ_FRAMES_DONE.add(_frame_tag)
                        self._last_viz_frame = _frame_tag
                        _block_name = getattr(self, 'name', None) or f"lid{id(self) & 0xFFFF:04x}"
                        _layer_tag = _block_name.replace('.', '_').replace('/', '_')[:60]
                        _dbg_seq = self.debug_counter
                        self.debug_counter += 1
                        print(f"[ECA VIZ] {_frame_tag} L{_H_in} K={K} mode=native beta=0.00 mask={self.mask_mode} (fast-path) block={_layer_tag} seq={_dbg_seq}")
                        # Manually compute Q/K using orig_attn1's projections so
                        # the dashboard can run. Use the same merged input.
                        _to_q = self.orig_attn1.to_q
                        _to_k = self.orig_attn1.to_k
                        _q_all = rearrange(_to_q(merged), "1 s (h d) -> 1 h s d", h=self.heads)
                        _k_all = rearrange(_to_k(merged), "1 s (h d) -> 1 h s d", h=self.heads)
                        _epi_caches = warp_metadata.get('epipolar_caches', [epipolar_cache])
                        _F_effs = [get_effective_F(c.F_norm, c.zoom, c.ref_crop_bbox) for c in _epi_caches]
                        _q_in = _q_all[:, :, :N]
                        self._save_dashboard(
                            _q_in, _k_all, _H_in, _W_in, N,
                            _F_effs, _epi_caches, N, 0.0,
                            filename=f"debug_{_frame_tag}_L{_H_in}_{_layer_tag}_s{_dbg_seq}.jpg",
                            ref_to_bin=None, latent_bin_map=None, K_orig=None,
                            warp_maps=None, warp_patch_size=64, warp_forced=False,
                            occ_masks=None,
                        )
                # --- End viz ---
                merged = self.orig_attn1(merged, encoder_hidden_states=None,
                                         attention_mask=None, **kwargs)
                return rearrange(merged, "1 (b n) d -> b n d", b=B)
            elif self.mask_mode == 'none' and self.cross_attn_mode != 'dampen':
                # 'none' with bins: explicitly no mask, use chunked SDPA unmasked
                beta = 0.0
            else:
                # native/dampen + any mask: symmetric flex_attention with beta=0
                beta = 0.0

        # --- Symmetric chunked SDPA path (native+epipolar or float modes) ---
        # Project all views, merge into single sequence
        merged = rearrange(hidden_states, "b n d -> 1 (b n) d")  # (1, (K+1)*N, D)
        total_len = B * N

        to_q = self.orig_attn1.to_q
        to_k = self.orig_attn1.to_k
        to_v = self.orig_attn1.to_v
        to_out = self.orig_attn1.to_out

        q_all = rearrange(to_q(merged), "1 s (h d) -> 1 h s d", h=self.heads)
        k_all = rearrange(to_k(merged), "1 s (h d) -> 1 h s d", h=self.heads)
        v_all = rearrange(to_v(merged), "1 s (h d) -> 1 h s d", h=self.heads)

        # Get spatial dims from warp_metadata
        H_in, W_in = warp_metadata['S_to_HW'][N * 2]
        H_in, W_in = int(H_in), int(W_in)

        # Get epipolar caches and F_effs
        epipolar_caches = warp_metadata.get('epipolar_caches', [epipolar_cache])
        F_effs = [get_effective_F(c.F_norm, c.zoom, c.ref_crop_bbox) for c in epipolar_caches]

        # Build attn_mask and score_bias for the full sequence (cached)
        crop_bboxes = tuple(c.ref_crop_bbox for c in epipolar_caches)
        ref_to_bin = warp_metadata.get('ref_to_bin', None)
        latent_bin_map = warp_metadata.get('latent_bin_map', None)
        occ_masks = warp_metadata.get('occ_masks', None) if warp_metadata.get('occ_mask_enabled', False) else None
        warp_maps = warp_metadata.get('warp_maps', None) if self.mask_mode == 'warp' else None
        warp_patch_size = warp_metadata.get('warp_patch', 64)
        warp_forced = warp_metadata.get('warp_forced', False)
        input_local_patch = warp_metadata.get('input_local_patch', 0)
        ref_boost = warp_metadata.get('ref_boost', 0.0)
        conf_mask = warp_metadata.get('confidence_mask', None) if warp_metadata.get('conf_mask_enabled', False) else None
        block_ref_to_input = warp_metadata.get('block_ref_to_input', False)
        has_bins = ref_to_bin is not None and latent_bin_map is not None

        # =====================================================================
        # SPLIT/FLEX ATTENTION PATH: per-bin ref SA (unmasked) + input queries
        #   - split: input queries use F.scaled_dot_product_attention with dense mask
        #   - flex:  input queries use flex_attention (block-sparse, faster)
        # Both paths share identical Phase 1 (ref SA) and identical mask construction.
        # =====================================================================
        if self.attention_mode in ('split', 'flex') and has_bins:
            use_flex = (self.attention_mode == 'flex') and _HAS_FLEX_ATTENTION
            attn_output = self._forward_split(
                q_all, k_all, v_all, N, K, B, total_len, beta, device,
                H_in, W_in, ref_to_bin, latent_bin_map,
                F_effs, epipolar_caches, occ_masks, warp_maps, conf_mask,
                warp_patch_size, warp_forced, input_local_patch, ref_boost,
                warp_metadata, use_flex=use_flex
            )
            # --- VISUALIZATION (same as full path) ---
            frame_tag = None
            if epipolar_caches:
                frame_tag = getattr(epipolar_caches[0], 'frame_name', None)
            _all_layers = os.environ.get("ECA_VIZ_CACHE_ALL_LAYERS", "").strip() not in ("", "0", "false")
            _include_down = os.environ.get("ECA_VIZ_ALL_LAYERS_INCLUDE_DOWN", "").strip() not in ("", "0", "false")
            should_viz = (
                self.viz_folder is not None and frame_tag is not None
                and H_in > 8
                and (_include_down or not self.is_down)
                and (_all_layers or frame_tag not in _ECA_VIZ_FRAMES_DONE)
            )
            if should_viz:
                if not _all_layers:
                    _ECA_VIZ_FRAMES_DONE.add(frame_tag)
                self._last_viz_frame = frame_tag
                mode_str = self.cross_attn_mode if isinstance(self.cross_attn_mode, str) else f"{self.cross_attn_mode:.2f}"
                attn_tag = 'flex' if use_flex else 'split'
                _block_name = getattr(self, 'name', None) or f"lid{id(self) & 0xFFFF:04x}"
                _layer_tag = _block_name.replace('.', '_').replace('/', '_')[:60]
                print(f"[ECA VIZ] {frame_tag} L{H_in} K={K} mode={mode_str} beta={beta:.2f} mask={self.mask_mode} attn={attn_tag} block={_layer_tag}")
                q_in = q_all[:, :, :N]
                self._save_dashboard(
                    q_in, k_all, H_in, W_in, N,
                    F_effs, epipolar_caches, N, beta,
                    filename=f"debug_{frame_tag}_L{H_in}_{_layer_tag}.jpg",
                    ref_to_bin=ref_to_bin,
                    latent_bin_map=latent_bin_map,
                    K_orig=warp_metadata.get('K_orig', None),
                    warp_maps=warp_maps,
                    warp_patch_size=warp_patch_size,
                    warp_forced=warp_forced,
                    occ_masks=occ_masks,
                )
            self.debug_counter += 1

            attn_output = rearrange(attn_output, "1 h (b n) d -> b n (h d)", b=B)
            attn_output = to_out[0](attn_output)
            return attn_output

        # =====================================================================
        # FULL ATTENTION PATH (original): build full mask, chunked SDPA
        # =====================================================================
        bin_key = tuple(ref_to_bin) if ref_to_bin else None
        occ_key = 'occ' if occ_masks is not None else None
        warp_key = 'warp' if warp_maps is not None else None
        conf_key = 'conf' if conf_mask is not None else None
        bri_key = 'bri' if block_ref_to_input else None
        mask_cache_key = (H_in, W_in, N, K, self.mask_mode, crop_bboxes, bin_key, occ_key, warp_key, conf_key, bri_key)
        # Warp/conf maps change per frame — don't cache
        no_cache = warp_maps is not None or conf_mask is not None
        if no_cache or mask_cache_key not in self._block_mask_cache:
            attn_mask, score_bias = self._build_symmetric_attn_mask(
                H_in, W_in, N, F_effs, K, B, total_len, beta, device,
                ref_to_bin=ref_to_bin, latent_bin_map=latent_bin_map,
                occ_masks=occ_masks, warp_maps=warp_maps, conf_mask=conf_mask,
                warp_patch_size=warp_patch_size, warp_forced=warp_forced,
                input_local_patch=input_local_patch,
                block_ref_to_input=block_ref_to_input
            )
            if not no_cache:
                self._block_mask_cache[mask_cache_key] = (attn_mask, score_bias)
        else:
            attn_mask, score_bias = self._block_mask_cache[mask_cache_key]

        # Precompute dampening flag
        is_dampen = self.cross_attn_mode == 'dampen'

        # Chunked SDPA
        CHUNK_SIZE = 16384 * 2
        outputs = []
        for chunk_start in range(0, total_len, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_len)
            q_chunk = q_all[:, :, chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start

            if attn_mask is not None:
                # attn_mask: (total_len, total_len) → chunk rows
                chunk_mask = attn_mask[chunk_start:chunk_end, :].to(device)  # (chunk, total_len)
                # SDPA expects (B, H, Q, KV) or broadcastable
                chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, chunk, total_len)
            else:
                chunk_mask = None

            # Build float attention bias
            chunk_attn = None
            if score_bias is not None:
                chunk_bias = score_bias[chunk_start:chunk_end, :]
                chunk_bias = chunk_bias.unsqueeze(0).unsqueeze(0)
                if chunk_mask is not None:
                    chunk_attn = torch.where(chunk_mask, chunk_bias, torch.tensor(float('-inf'), device=device))
                else:
                    chunk_attn = chunk_bias
            elif chunk_mask is not None:
                # Convert bool mask to float: True→0, False→-inf
                chunk_attn = torch.where(chunk_mask, torch.tensor(0.0, device=device),
                                         torch.tensor(float('-inf'), device=device))

            # Apply dampening: for input query tokens, add log(N_ref/N_in) to input columns
            if is_dampen and attn_mask is not None and chunk_start < N:
                if chunk_attn is None:
                    chunk_attn = torch.zeros(1, 1, chunk_len, total_len, device=device)
                # Only input tokens (global index < N) get dampened
                n_input_in_chunk = min(N - chunk_start, chunk_len)
                # Count visible input/ref tokens per input query (vectorized)
                input_rows = attn_mask[chunk_start:chunk_start + n_input_in_chunk, :]  # (n, total_len)
                n_in_per_q = input_rows[:, :N].sum(dim=1).float()  # (n,)
                n_ref_per_q = input_rows[:, N:].sum(dim=1).float()  # (n,)
                # Compute log ratio, guard against zero
                valid = (n_in_per_q > 0) & (n_ref_per_q > 0)
                dampen_vals = torch.zeros(n_input_in_chunk, device='cpu')
                dampen_vals[valid] = torch.log(n_ref_per_q[valid] / n_in_per_q[valid])
                # Add to input columns for these query rows
                chunk_attn[0, 0, :n_input_in_chunk, :N] += dampen_vals.to(device).unsqueeze(1)

            # Apply ref boost: add positive bias to ref columns for input query rows
            if ref_boost > 0 and chunk_start < N:
                if chunk_attn is None:
                    chunk_attn = torch.zeros(1, 1, chunk_len, total_len, device=device)
                n_input_in_chunk = min(N - chunk_start, chunk_len)
                chunk_attn[0, 0, :n_input_in_chunk, N:] += ref_boost

            if chunk_attn is None and chunk_mask is not None:
                chunk_attn = chunk_mask

            # Ensure attn bias dtype matches query dtype (e.g. float16 under AMP)
            if chunk_attn is not None and chunk_attn.dtype != q_chunk.dtype:
                chunk_attn = chunk_attn.to(dtype=q_chunk.dtype)

            out_chunk = F.scaled_dot_product_attention(
                q_chunk, k_all, v_all,
                attn_mask=chunk_attn,
                scale=self.scale
            )
            outputs.append(out_chunk)

        attn_output = torch.cat(outputs, dim=2)  # (1, h, (K+1)*N, d)

        # --- VISUALIZATION ---
        frame_tag = None
        if epipolar_caches:
            frame_tag = getattr(epipolar_caches[0], 'frame_name', None)
        # Process-global dedup: one dashboard per frame_tag, regardless of
        # which EpipolarMixingBlock / layer / resolution is currently running.
        # Also skip tiny layers (H_in <= 34) where the heatmap is too coarse.
        viz_key = (frame_tag, int(H_in))
        should_viz = (
            self.viz_folder is not None and frame_tag is not None
            and H_in > 34 and not self.is_down
            and frame_tag not in _ECA_VIZ_FRAMES_DONE
        )
        if os.environ.get("ECA_VIZ_DEBUG"):
            print(f"[ECA VIZ GATE full-attn] ft={frame_tag} H={H_in} is_down={self.is_down} vf={self.viz_folder is not None} done={frame_tag in _ECA_VIZ_FRAMES_DONE} -> {should_viz}")
        if should_viz:
            _ECA_VIZ_FRAMES_DONE.add(frame_tag)
            self._last_viz_frame = frame_tag
            mode_str = self.cross_attn_mode if isinstance(self.cross_attn_mode, str) else f"{self.cross_attn_mode:.2f}"
            print(f"[ECA VIZ] {frame_tag} L{H_in} K={K} mode={mode_str} beta={beta:.2f} mask={self.mask_mode}")
            # Viz uses input view's Q and full KV
            q_in = q_all[:, :, :N]
            self._save_dashboard(
                q_in, k_all, H_in, W_in, N,
                F_effs, epipolar_caches, N, beta,
                filename=f"debug_{frame_tag}_L{H_in}.jpg",
                ref_to_bin=ref_to_bin,
                latent_bin_map=latent_bin_map,
                K_orig=warp_metadata.get('K_orig', None),
                warp_maps=warp_maps,
                warp_patch_size=warp_patch_size,
                warp_forced=warp_forced,
                occ_masks=occ_masks,
            )
        self.debug_counter += 1

        attn_output = rearrange(attn_output, "1 h (b n) d -> b n (h d)", b=B)
        attn_output = to_out[0](attn_output)
        return attn_output

    def _forward_split(self, q_all, k_all, v_all, N, K, B, total_len, beta, device,
                        H_in, W_in, ref_to_bin, latent_bin_map,
                        F_effs, epipolar_caches, occ_masks, warp_maps, conf_mask,
                        warp_patch_size, warp_forced, input_local_patch, ref_boost,
                        warp_metadata, use_flex=False):
        """
        Split attention: per-bin ref self-attention (unmasked) + input queries (masked).

        Phase 1: For each bin m, gather the K_m ref views in that bin.
                 Run unmasked SDPA on their stacked Q/K/V. No mask needed.
        Phase 2: Input view queries (N tokens) attend to full KV sequence
                 with the standard bin routing + epipolar/occ/warp/conf mask,
                 but only N query rows instead of total_len.

                 When use_flex=True, Phase 2 uses flex_attention with a BlockMask
                 built from the same (N, total_len) boolean mask. This gives
                 block-sparse attention that skips fully-masked blocks, much
                 faster than dense-mask SDPA. Numerical output is identical
                 up to fp rounding (same mask + same score_bias).

        Returns: (1, heads, total_len, head_dim) attention output
        """
        head_dim = q_all.shape[-1]

        # Group ref views by bin
        M = max(ref_to_bin) + 1
        bin_to_refs = [[] for _ in range(M)]
        for r_idx, b in enumerate(ref_to_bin):
            if 0 <= b < M:
                bin_to_refs[b].append(r_idx)

        # Allocate output buffer
        out_all = torch.zeros(1, self.heads, total_len, head_dim, device=device, dtype=q_all.dtype)

        # ---- Phase 1: Per-bin ref self-attention (unmasked) ----
        for m in range(M):
            ref_indices = bin_to_refs[m]
            if not ref_indices:
                continue
            # Gather Q/K/V for refs in this bin
            kv_slices = []
            q_slices = []
            view_ranges = []
            for r_idx in ref_indices:
                s = (r_idx + 1) * N
                e = (r_idx + 2) * N
                q_slices.append(q_all[:, :, s:e])
                kv_slices.append((k_all[:, :, s:e], v_all[:, :, s:e]))
                view_ranges.append((s, e))

            # Stack into single sequence for this bin
            q_bin = torch.cat(q_slices, dim=2)  # (1, h, K_m*N, d)
            k_bin = torch.cat([kv[0] for kv in kv_slices], dim=2)
            v_bin = torch.cat([kv[1] for kv in kv_slices], dim=2)

            # Score bias for cross-view tokens within this bin
            bin_attn_bias = None
            if beta != 0.0:
                K_m = len(ref_indices)
                bin_len = K_m * N
                view_ids = torch.arange(K_m, device=device).unsqueeze(1).expand(K_m, N).reshape(-1)
                is_cross = view_ids.unsqueeze(0) != view_ids.unsqueeze(1)
                bin_attn_bias = torch.where(is_cross,
                    torch.tensor(beta, device=device, dtype=torch.float32),
                    torch.zeros(1, device=device, dtype=torch.float32))
                bin_attn_bias = bin_attn_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, bin_len, bin_len)

            # Unmasked SDPA — this is the fast path
            if bin_attn_bias is not None and bin_attn_bias.dtype != q_bin.dtype:
                bin_attn_bias = bin_attn_bias.to(dtype=q_bin.dtype)
            out_bin = F.scaled_dot_product_attention(
                q_bin, k_bin, v_bin,
                attn_mask=bin_attn_bias,
                scale=self.scale
            )

            # Write outputs back to correct positions
            offset = 0
            for s, e in view_ranges:
                out_all[:, :, s:e] = out_bin[:, :, offset:offset + N]
                offset += N

        # ---- Phase 2: Input queries with masked cross-attention ----
        q_input = q_all[:, :, :N]  # (1, h, N, d)

        # Build input-only mask: (N, total_len)
        input_mask, input_score_bias = self._build_input_query_mask(
            H_in, W_in, N, F_effs, K, B, total_len, beta, device,
            ref_to_bin=ref_to_bin, latent_bin_map=latent_bin_map,
            occ_masks=occ_masks, warp_maps=warp_maps, conf_mask=conf_mask,
            warp_patch_size=warp_patch_size, warp_forced=warp_forced,
            input_local_patch=input_local_patch
        )

        is_dampen = self.cross_attn_mode == 'dampen'

        # Build float attention bias for input queries (SDPA path only — flex path
        # builds its own bias inside _input_queries_flex from the same inputs).
        input_attn = None
        if not use_flex:
            if input_score_bias is not None:
                input_attn = input_score_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, N, total_len)
                if input_mask is not None:
                    chunk_mask = input_mask.to(device).unsqueeze(0).unsqueeze(0)
                    input_attn = torch.where(chunk_mask, input_attn,
                        torch.tensor(float('-inf'), device=device))
            elif input_mask is not None:
                chunk_mask = input_mask.to(device).unsqueeze(0).unsqueeze(0)
                input_attn = torch.where(chunk_mask,
                    torch.tensor(0.0, device=device),
                    torch.tensor(float('-inf'), device=device))

            # Apply dampening
            if is_dampen and input_mask is not None:
                if input_attn is None:
                    input_attn = torch.zeros(1, 1, N, total_len, device=device)
                n_in_per_q = input_mask[:, :N].sum(dim=1).float()
                n_ref_per_q = input_mask[:, N:].sum(dim=1).float()
                valid = (n_in_per_q > 0) & (n_ref_per_q > 0)
                dampen_vals = torch.zeros(N, device='cpu')
                dampen_vals[valid] = torch.log(n_ref_per_q[valid] / n_in_per_q[valid])
                input_attn[0, 0, :, :N] += dampen_vals.to(device).unsqueeze(1)

            # Apply ref boost
            if ref_boost > 0:
                if input_attn is None:
                    input_attn = torch.zeros(1, 1, N, total_len, device=device)
                input_attn[0, 0, :, N:] += ref_boost

        # Run attention for input queries against full KV
        if use_flex:
            out_input = self._input_queries_flex(
                q_input, k_all, v_all, input_mask, input_score_bias,
                is_dampen, ref_boost, N, total_len, device
            )
        else:
            if input_attn is not None and input_attn.dtype != q_input.dtype:
                input_attn = input_attn.to(dtype=q_input.dtype)
            out_input = F.scaled_dot_product_attention(
                q_input, k_all, v_all,
                attn_mask=input_attn,
                scale=self.scale
            )
        out_all[:, :, :N] = out_input

        return out_all

    def _input_queries_flex(self, q_input, k_all, v_all, input_mask, input_score_bias,
                             is_dampen, ref_boost, N, total_len, device):
        """
        flex_attention path for Phase 2 input queries.

        Uses the already-built dense `input_mask` (N, total_len) as ground truth
        for what positions are allowed. Reuses the identical masking logic as the
        SDPA path — the only difference is the attention kernel itself.

        Dampening and ref_boost are applied via score_mod (additive biases indexed
        into a precomputed (N, total_len) bias tensor).

        When no masking or biasing is needed, falls through to unmasked SDPA
        (flex_attention offers no advantage over Flash Attention in that case).
        """
        # No mask + no bias → unmasked SDPA is optimal (Flash Attention eligible)
        needs_bias = (input_score_bias is not None) or is_dampen or (ref_boost > 0)
        if input_mask is None and not needs_bias:
            return F.scaled_dot_product_attention(
                q_input, k_all, v_all, attn_mask=None, scale=self.scale
            )

        # Build BlockMask from dense mask if mask is present.
        block_mask = None
        if input_mask is not None:
            mask_dev = input_mask.to(device)  # (N, total_len) bool

            def mask_mod(b, h, q_idx, kv_idx):
                return mask_dev[q_idx, kv_idx]

            block_mask = create_block_mask(
                mask_mod,
                B=None, H=None,
                Q_LEN=N, KV_LEN=total_len,
                device=device,
                _compile=False,
            )

        # Build score_mod for additive biases (beta cross-view, dampening, ref_boost).
        # All biases are computed identically to the SDPA path to guarantee numerical match.
        score_mod = None
        if needs_bias:
            bias = torch.zeros(N, total_len, device=device, dtype=torch.float32)

            if input_score_bias is not None:
                bias += input_score_bias.to(device=device, dtype=torch.float32)

            if is_dampen and input_mask is not None:
                mask_dev_cpu = input_mask  # kept on CPU/device for counting
                n_in_per_q = input_mask[:, :N].sum(dim=1).float()
                n_ref_per_q = input_mask[:, N:].sum(dim=1).float()
                valid = (n_in_per_q > 0) & (n_ref_per_q > 0)
                dampen_vals = torch.zeros(N, device='cpu')
                dampen_vals[valid] = torch.log(n_ref_per_q[valid] / n_in_per_q[valid])
                bias[:, :N] += dampen_vals.to(device).unsqueeze(1)

            if ref_boost > 0:
                bias[:, N:] += ref_boost

            bias = bias.to(dtype=q_input.dtype)

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[q_idx, kv_idx]

        return _flex_attention_compiled(
            q_input, k_all, v_all,
            block_mask=block_mask,
            score_mod=score_mod,
            scale=self.scale,
        )

    def _build_input_query_mask(self, H, W, N, F_effs, K, B,
                                 total_len, beta, device, ref_to_bin=None,
                                 latent_bin_map=None, occ_masks=None,
                                 warp_maps=None, conf_mask=None,
                                 warp_patch_size=64, warp_forced=False,
                                 input_local_patch=0):
        """
        Build attention mask for input query rows only: (N, total_len).
        Used by split attention mode. Only builds the input view's query rows,
        skipping all ref→ref and ref→input mask computation.

        Returns:
          input_mask: (N, total_len) bool tensor, or None if no masking needed
          score_bias: (N, total_len) float tensor, or None if beta=0
        """
        # --- Score bias (input rows only) ---
        if beta != 0.0:
            # Input tokens (view 0) → view assignment for all tokens
            view_ids = torch.arange(total_len, device=device) // N
            is_cross = view_ids != 0  # True for all non-input tokens
            score_bias = torch.where(is_cross,
                torch.tensor(beta, device=device, dtype=torch.float32),
                torch.zeros(1, device=device, dtype=torch.float32))
            score_bias = score_bias.unsqueeze(0).expand(N, -1).clone()  # (N, total_len)
        else:
            score_bias = None

        # --- Determine what masking is needed ---
        need_epipolar = self.mask_mode not in ('all', 'all2', 'all3', 'none', 'warp')
        has_bins = ref_to_bin is not None and latent_bin_map is not None
        has_occ = occ_masks is not None
        has_warp = warp_maps is not None
        has_conf = conf_mask is not None
        has_input_local = input_local_patch > 0

        if self.mask_mode == 'none':
            return None, score_bias

        if not need_epipolar and not has_bins and not has_occ and not has_warp and not has_conf and not has_input_local:
            return None, score_bias

        input_mask = None

        # --- Epipolar masking (input→ref only) ---
        if need_epipolar:
            input_mask = torch.zeros(N, total_len, dtype=torch.bool, device='cpu')
            # Input self-attention: always allowed
            input_mask[:, :N] = True

            for r_idx in range(K):
                F_fwd = F_effs[r_idx].to(device)
                ref_start = (r_idx + 1) * N

                cols = torch.arange(W, device=device).float()
                rows = torch.arange(H, device=device).float()
                u_q = ((torch.arange(N, device=device) % W).float() + 0.5) / W
                v_q = ((torch.arange(N, device=device) // W).float() + 0.5) / H

                ref_cols = torch.arange(W, device=device).float()
                ref_rows = torch.arange(H, device=device).float()
                u_k = ((torch.arange(N, device=device) % W).float() + 0.5) / W
                v_k = ((torch.arange(N, device=device) // W).float() + 0.5) / H

                # Epipolar line in ref from each input query
                a_f = F_fwd[0, 0] * u_q.unsqueeze(1) + F_fwd[0, 1] * v_q.unsqueeze(1) + F_fwd[0, 2]
                b_f = F_fwd[1, 0] * u_q.unsqueeze(1) + F_fwd[1, 1] * v_q.unsqueeze(1) + F_fwd[1, 2]
                c_f = F_fwd[2, 0] * u_q.unsqueeze(1) + F_fwd[2, 1] * v_q.unsqueeze(1) + F_fwd[2, 2]
                denom_f = torch.sqrt(a_f * a_f + b_f * b_f + 1e-6)
                dist_fwd = torch.abs(a_f * u_k.unsqueeze(0) + b_f * v_k.unsqueeze(0) + c_f) / denom_f
                fwd_ok = dist_fwd < self.threshold  # (N, N)

                input_mask[:, ref_start:ref_start + N] = fwd_ok.cpu()

        # --- Bin routing (input rows only) ---
        if has_bins:
            H_lat, W_lat = latent_bin_map.shape
            if H_lat * W_lat == N:
                input_token_bins = latent_bin_map.flatten()
            else:
                import torch.nn.functional as Fnn
                _lbm = latent_bin_map.float().unsqueeze(0).unsqueeze(0)
                _lbm_resized = Fnn.interpolate(_lbm, size=(H, W), mode='nearest').long().squeeze()
                input_token_bins = _lbm_resized.flatten()

            view_bins = [None]
            for r_idx in range(K):
                view_bins.append(ref_to_bin[r_idx] if r_idx < len(ref_to_bin) else -1)

            if input_mask is None:
                input_mask = torch.ones(N, total_len, dtype=torch.bool, device='cpu')

            # Input→ref: block if input token's bin != ref's bin
            for vj in range(1, B):
                sj, ej = vj * N, (vj + 1) * N
                ref_bin = view_bins[vj]
                mismatch = (input_token_bins != ref_bin).cpu()
                input_mask[:, sj:ej][mismatch] = False

        # --- Occlusion masking (input→ref only) ---
        if has_occ:
            import torch.nn.functional as Fnn
            for r_idx in range(K):
                if r_idx >= len(occ_masks) or occ_masks[r_idx] is None:
                    continue
                om = occ_masks[r_idx]
                if om.shape[0] != H or om.shape[1] != W:
                    om_float = om.float().unsqueeze(0).unsqueeze(0)
                    om_resized = Fnn.interpolate(om_float, size=(H, W), mode='nearest').squeeze()
                    om_layer = om_resized > 0.5
                else:
                    om_layer = om
                occluded_tokens = (~om_layer).flatten().cpu()
                ref_start = (r_idx + 1) * N

                if input_mask is None:
                    input_mask = torch.ones(N, total_len, dtype=torch.bool, device='cpu')

                input_mask[:, ref_start:ref_start + N][:, occluded_tokens] = False

        # --- Warp masking (input→ref only) ---
        if has_warp:
            import torch.nn.functional as Fnn
            warp_radius_full = warp_patch_size // 2
            scale_factor = H / float(warp_maps[0].shape[0]) if warp_maps[0] is not None else 1.0
            warp_radius = max(1, int(warp_radius_full * scale_factor))

            if input_mask is None:
                input_mask = torch.ones(N, total_len, dtype=torch.bool, device='cpu')

            for r_idx in range(K):
                if r_idx >= len(warp_maps) or warp_maps[r_idx] is None:
                    continue
                wm = warp_maps[r_idx]
                if warp_forced:
                    cols_grid = torch.arange(W, device='cpu').float().unsqueeze(0).expand(H, W)
                    rows_grid = torch.arange(H, device='cpu').float().unsqueeze(1).expand(H, W)
                    wm_u_scaled = cols_grid
                    wm_v_scaled = rows_grid
                else:
                    wm_u = wm[..., 0].unsqueeze(0).unsqueeze(0)
                    wm_v = wm[..., 1].unsqueeze(0).unsqueeze(0)
                    wm_u_down = Fnn.interpolate(wm_u.float(), size=(H, W), mode='nearest').squeeze()
                    wm_v_down = Fnn.interpolate(wm_v.float(), size=(H, W), mode='nearest').squeeze()
                    wm_u_scaled = wm_u_down * (W / float(wm.shape[1]))
                    wm_v_scaled = wm_v_down * (H / float(wm.shape[0]))

                ref_start = (r_idx + 1) * N
                ref_rows = torch.arange(H, device='cpu').unsqueeze(1).expand(H, W).flatten().float()
                ref_cols = torch.arange(W, device='cpu').unsqueeze(0).expand(H, W).flatten().float()

                for inp_tok in range(N):
                    target_u = wm_u_scaled.flatten()[inp_tok].item()
                    target_v = wm_v_scaled.flatten()[inp_tok].item()
                    if not warp_forced and (target_u < 0 or target_v < 0):
                        continue
                    dist_u = (ref_cols - target_u).abs()
                    dist_v = (ref_rows - target_v).abs()
                    outside = (dist_u > warp_radius) | (dist_v > warp_radius)
                    input_mask[inp_tok, ref_start:ref_start + N][outside] = False

        # --- Confidence masking (input self-attention only) ---
        if has_conf:
            import torch.nn.functional as Fnn
            cm = conf_mask.float().unsqueeze(0).unsqueeze(0)
            cm_down = Fnn.interpolate(cm, size=(H, W), mode='nearest').squeeze()
            uncovered = (cm_down < 1).flatten().cpu()

            if input_mask is None:
                input_mask = torch.ones(N, total_len, dtype=torch.bool, device='cpu')

            input_mask[:, :N][:, uncovered] = False

        # --- Input local patch ---
        if has_input_local:
            if input_mask is None:
                input_mask = torch.ones(N, total_len, dtype=torch.bool, device='cpu')

            ilp_radius = input_local_patch // 2
            scale_factor = H / float(warp_maps[0].shape[0]) if (warp_maps and warp_maps[0] is not None) else 1.0
            ilp_radius_scaled = max(1, int(ilp_radius * scale_factor))

            inp_rows = torch.arange(H, device='cpu').unsqueeze(1).expand(H, W).flatten().float()
            inp_cols = torch.arange(W, device='cpu').unsqueeze(0).expand(H, W).flatten().float()

            for tok in range(N):
                tok_r = inp_rows[tok]
                tok_c = inp_cols[tok]
                dist_r = (inp_rows - tok_r).abs()
                dist_c = (inp_cols - tok_c).abs()
                outside = (dist_r > ilp_radius_scaled) | (dist_c > ilp_radius_scaled)
                input_mask[tok, :N][outside] = False

        return input_mask, score_bias

    def _build_symmetric_attn_mask(self, H, W, N_per_view, F_effs, K, B,
                                      total_len, beta, device, ref_to_bin=None,
                                      latent_bin_map=None, occ_masks=None,
                                      warp_maps=None, conf_mask=None,
                                      warp_patch_size=64, warp_forced=False,
                                      input_local_patch=0, block_ref_to_input=False):
        """
        Build explicit attn_mask tensor and optional score_bias for SDPA.
        KV layout: [view0: N | view1: N | ... | viewK: N].

        When ref_to_bin and latent_bin_map are provided (bins mode):
          - Input token in bin m can only attend to refs with ref_to_bin == m
          - Ref-to-ref: allowed only within same bin
          - Input self-attention: fully connected (no bin constraint)

        Returns:
          attn_mask: (total_len, total_len) bool tensor, or None if mask_mode='all' and no bins
          score_bias: (total_len, total_len) float tensor, or None if beta=0
        """
        N = N_per_view

        # --- Score bias ---
        if beta != 0.0:
            # Cross-view tokens get additive bias
            view_ids = torch.arange(total_len, device=device) // N  # which view each token belongs to
            is_cross = view_ids.unsqueeze(0) != view_ids.unsqueeze(1)  # (total, total)
            score_bias = torch.where(is_cross, torch.tensor(beta, device=device, dtype=torch.float32),
                                     torch.zeros(1, device=device, dtype=torch.float32))
        else:
            score_bias = None

        # --- Attention mask ---
        # Determine what masking is needed
        need_epipolar = self.mask_mode not in ('all', 'all2', 'all3', 'none', 'warp')
        has_bins = ref_to_bin is not None and latent_bin_map is not None
        has_occ = occ_masks is not None
        has_warp = warp_maps is not None
        has_conf = conf_mask is not None
        has_input_local = input_local_patch > 0

        # 'none' = explicitly no mask (for A/B comparison)
        if self.mask_mode == 'none':
            return None, score_bias

        # If nothing to mask, return early
        if not need_epipolar and not has_bins and not has_occ and not has_warp and not has_conf and not has_input_local:
            return None, score_bias

        # Build base mask: start with None (bin routing / occ will create if needed)
        attn_mask = None

        if need_epipolar:
            # Epipolar masking — build mask from scratch
            # Token indices → (view, local_row, local_col) → UV coords
            tok_ids = torch.arange(total_len, device=device)
            views = tok_ids // N
            local_ids = tok_ids % N
            cols = local_ids % W
            rows = local_ids // W
            u = (cols.float() + 0.5) / W
            v = (rows.float() + 0.5) / H

            # Start with same-view = True (on CPU to save GPU memory for epipolar computation)
            attn_mask = torch.zeros(total_len, total_len, dtype=torch.bool, device='cpu')
            for vi in range(B):
                s, e = vi * N, (vi + 1) * N
                attn_mask[s:e, s:e] = True

            # ref↔ref: allow all (block-wise to avoid OOM)
            for vi in range(1, B):
                for vj in range(1, B):
                    if vi != vj:
                        attn_mask[vi * N:(vi + 1) * N, vj * N:(vj + 1) * N] = True

        # input(0) ↔ ref(r): epipolar mask (only when need_epipolar)
        if need_epipolar:
            # Precompute epipolar lines for each ref
            for r_idx in range(K):
                F_fwd = F_effs[r_idx].to(device)  # input→ref
                F_bwd = F_fwd.T                    # ref→input
                ref_view = r_idx + 1

                # --- input(0) queries → ref(r) keys ---
                # For each input token, compute epipolar line in ref space
                input_mask = (views == 0)
                ref_mask = (views == ref_view)

                input_ids = torch.where(input_mask)[0]
                ref_ids = torch.where(ref_mask)[0]

                if input_ids.numel() > 0 and ref_ids.numel() > 0:
                    # Input token UV: (N_in, 1)
                    u_q = u[input_ids].unsqueeze(1)
                    v_q = v[input_ids].unsqueeze(1)
                    # Ref token UV: (1, N_ref)
                    u_k = u[ref_ids].unsqueeze(0)
                    v_k = v[ref_ids].unsqueeze(0)

                    # Epipolar line in ref from input query: l = F @ [u_q, v_q, 1]
                    a_f = F_fwd[0, 0] * u_q + F_fwd[0, 1] * v_q + F_fwd[0, 2]
                    b_f = F_fwd[1, 0] * u_q + F_fwd[1, 1] * v_q + F_fwd[1, 2]
                    c_f = F_fwd[2, 0] * u_q + F_fwd[2, 1] * v_q + F_fwd[2, 2]
                    denom_f = torch.sqrt(a_f * a_f + b_f * b_f + 1e-6)
                    dist_fwd = torch.abs(a_f * u_k + b_f * v_k + c_f) / denom_f  # (N_in, N_ref)
                    fwd_ok = dist_fwd < self.threshold

                    # Write into attn_mask (CPU): input_ids (rows) × ref_ids (cols)
                    attn_mask[input_ids.cpu().unsqueeze(1), ref_ids.cpu().unsqueeze(0)] = fwd_ok.cpu()

                    # --- ref(r) queries → input(0) keys (symmetric) ---
                    # Epipolar line in input from ref query: l = F.T @ [u_q_ref, v_q_ref, 1]
                    u_qr = u[ref_ids].unsqueeze(1)
                    v_qr = v[ref_ids].unsqueeze(1)
                    u_ki = u[input_ids].unsqueeze(0)
                    v_ki = v[input_ids].unsqueeze(0)

                    a_b = F_bwd[0, 0] * u_qr + F_bwd[0, 1] * v_qr + F_bwd[0, 2]
                    b_b = F_bwd[1, 0] * u_qr + F_bwd[1, 1] * v_qr + F_bwd[1, 2]
                    c_b = F_bwd[2, 0] * u_qr + F_bwd[2, 1] * v_qr + F_bwd[2, 2]
                    denom_b = torch.sqrt(a_b * a_b + b_b * b_b + 1e-6)
                    dist_bwd = torch.abs(a_b * u_ki + b_b * v_ki + c_b) / denom_b  # (N_ref, N_in)
                    bwd_ok = dist_bwd < self.threshold

                    attn_mask[ref_ids.cpu().unsqueeze(1), input_ids.cpu().unsqueeze(0)] = bwd_ok.cpu()

        # --- Bin routing constraint (block-wise to avoid OOM) ---
        if ref_to_bin is not None and latent_bin_map is not None:
            # Build per-view bin assignment
            # View 0 (input): per-token bins from latent_bin_map
            H_lat, W_lat = latent_bin_map.shape
            if H_lat * W_lat == N:
                input_token_bins = latent_bin_map.flatten()  # (N,)
            else:
                import torch.nn.functional as Fnn
                _lbm = latent_bin_map.float().unsqueeze(0).unsqueeze(0)
                _lbm_resized = Fnn.interpolate(_lbm, size=(H, W), mode='nearest').long().squeeze()
                input_token_bins = _lbm_resized.flatten()  # (N,)

            # Per-ref view bin (scalar per view)
            view_bins = [None]  # view 0 = input, per-token bins
            for r_idx in range(K):
                view_bins.append(ref_to_bin[r_idx] if r_idx < len(ref_to_bin) else -1)

            if attn_mask is None:
                attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device='cpu')

            # Block-wise: for each pair of views, check bin compatibility
            for vi in range(B):
                for vj in range(B):
                    if vi == vj:
                        continue  # same-view: always allowed
                    if vi == 0 and vj == 0:
                        continue  # input↔input: no constraint

                    si, ei = vi * N, (vi + 1) * N
                    sj, ej = vj * N, (vj + 1) * N

                    if vi == 0:
                        ref_bin = view_bins[vj]
                        mismatch = (input_token_bins != ref_bin).cpu()
                        attn_mask[si:ei, sj:ej][mismatch] = False
                    elif vj == 0:
                        ref_bin = view_bins[vi]
                        mismatch = (input_token_bins != ref_bin).cpu()
                        attn_mask[si:ei, sj:ej][:, mismatch] = False
                    else:
                        # ref↔ref: block if different bins
                        if view_bins[vi] != view_bins[vj]:
                            attn_mask[si:ei, sj:ej] = False

        # --- Occlusion masking: block input↔occluded ref tokens ---
        if occ_masks is not None:
            import torch.nn.functional as Fnn
            for r_idx in range(K):
                if r_idx >= len(occ_masks) or occ_masks[r_idx] is None:
                    continue
                om = occ_masks[r_idx]  # (H_model, W_model) bool, True=valid
                # Resize to current layer resolution
                if om.shape[0] != H or om.shape[1] != W:
                    om_float = om.float().unsqueeze(0).unsqueeze(0)
                    om_resized = Fnn.interpolate(om_float, size=(H, W), mode='nearest').squeeze()
                    om_layer = om_resized > 0.5
                else:
                    om_layer = om
                # Flatten to per-token mask
                occluded_tokens = (~om_layer).flatten().cpu()  # (N,) True=occluded
                ref_start = (r_idx + 1) * N
                ref_end = (r_idx + 2) * N

                if attn_mask is None:
                    attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device='cpu')

                # Block input→occluded ref tokens (columns)
                attn_mask[:N, ref_start:ref_end][:, occluded_tokens] = False
                # Block occluded ref tokens→input (rows)
                attn_mask[ref_start:ref_end, :N][occluded_tokens, :] = False

        # --- Warp masking: restrict input→ref attention to spatial neighborhood of warp target ---
        if has_warp:
            import torch.nn.functional as Fnn
            warp_radius_full = warp_patch_size // 2
            scale_factor = H / float(warp_maps[0].shape[0]) if warp_maps[0] is not None else 1.0
            warp_radius = max(1, int(warp_radius_full * scale_factor))

            if attn_mask is None:
                attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device='cpu')

            for r_idx in range(K):
                if r_idx >= len(warp_maps) or warp_maps[r_idx] is None:
                    continue
                wm = warp_maps[r_idx]  # (H_in, W_in, 2) — coords in model space

                if warp_forced:
                    # Teacher-forced: identity mapping (GT refs are pixel-aligned with input)
                    # Each input pixel (r, c) looks at (c, r) in model space
                    cols_grid = torch.arange(W, device='cpu').float().unsqueeze(0).expand(H, W)
                    rows_grid = torch.arange(H, device='cpu').float().unsqueeze(1).expand(H, W)
                    wm_u_scaled = cols_grid  # target u = col
                    wm_v_scaled = rows_grid  # target v = row
                else:
                    # Downsample warp map to current layer resolution
                    wm_u = wm[..., 0].unsqueeze(0).unsqueeze(0)
                    wm_v = wm[..., 1].unsqueeze(0).unsqueeze(0)
                    wm_u_down = Fnn.interpolate(wm_u.float(), size=(H, W), mode='nearest').squeeze()
                    wm_v_down = Fnn.interpolate(wm_v.float(), size=(H, W), mode='nearest').squeeze()
                    wm_u_scaled = wm_u_down * (W / float(wm.shape[1]))
                    wm_v_scaled = wm_v_down * (H / float(wm.shape[0]))

                ref_start = (r_idx + 1) * N
                ref_end = (r_idx + 2) * N

                ref_rows = torch.arange(H, device='cpu').unsqueeze(1).expand(H, W).flatten().float()
                ref_cols = torch.arange(W, device='cpu').unsqueeze(0).expand(H, W).flatten().float()

                for inp_tok in range(N):
                    target_u = wm_u_scaled.flatten()[inp_tok].item()
                    target_v = wm_v_scaled.flatten()[inp_tok].item()
                    if not warp_forced and (target_u < 0 or target_v < 0):
                        continue
                    dist_u = (ref_cols - target_u).abs()
                    dist_v = (ref_rows - target_v).abs()
                    outside = (dist_u > warp_radius) | (dist_v > warp_radius)
                    attn_mask[inp_tok, ref_start:ref_end][outside] = False
                    attn_mask[ref_start:ref_end, inp_tok][outside] = False

        # --- Confidence masking: block uncovered input tokens as keys ---
        if has_conf:
            import torch.nn.functional as Fnn
            # Resize confidence mask to current layer resolution
            cm = conf_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H_in,W_in)
            cm_down = Fnn.interpolate(cm, size=(H, W), mode='nearest').squeeze()  # (H, W)
            uncovered = (cm_down < 1).flatten().cpu()  # (N,) True = uncovered (confidence 0)

            if attn_mask is None:
                attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device='cpu')

            # Block uncovered input tokens as keys for ALL input queries
            # (no input token should attend to an uncovered input token)
            attn_mask[:N, :N][:, uncovered] = False

        # --- Input local patch: restrict input self-attention to spatial neighborhood ---
        if has_input_local:
            if attn_mask is None:
                attn_mask = torch.ones(total_len, total_len, dtype=torch.bool, device='cpu')

            ilp_radius = input_local_patch // 2
            scale_factor = H / float(warp_maps[0].shape[0]) if (warp_maps and warp_maps[0] is not None) else 1.0
            ilp_radius_scaled = max(1, int(ilp_radius * scale_factor))

            inp_rows = torch.arange(H, device='cpu').unsqueeze(1).expand(H, W).flatten().float()
            inp_cols = torch.arange(W, device='cpu').unsqueeze(0).expand(H, W).flatten().float()

            for tok in range(N):
                tok_r = inp_rows[tok]
                tok_c = inp_cols[tok]
                dist_r = (inp_rows - tok_r).abs()
                dist_c = (inp_cols - tok_c).abs()
                outside = (dist_r > ilp_radius_scaled) | (dist_c > ilp_radius_scaled)
                attn_mask[tok, :N][outside] = False

        return attn_mask, score_bias

    def _save_dashboard(self, query, key, h_in, w_in, N_ref,
                        F_effs, epipolar_caches, N_in, beta, filename,
                        ref_to_bin=None, latent_bin_map=None, K_orig=None,
                        warp_maps=None, warp_patch_size=64, warp_forced=False,
                        occ_masks=None):
        """
        Attention dashboard. When bins active: two M×(K_orig+1) grids (heatmap + epipolar).
        When no bins: original 2-row flat layout.
        """
        K = len(epipolar_caches)
        L_in = query.shape[2]
        total_kv = key.shape[2]
        ref_H, ref_W = h_in, w_in
        has_bins = ref_to_bin is not None and latent_bin_map is not None and K_orig is not None
        mode_str = self.cross_attn_mode if isinstance(self.cross_attn_mode, str) else f"{self.cross_attn_mode:.2f}"

        # -- custom points override (env ATTN_VIZ_POINTS_JSON) -----------------
        # If a JSON is provided and contains an entry for this frame, use those
        # (u, v) coords as the query points instead of the auto-sampled grid.
        custom_uvs = None  # list of (u, v) in [0, 1], or None
        frame_stem = None
        _pts = _load_viz_points()
        if _pts is not None:
            # Recover the frame stem from the filename we were passed, which is
            # 'debug_<frame_tag>_L<H>.jpg' or 'debug_<frame_tag>_L<H>_heatmap.jpg'.
            fn_base = os.path.basename(filename)
            for _stem, _entry in _pts.items():
                if _stem in fn_base:
                    custom_uvs = [(float(p["u"]), float(p["v"])) for p in _entry.get("points", [])]
                    frame_stem = _stem
                    print(f"[ECA VIZ]   using {len(custom_uvs)} custom points for {_stem}")
                    break
        # ---------------------------------------------------------------------
        W_panel = 320
        H_panel = int(W_panel * (h_in / w_in))
        cache0 = epipolar_caches[0]
        img_input_display = getattr(cache0, 'img_model_input_pil', cache0.img_query_pil)
        qcolors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,255,0),(255,0,255),(255,128,0),(0,255,128),(128,0,255)]

        if has_bins:
            # ----- Custom-points override (bins path) ------------------------
            # We deliberately bypass the existing per-bin query sampling loop.
            # Instead, we draw two parallel figures:
            #   <frame>_macro_unmasked.jpg  — softmax over the full KV, no bin
            #                                 mask, no epipolar/occ/warp mask.
            #                                 Shows where each query would
            #                                 naturally attend.
            #   <frame>_macro_masked.jpg    — softmax after applying the same
            #                                 masks the forward pass uses
            #                                 (bin routing + epipolar/occ/warp
            #                                 as applicable).
            # Each figure is [input_with_stars | ref_crop_0 | ref_crop_1 | ... | ref_crop_{K-1}]
            if custom_uvs is not None and len(custom_uvs) > 0:
                self._save_dashboard_custom(
                    query, key, h_in, w_in, N_ref,
                    F_effs, epipolar_caches, beta, custom_uvs, frame_stem,
                    filename, is_macro=True, K=K, N_in=N_in,
                    ref_to_bin=ref_to_bin, latent_bin_map=latent_bin_map,
                    warp_maps=warp_maps, warp_patch_size=warp_patch_size,
                    warp_forced=warp_forced, occ_masks=occ_masks,
                    qcolors=qcolors, W_panel=W_panel, H_panel=H_panel,
                    img_input_display=img_input_display,
                )
                return
            # ----- End custom-points override --------------------------------
            M = max(ref_to_bin) + 1 if ref_to_bin else 1
            # bin→batch ref indices
            bin_refs = [[] for _ in range(M)]
            slot_count = [0] * M
            for r_idx, b in enumerate(ref_to_bin):
                if 0 <= b < M:
                    bin_refs[b].append((r_idx, slot_count[b]))
                    slot_count[b] += 1
            # input token bins
            H_lat, W_lat = latent_bin_map.shape
            if H_lat * W_lat == N_in:
                input_bins = latent_bin_map.flatten()
            else:
                import torch.nn.functional as Fnn
                _lbm = latent_bin_map.float().unsqueeze(0).unsqueeze(0)
                input_bins = Fnn.interpolate(_lbm, size=(h_in, w_in), mode='nearest').long().squeeze().flatten()

            for viz_type in ['heatmap']:  # epipolar viz disabled — noisy and not needed for the paper
                grid_rows = []
                for m in range(M):
                    row_panels = []
                    bin_toks = torch.where(input_bins == m)[0]
                    if len(bin_toks) > 5:
                        q_ids = bin_toks[::len(bin_toks)//5][:5].tolist()
                    elif len(bin_toks) > 0:
                        q_ids = bin_toks.tolist()
                    else:
                        q_ids = []
                    # Compute scores + softmax for these query points
                    probs = None
                    if q_ids:
                        Q_sub = query[0, 0, q_ids, :]
                        K_h0 = key[0, 0, :, :]
                        sc = torch.matmul(Q_sub, K_h0.T) * self.scale
                        sc_cpu = sc.cpu()
                        for qi in range(len(q_ids)):
                            qt = q_ids[qi]
                            u_q = (float(qt % w_in) + 0.5) / w_in
                            v_q = (float(qt // w_in) + 0.5) / h_in
                            for r in range(K):
                                rs = (r + 1) * N_ref
                                re = (r + 2) * N_ref
                                if ref_to_bin[r] != m:
                                    sc_cpu[qi, rs:re] = float('-inf')
                                elif self.mask_mode == 'epipolar':
                                    Fe = F_effs[r].to(query.device)
                                    pq = torch.tensor([u_q, v_q, 1.0], device=Fe.device, dtype=Fe.dtype)
                                    ln = Fe @ pq
                                    a, b, c = ln[0], ln[1], ln[2]
                                    yr = torch.arange(ref_H, device=Fe.device, dtype=Fe.dtype)
                                    xr = torch.arange(ref_W, device=Fe.device, dtype=Fe.dtype)
                                    gy, gx = torch.meshgrid(yr, xr, indexing='ij')
                                    uk = (gx + 0.5) / ref_W
                                    vk = (gy + 0.5) / ref_H
                                    d = torch.abs(a*uk + b*vk + c) / torch.sqrt(a*a + b*b + 1e-6)
                                    em = (d < self.threshold).flatten().cpu()
                                    sc_cpu[qi, rs:re][~em] = float('-inf')
                                elif self.mask_mode == 'warp' and warp_maps is not None and r < len(warp_maps) and warp_maps[r] is not None:
                                    wm = warp_maps[r]  # (H_in, W_in, 2)
                                    inp_row = qt // w_in
                                    inp_col = qt % w_in
                                    if warp_forced:
                                        # Teacher-forced: identity mapping
                                        tu = float(inp_col)
                                        tv = float(inp_row)
                                    else:
                                        wm_row = int(inp_row * wm.shape[0] / h_in)
                                        wm_col = int(inp_col * wm.shape[1] / w_in)
                                        wm_row = min(wm_row, wm.shape[0] - 1)
                                        wm_col = min(wm_col, wm.shape[1] - 1)
                                        target_u = wm[wm_row, wm_col, 0].item()
                                        target_v = wm[wm_row, wm_col, 1].item()
                                        if target_u < 0 or target_v < 0:
                                            tu, tv = -1, -1
                                        else:
                                            tu = target_u * ref_W / wm.shape[1]
                                            tv = target_v * ref_H / wm.shape[0]
                                    if tu >= 0 and tv >= 0:
                                        warp_radius = max(1, int((warp_patch_size // 2) * ref_H / (wm.shape[0] if not warp_forced else h_in)))
                                        ref_r = torch.arange(ref_H).unsqueeze(1).expand(ref_H, ref_W).flatten().float()
                                        ref_c = torch.arange(ref_W).unsqueeze(0).expand(ref_H, ref_W).flatten().float()
                                        outside = (ref_c - tu).abs() > warp_radius
                                        outside = outside | ((ref_r - tv).abs() > warp_radius)
                                        sc_cpu[qi, rs:re][outside] = float('-inf')
                                if beta != 0.0 and ref_to_bin[r] == m:
                                    vld = sc_cpu[qi, rs:re] != float('-inf')
                                    sc_cpu[qi, rs:re][vld] += beta
                                # Padding-aware mask: zero out ref tokens outside
                                # the crop's valid region so the dashboard reflects
                                # what the attention path actually sees.
                                if occ_masks is not None and r < len(occ_masks) and occ_masks[r] is not None:
                                    om = occ_masks[r]
                                    # Resize to current layer resolution (H=ref_H, W=ref_W)
                                    if om.shape[0] != ref_H or om.shape[1] != ref_W:
                                        import torch.nn.functional as Fnn
                                        om_f = om.float().unsqueeze(0).unsqueeze(0)
                                        om_r = Fnn.interpolate(om_f, size=(ref_H, ref_W), mode='nearest')
                                        om_flat = (om_r.squeeze() > 0.5).flatten().cpu()
                                    else:
                                        om_flat = om.flatten().cpu()
                                    sc_cpu[qi, rs:re][~om_flat] = float('-inf')
                        # Apply dampening bias before softmax
                        if self.cross_attn_mode == 'dampen':
                            for qi in range(len(q_ids)):
                                n_in = (sc_cpu[qi, :N_ref] != float('-inf')).sum().item()
                                n_ref = (sc_cpu[qi, N_ref:] != float('-inf')).sum().item()
                                if n_in > 0 and n_ref > 0:
                                    sc_cpu[qi, :N_ref][sc_cpu[qi, :N_ref] != float('-inf')] += math.log(n_ref / n_in)
                        probs = torch.softmax(sc_cpu, dim=1)

                    for r_slot in range(K_orig):
                        batch_idx = None
                        for bi, sl in bin_refs[m]:
                            if sl == r_slot:
                                batch_idx = bi
                                break
                        if batch_idx is None:
                            p = np.zeros((H_panel, W_panel, 3), dtype=np.uint8)
                            cv2.line(p, (0,0), (W_panel,H_panel), (0,0,255), 2)
                            cv2.line(p, (W_panel,0), (0,H_panel), (0,0,255), 2)
                        else:
                            cr = epipolar_caches[batch_idx]
                            p = cv2.cvtColor(np.array(cr.img_target_pil.resize((W_panel, H_panel))), cv2.COLOR_RGB2BGR)
                            # Grayscale background so colorful overlays pop
                            p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
                            p = cv2.cvtColor(p_gray, cv2.COLOR_GRAY2BGR)
                            rs = (batch_idx + 1) * N_ref
                            re = (batch_idx + 2) * N_ref
                            if probs is not None and q_ids:
                                for qi, qt in enumerate(q_ids):
                                    col = qcolors[qi % len(qcolors)]
                                    if viz_type == 'heatmap':
                                        rp = probs[qi, rs:re].numpy().reshape(ref_H, ref_W)
                                        rp_up = cv2.resize(rp, (W_panel, H_panel), interpolation=cv2.INTER_NEAREST)
                                        if rp_up.max() > 0: rp_up /= rp_up.max()
                                        ys, xs = np.where(rp_up > 0.1)
                                        for j in range(len(xs)):
                                            it = rp_up[ys[j], xs[j]]
                                            cur = p[ys[j], xs[j]].astype(float)
                                            tgt = np.array(col, dtype=float)
                                            p[ys[j], xs[j]] = (cur*(1-it) + tgt*it).astype(np.uint8)
                                        # Draw warp patch rectangle if warp mode
                                        if self.mask_mode == 'warp' and warp_maps is not None and batch_idx < len(warp_maps) and warp_maps[batch_idx] is not None:
                                            wm = warp_maps[batch_idx]
                                            inp_row = qt // w_in
                                            inp_col = qt % w_in
                                            if warp_forced:
                                                tu = float(inp_col)
                                                tv = float(inp_row)
                                                wr = max(1, int((warp_patch_size // 2) * ref_H / h_in))
                                            else:
                                                wm_row = min(int(inp_row * wm.shape[0] / h_in), wm.shape[0] - 1)
                                                wm_col = min(int(inp_col * wm.shape[1] / w_in), wm.shape[1] - 1)
                                                tu = wm[wm_row, wm_col, 0].item()
                                                tv = wm[wm_row, wm_col, 1].item()
                                                wr = max(1, int((warp_patch_size // 2) * ref_H / wm.shape[0]))
                                            if tu >= 0 and tv >= 0:
                                                px0 = int((tu / ref_W - wr / ref_W) * W_panel)
                                                py0 = int((tv / ref_H - wr / ref_H) * H_panel)
                                                px1 = int((tu / ref_W + wr / ref_W) * W_panel)
                                                py1 = int((tv / ref_H + wr / ref_H) * H_panel)
                                                cv2.rectangle(p, (px0, py0), (px1, py1), col, 1)
                                    else:
                                        Fe = F_effs[batch_idx].to(query.device)
                                        uq = (float(qt % w_in) + 0.5) / w_in
                                        vq = (float(qt // w_in) + 0.5) / h_in
                                        pq = torch.tensor([uq, vq, 1.0], device=Fe.device, dtype=Fe.dtype)
                                        ln = Fe @ pq
                                        a, b, c = ln[0], ln[1], ln[2]
                                        yr = torch.arange(ref_H, device=Fe.device, dtype=Fe.dtype)
                                        xr = torch.arange(ref_W, device=Fe.device, dtype=Fe.dtype)
                                        gy, gx = torch.meshgrid(yr, xr, indexing='ij')
                                        uk = (gx+0.5)/ref_W; vk = (gy+0.5)/ref_H
                                        d = torch.abs(a*uk+b*vk+c)/torch.sqrt(a*a+b*b+1e-6)
                                        em = (d < self.threshold).cpu().numpy().astype(np.uint8)
                                        mv = cv2.resize(em, (W_panel, H_panel), interpolation=cv2.INTER_NEAREST)
                                        ct, _ = cv2.findContours(mv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        cv2.drawContours(p, ct, -1, col, 1)
                            if probs is not None:
                                ms = probs[:, rs:re].sum(dim=1).mean().item()
                                # Caption omitted for clean paper-ready images.
                                _ = ms
                        row_panels.append(p)
                    # Input column with query stars
                    inp = cv2.cvtColor(np.array(img_input_display.resize((W_panel, H_panel))), cv2.COLOR_RGB2BGR)
                    for qi, qt in enumerate(q_ids):
                        col = qcolors[qi % len(qcolors)]
                        cx = int((qt % w_in + 0.5) / w_in * W_panel)
                        cy = int((qt // w_in + 0.5) / h_in * H_panel)
                        cv2.drawMarker(inp, (cx, cy), col, markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)
                    # Caption omitted for clean paper-ready images.
                    row_panels.append(inp)
                    grid_rows.append(np.hstack(row_panels))
                grid = np.vstack(grid_rows)
                out_name = filename.replace(".jpg", f"_{viz_type}.jpg")
                cv2.imwrite(os.path.join(self.viz_folder, out_name), grid)
        else:
            # --- ORIGINAL FLAT DASHBOARD (no bins) ---
            # Custom-points override for DiFix: single figure, no mask logic
            # (DiFix uses full refs without bins, so "masked" and "unmasked"
            #  collapse to the same thing).
            if custom_uvs is not None and len(custom_uvs) > 0:
                self._save_dashboard_custom(
                    query, key, h_in, w_in, N_ref,
                    F_effs, epipolar_caches, beta, custom_uvs, frame_stem,
                    filename, is_macro=False, K=K, N_in=N_in,
                    ref_to_bin=ref_to_bin, latent_bin_map=latent_bin_map,
                    warp_maps=warp_maps, warp_patch_size=warp_patch_size,
                    warp_forced=warp_forced, occ_masks=occ_masks,
                    qcolors=qcolors, W_panel=W_panel, H_panel=H_panel,
                    img_input_display=img_input_display,
                )
                return
            qy = np.linspace(h_in // 4, h_in * 3 // 4, 3).astype(int)
            qx = np.linspace(w_in // 4, w_in * 3 // 4, 3).astype(int)
            query_indices = [y * w_in + x for y in qy for x in qx]
            Q_sub = query[0, 0, query_indices, :]
            K_h0 = key[0, 0, :, :]
            sc = torch.matmul(Q_sub, K_h0.T) * self.scale
            sc_cpu = sc.cpu()
            for i, qi in enumerate(query_indices):
                u_q = (float(qi % w_in) + 0.5) / w_in
                v_q = (float(qi // w_in) + 0.5) / h_in
                for r in range(K):
                    rs, re = (r+1)*N_ref, (r+2)*N_ref
                    if self.mask_mode == 'epipolar':
                        Fe = F_effs[r].to(query.device)
                        pq = torch.tensor([u_q, v_q, 1.0], device=Fe.device, dtype=Fe.dtype)
                        ln = Fe @ pq; a, b, c = ln[0], ln[1], ln[2]
                        yr = torch.arange(ref_H, device=Fe.device, dtype=Fe.dtype)
                        xr = torch.arange(ref_W, device=Fe.device, dtype=Fe.dtype)
                        gy, gx = torch.meshgrid(yr, xr, indexing='ij')
                        uk = (gx+0.5)/ref_W; vk = (gy+0.5)/ref_H
                        d = torch.abs(a*uk+b*vk+c)/torch.sqrt(a*a+b*b+1e-6)
                        em = (d < self.threshold).flatten().cpu()
                        sc_cpu[i, rs:re][~em] = float('-inf')
                    if beta != 0.0:
                        vld = sc_cpu[i, rs:re] != float('-inf')
                        sc_cpu[i, rs:re][vld] += beta
                    # Padding-aware mask (no-bins dashboard)
                    if occ_masks is not None and r < len(occ_masks) and occ_masks[r] is not None:
                        om = occ_masks[r]
                        if om.shape[0] != ref_H or om.shape[1] != ref_W:
                            import torch.nn.functional as Fnn
                            om_f = om.float().unsqueeze(0).unsqueeze(0)
                            om_r = Fnn.interpolate(om_f, size=(ref_H, ref_W), mode='nearest')
                            om_flat = (om_r.squeeze() > 0.5).flatten().cpu()
                        else:
                            om_flat = om.flatten().cpu()
                        sc_cpu[i, rs:re][~om_flat] = float('-inf')
            probs = torch.softmax(sc_cpu, dim=1)
            masses = [probs[:, v*N_ref:(v+1)*N_ref].sum(dim=1).mean().item() for v in range(K+1)]
            viz_in = cv2.cvtColor(np.array(img_input_display.resize((W_panel, H_panel))), cv2.COLOR_RGB2BGR)
            for i, idx in enumerate(query_indices):
                col = qcolors[i % len(qcolors)]
                cx = int((idx % w_in + 0.5) / w_in * W_panel)
                cy = int((idx // w_in + 0.5) / h_in * H_panel)
                cv2.drawMarker(viz_in, (cx, cy), col, markerType=cv2.MARKER_STAR, markerSize=15, thickness=3)
            # Caption omitted for clean paper-ready images.
            bands, heats = [], []
            for r in range(K):
                cr = epipolar_caches[r]
                Fe = F_effs[r]
                rb = cv2.cvtColor(np.array(cr.img_target_pil.resize((W_panel, H_panel))), cv2.COLOR_RGB2BGR)
                # Grayscale background so colorful overlays pop
                rb_gray = cv2.cvtColor(rb, cv2.COLOR_BGR2GRAY)
                rb = cv2.cvtColor(rb_gray, cv2.COLOR_GRAY2BGR)
                bi, hi = rb.copy(), rb.copy()
                rs, re = (r+1)*N_ref, (r+2)*N_ref
                for i, qi in enumerate(query_indices):
                    col = qcolors[i % len(qcolors)]
                    u_q = (float(qi % w_in)+0.5)/w_in; v_q = (float(qi // w_in)+0.5)/h_in
                    dev = Fe.device
                    pq = torch.tensor([u_q, v_q, 1.0], device=dev, dtype=Fe.dtype)
                    ln = Fe @ pq; a, b, c = ln[0], ln[1], ln[2]
                    yr = torch.arange(ref_H, device=dev, dtype=Fe.dtype)
                    xr = torch.arange(ref_W, device=dev, dtype=Fe.dtype)
                    gy, gx = torch.meshgrid(yr, xr, indexing='ij')
                    uk = (gx+0.5)/ref_W; vk = (gy+0.5)/ref_H
                    d = torch.abs(a*uk+b*vk+c)/torch.sqrt(a*a+b*b+1e-6)
                    em = (d < self.threshold).cpu().numpy().astype(np.uint8)
                    mv = cv2.resize(em, (W_panel, H_panel), interpolation=cv2.INTER_NEAREST)
                    ct, _ = cv2.findContours(mv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(bi, ct, -1, col, 1)
                    rp = probs[i, rs:re].numpy().reshape(ref_H, ref_W)
                    rp_up = cv2.resize(rp, (W_panel, H_panel), interpolation=cv2.INTER_NEAREST)
                    if rp_up.max() > 0: rp_up /= rp_up.max()
                    ys, xs = np.where(rp_up > 0.1)
                    for j in range(len(xs)):
                        it = rp_up[ys[j], xs[j]]
                        cur = hi[ys[j], xs[j]].astype(float)
                        tgt = np.array(col, dtype=float)
                        hi[ys[j], xs[j]] = (cur*(1-it)+tgt*it).astype(np.uint8)
                # Captions omitted for clean paper-ready images.
                bands.append(bi); heats.append(hi)
            def pad_h(img, th):
                return cv2.copyMakeBorder(img, 0, th-img.shape[0], 0, 0, cv2.BORDER_CONSTANT) if img.shape[0] < th else img
            rh = max(viz_in.shape[0], max(p.shape[0] for p in bands))
            r1 = np.hstack([pad_h(viz_in, rh)] + [pad_h(p, rh) for p in bands])
            bk = np.zeros_like(pad_h(viz_in, rh))
            r2 = np.hstack([bk] + [pad_h(p, rh) for p in heats])
            cv2.imwrite(os.path.join(self.viz_folder, filename), np.vstack([r1, r2]))

    # ------------------------------------------------------------------------
    # Custom-points dashboard helper (used when ATTN_VIZ_POINTS_JSON matches)
    # ------------------------------------------------------------------------
    # Produces 1 (DiFix) or 2 (Macro: unmasked + masked) horizontally-stacked
    # rows of [input_with_stars | ref_panel_0 | ref_panel_1 | ...]. The same
    # (u, v) query points are used across both variants. Colors follow qcolors
    # in order, so rows are visually comparable across figures.
    def _save_dashboard_custom(self, query, key, h_in, w_in, N_ref,
                                F_effs, epipolar_caches, beta, custom_uvs, frame_stem,
                                filename, is_macro, K, N_in,
                                ref_to_bin, latent_bin_map, warp_maps,
                                warp_patch_size, warp_forced, occ_masks,
                                qcolors, W_panel, H_panel, img_input_display):
        import os
        # Map (u, v) -> latent query token index q = round(v*h_in)*w_in + round(u*w_in)
        q_ids = []
        q_uvs = []
        for (u, v) in custom_uvs:
            cx = int(round(u * w_in - 0.5))
            cy = int(round(v * h_in - 0.5))
            cx = max(0, min(w_in - 1, cx))
            cy = max(0, min(h_in - 1, cy))
            q_ids.append(cy * w_in + cx)
            q_uvs.append((u, v))

        # Precompute scores for all q_ids vs full KV, across ALL heads.
        # query shape: (1, H_heads, seq, d_head); key: (1, H_heads, kv_seq, d_head)
        # Q_all: (H_heads, Q, d_head)   K_all: (H_heads, kv_seq, d_head)
        Q_all = query[0, :, q_ids, :]
        K_all = key[0, :, :, :]
        # sc per head: (H_heads, Q, (K+1)*N_ref)
        sc_all_heads = torch.matmul(Q_all, K_all.transpose(-1, -2)) * self.scale
        # Back-compat single-head view (head 0) for downstream non-cache code
        # that still expects (Q, (K+1)*N_ref).
        sc = sc_all_heads[0]
        # Keep ref columns only (strip the first N_ref input-self columns before
        # softmax, so heatmap probs concentrate on the refs, not on input self-
        # attention).
        sc_ref_only = sc[:, N_ref:].clone()  # shape (Q, K*N_ref)

        # ---- compute probs for the two variants --------------------------
        # Variant A: unmasked (raw softmax across K ref panels)
        probs_unmasked = torch.softmax(sc_ref_only.cpu(), dim=1)

        # Optional: cache inputs for offline re-rendering at multiple
        # thresholds. Set env ECA_VIZ_CACHE_PT=/path/to/cache.pt to enable.
        #   ECA_VIZ_CACHE_ALL_LAYERS=1  — save one file per layer into a dir
        #   ECA_VIZ_CACHE_PT=... (file) — save a single file (default)
        _cache_path = os.environ.get("ECA_VIZ_CACHE_PT")
        _cache_all = os.environ.get("ECA_VIZ_CACHE_ALL_LAYERS", "").strip() not in ("", "0", "false")
        if _cache_path:
            try:
                import pathlib as _pl
                # Save the full pre-softmax logits so analysis can split self vs ref.
                sc_full = sc.cpu().clone()                    # (Q, (K+1)*N_ref)
                sc_full_all_heads = sc_all_heads.cpu().clone()  # (H_heads, Q, (K+1)*N_ref)
                _payload = {
                    "sc_full": sc_full,                    # head-0 logits, for back-compat
                    "sc_full_all_heads": sc_full_all_heads,  # all heads, (H, Q, (K+1)*N_ref)
                    "probs": probs_unmasked,               # ref-only softmax (Q, K*N_ref), head 0
                    "h_in": h_in, "w_in": w_in, "N_ref": N_ref, "K": K,
                    "is_down": bool(getattr(self, "is_down", False)),
                    "custom_uvs": q_uvs,
                    "frame_stem": frame_stem,
                    "filename": filename,
                    "input_pil": img_input_display.copy(),
                    "ref_pils": [c.img_target_pil.copy() for c in epipolar_caches],
                    "is_macro": bool(is_macro),
                }
                if _cache_all:
                    out_dir = _pl.Path(_cache_path)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    _name = os.path.splitext(os.path.basename(filename))[0]
                    _out = out_dir / f"{_name}.pt"
                else:
                    _pl.Path(_cache_path).parent.mkdir(parents=True, exist_ok=True)
                    _out = _pl.Path(_cache_path)
                torch.save(_payload, _out)
                print(f"[ECA VIZ] cached probs+logits to {_out}")
            except Exception as _e:
                print(f"[ECA VIZ] cache failed: {_e}")

        # Variant B: masked — apply the same masks the forward pass uses.
        probs_masked = None
        sc_masked_cached = None  # for cache payload (macro only)
        if is_macro:
            sc_masked = sc_ref_only.cpu().clone()
            # Determine bin of each query point using latent_bin_map
            H_lat, W_lat = latent_bin_map.shape
            for qi in range(len(q_ids)):
                u, v = q_uvs[qi]
                lx = int(round(u * W_lat - 0.5))
                ly = int(round(v * H_lat - 0.5))
                lx = max(0, min(W_lat - 1, lx))
                ly = max(0, min(H_lat - 1, ly))
                q_bin = int(latent_bin_map[ly, lx].item())
                for r in range(K):
                    rs = r * N_ref
                    re = (r + 1) * N_ref
                    # Bin mask: zero out refs not in the query's bin
                    if ref_to_bin[r] != q_bin:
                        sc_masked[qi, rs:re] = float('-inf')
                        continue
                    # Epipolar mask (if enabled)
                    if self.mask_mode == 'epipolar':
                        Fe = F_effs[r].to(query.device)
                        pq = torch.tensor([u, v, 1.0], device=Fe.device, dtype=Fe.dtype)
                        ln = Fe @ pq
                        a, b, c = ln[0], ln[1], ln[2]
                        yr = torch.arange(h_in, device=Fe.device, dtype=Fe.dtype)
                        xr = torch.arange(w_in, device=Fe.device, dtype=Fe.dtype)
                        gy, gx = torch.meshgrid(yr, xr, indexing='ij')
                        uk = (gx + 0.5) / w_in; vk = (gy + 0.5) / h_in
                        d = torch.abs(a*uk + b*vk + c) / torch.sqrt(a*a + b*b + 1e-6)
                        em = (d < self.threshold).flatten().cpu()
                        sc_masked[qi, rs:re][~em] = float('-inf')
                    # beta prior
                    if beta != 0.0:
                        vld = sc_masked[qi, rs:re] != float('-inf')
                        sc_masked[qi, rs:re][vld] += beta
                    # occlusion (padding-aware) mask
                    if occ_masks is not None and r < len(occ_masks) and occ_masks[r] is not None:
                        om = occ_masks[r]
                        if om.shape[0] != h_in or om.shape[1] != w_in:
                            import torch.nn.functional as Fnn
                            om_f = om.float().unsqueeze(0).unsqueeze(0)
                            om_r = Fnn.interpolate(om_f, size=(h_in, w_in), mode='nearest')
                            om_flat = (om_r.squeeze() > 0.5).flatten().cpu()
                        else:
                            om_flat = om.flatten().cpu()
                        sc_masked[qi, rs:re][~om_flat] = float('-inf')
            probs_masked = torch.softmax(sc_masked, dim=1)
            sc_masked_cached = sc_masked.clone()

        # Re-cache including masked logits for macro (so we can render the
        # full-softmax masked variant offline without re-running the model).
        if _cache_path and is_macro and sc_masked_cached is not None:
            try:
                import pathlib as _pl
                # Rebuild full-length masked logits:
                # columns 0..N_ref are self (kept from sc), N_ref.. are sc_masked
                _full_masked = sc.cpu().clone()
                _full_masked[:, N_ref:] = sc_masked_cached

                # Build all-heads masked tensor by applying the same -inf mask
                # pattern (which is head-agnostic) to each head.
                sc_all_cpu = sc_all_heads.cpu()  # (H_heads, Q, (K+1)*N_ref)
                ref_mask_inf = torch.isinf(sc_masked_cached)  # (Q, K*N_ref)
                _full_masked_all = sc_all_cpu.clone()
                for _h in range(_full_masked_all.shape[0]):
                    _hm = _full_masked_all[_h]
                    _hm[:, N_ref:][ref_mask_inf] = float('-inf')
                    _full_masked_all[_h] = _hm

                _payload_m = {
                    "sc_full": sc.cpu(),
                    "sc_full_masked": _full_masked,
                    "sc_full_all_heads": sc_all_cpu,
                    "sc_full_masked_all_heads": _full_masked_all,
                    "probs": probs_unmasked,
                    "probs_masked": probs_masked,
                    "h_in": h_in, "w_in": w_in, "N_ref": N_ref, "K": K,
                    "is_down": bool(getattr(self, "is_down", False)),
                    "custom_uvs": q_uvs,
                    "frame_stem": frame_stem,
                    "filename": filename,
                    "input_pil": img_input_display.copy(),
                    "ref_pils": [c.img_target_pil.copy() for c in epipolar_caches],
                    "is_macro": True,
                }
                if _cache_all:
                    out_dir = _pl.Path(_cache_path)
                    _name = os.path.splitext(os.path.basename(filename))[0]
                    _out = out_dir / f"{_name}.pt"
                else:
                    _out = _pl.Path(_cache_path)
                torch.save(_payload_m, _out)
            except Exception as _e:
                print(f"[ECA VIZ] cache macro masked failed: {_e}")

        # ---- render helper ------------------------------------------------
        def _render(probs, variant_tag):
            # Input panel with stars
            inp = cv2.cvtColor(np.array(img_input_display.resize((W_panel, H_panel))), cv2.COLOR_RGB2BGR)
            for qi in range(len(q_ids)):
                u, v = q_uvs[qi]
                col = qcolors[qi % len(qcolors)]
                cx = int(u * W_panel)
                cy = int(v * H_panel)
                cv2.drawMarker(inp, (cx, cy), col, markerType=cv2.MARKER_STAR,
                               markerSize=14, thickness=2)
            # Ref panels
            ref_panels = []
            for r in range(K):
                cr = epipolar_caches[r]
                ref_pil = cr.img_target_pil.resize((W_panel, H_panel))
                panel = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
                # Grayscale background so colorful overlays pop
                pg = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
                panel = cv2.cvtColor(pg, cv2.COLOR_GRAY2BGR)
                rs = r * N_ref
                re = (r + 1) * N_ref
                for qi in range(len(q_ids)):
                    col = qcolors[qi % len(qcolors)]
                    rp = probs[qi, rs:re].numpy().reshape(h_in, w_in)
                    rp_up = cv2.resize(rp, (W_panel, H_panel), interpolation=cv2.INTER_NEAREST)
                    if rp_up.max() > 0:
                        rp_up /= rp_up.max()
                    ys, xs = np.where(rp_up > 0.1)
                    for j in range(len(xs)):
                        it = rp_up[ys[j], xs[j]]
                        cur = panel[ys[j], xs[j]].astype(float)
                        tgt = np.array(col, dtype=float)
                        panel[ys[j], xs[j]] = (cur * (1 - it) + tgt * it).astype(np.uint8)
                ref_panels.append(panel)
            row = np.hstack([inp] + ref_panels)
            out_name = filename.replace('.jpg', f'_custom_{variant_tag}.jpg')
            cv2.imwrite(os.path.join(self.viz_folder, out_name), row)
            print(f"[ECA VIZ]   wrote {out_name}")

        if is_macro:
            _render(probs_unmasked, 'macro_unmasked')
            _render(probs_masked,  'macro_masked')
        else:
            _render(probs_unmasked, 'difix')

