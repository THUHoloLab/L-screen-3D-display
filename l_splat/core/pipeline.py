"""
L-Splat Pipeline v3: RGB Image -> DAV3 Depth -> 3DGS -> Dual-Frustum L-Screen Rendering
=====================================================================================

     
  1. Depth Anything V3      (DINOv2-ViT-S, process_res=504 -> cv2 C++ JBF)
  2.        +      ->    3D    (45       X+Y=0, Z-up)
  3. Gaussian     (     KNN     Scale +        clamp)
  4a.         (5x5      ->        )
  4b.        (   3D      ->          )
  5.        (Candidate H view matrix +        , use_asymmetric=True)
  6. 2D Alpha      (       ramp, gamma-corrected)

   : Z-up    ,    =Z 
  Sweet Spot: (3/sqrt(2), 3/sqrt(2), 0.40) m in screen-centred coordinates
    : Y=0, X   [-1.28, 0] m,   : X=0, Y   [-1.28, 0] m

Usage:
    conda activate env_5080
    python l_splat/core/pipeline.py --image ~/Desktop/hypercar.jpg
    python l_splat/core/pipeline.py --video ~/Videos/test.mp4 --ema-alpha 0.4
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

#     Path setup                                                  
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GS_DIR = os.path.join(_PROJ_ROOT, "gaussian-splatting-main")
sys.path.insert(0, GS_DIR)
sys.path.insert(0, os.path.join(GS_DIR, "submodules", "diff-gaussian-rasterization"))
sys.path.insert(0, os.path.join(GS_DIR, "submodules", "simple-knn"))

# Mock simple_knn._C (needed by scene.gaussian_model import)
_simple_knn = type("simple_knn", (), {})()
_simple_knn._C = type("simple_knn._C", (), {})()
_simple_knn._C.distCUDA2 = lambda x: __import__("torch").zeros(0)
__import__("sys").modules["simple_knn"] = _simple_knn
__import__("sys").modules["simple_knn._C"] = _simple_knn._C

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_dual_frustum
from arguments import GroupParams
from utils.sh_utils import RGB2SH

from l_splat.core.dual_frustum import (
    DualFrustumCamera,
    print_frustum_diagnostics,
)
from l_splat.core.gpu_pointcloud import depth_to_unified_pointcloud_cuda
from l_splat.core.output_orientation import (
    PATENT_RAW_ROW0_EDGE,
    SCREEN_BOTTOM,
    SCREEN_TOP,
    adapt_panel_row0_edge,
)
from l_splat.geometry.frustum import build_l_screen_corners
from l_splat.core.depth_filter import cv2_jbf
from l_splat.depth.streamer import VideoDepthStreamer
from l_splat.depth.loader import load_dav3, predict_depth_fast, predict_depth_with_jbf
from l_splat.depth.anchor import DepthAnchorStream
from l_splat.depth.structure_refine import depth_debug_images, refine_depth_structure
from l_splat.geometry.virtual_box import map_background_to_panels


# ============================================================================
# 0   Configuration
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#    L-Screen      (Z-up)   
# Task07A accepted hardware profile: each physical panel renders 936x1176 px.
PANEL_PIXEL_W = 936
PANEL_PIXEL_H = 1176
PANEL_PHYS_W_M = 1.28                                    #        (m)
PANEL_PHYS_H_M = 1.60                                    #        (m)  160cm
#   : phys_h / phys_w = 1.60 / 1.28 = 1.25
DEFAULT_SWEET_SPOT_DISTANCE_CM = 300.0
DEFAULT_SWEET_SPOT_HEIGHT_CM = 170.0
DEFAULT_SWEET_SPOT_HEIGHT_MODE = "eye-from-floor"
DEFAULT_SCREEN_BOTTOM_HEIGHT_CM = 50.0


def resolve_sweet_spot(distance_cm=DEFAULT_SWEET_SPOT_DISTANCE_CM,
                       height_cm=DEFAULT_SWEET_SPOT_HEIGHT_CM,
                       panel_height_m=None,
                       height_mode=DEFAULT_SWEET_SPOT_HEIGHT_MODE,
                       screen_bottom_height_cm=DEFAULT_SCREEN_BOTTOM_HEIGHT_CM):
    """Resolve calibrated sweet-spot input into centered Z-up world coords."""
    mode = str(height_mode).strip().lower()
    if mode not in ("eye-from-floor", "world-z"):
        raise ValueError(
            "height_mode must be 'eye-from-floor' or 'world-z', "
            f"got {height_mode!r}")
    panel_h = float(PANEL_PHYS_H_M if panel_height_m is None else panel_height_m)
    eye_height_cm = float(height_cm)
    bottom_cm = float(screen_bottom_height_cm)
    center_cm = bottom_cm + panel_h * 100.0 * 0.5
    if mode == "eye-from-floor":
        resolved_z_m = (eye_height_cm - center_cm) / 100.0
        input_semantics = "eye height above floor"
    else:
        resolved_z_m = eye_height_cm / 100.0
        input_semantics = "world Z relative to screen center"

    xy_m = (float(distance_cm) / 100.0) / math.sqrt(2.0)
    sweet = np.array([xy_m, xy_m, resolved_z_m], dtype=np.float64)
    profile = {
        "mode": mode,
        "input_height_cm": eye_height_cm,
        "input_height_semantics": input_semantics,
        "screen_bottom_height_cm": bottom_cm,
        "screen_center_height_cm": center_cm,
        "panel_height_m": panel_h,
        "resolved_world_z_m": float(resolved_z_m),
        "resolved_sweet_spot_m": [float(v) for v in sweet],
        "distance_cm": float(distance_cm),
    }
    return sweet, profile


SWEET_SPOT, SWEET_SPOT_PROFILE = resolve_sweet_spot()

#         (X+Y=0, Z-up)   
#     XY      45   ,         (+X,+Y)
#             (-1,+1,0)/sqrt2,       Z   (0,0,1)
# ***   : content_w / content_h = W_img / H_img = 1440/1326 ***
#      : 1440/1326   1.08597
CONTENT_W_M = PANEL_PHYS_W_M   # = 1.28m,         ,         
#          (m)           : content_h = CONTENT_W_M * (H_img / W_img)

#             
DEPTH_FWD_M  = +0.15   # max forward displacement toward viewer (symmetric ±0.15m)
DEPTH_BACK_M = -0.15   # max backward displacement from panel (symmetric ±0.15m)

#            
SH_DEGREE = 0
GAUSSIAN_OVERLAP_FACTOR = 1.0  # was 2.0; reduced to narrow boundary transition zone from ~6px to ~2px, cutting TIR
GAUSSIAN_OPACITY_SIGMOID = 0.999  # target opacity after sigmoid activation (was 0.99; raised to compensate for smaller scales)

# Owner-preserving fold-edge support is a manual ablation only.
# Task29R rollback: the production path must not add synthetic seam/fold
# support by default, because filling the middle gap is not display-aware.
SEAM_BOUNDARY_SUPPORT_MODE = "off"  # "on" | "off"
SEAM_BOUNDARY_SUPPORT_ENABLED = False
SEAM_BOUNDARY_SUPPORT_BAND_PX = 24
SEAM_BOUNDARY_SUPPORT_SCALE_FLOOR_PANEL_PITCH = 1.25
SEAM_BOUNDARY_SUPPORT_SURFACE_EPS_M = 1e-7
SEAM_BOUNDARY_SUPPORT_LAST_META = {"enabled": False, "support_count": 0}

# Task10I-E edge-aware Gaussian footprint / density repair.
# This adjusts existing edge primitives only; it does not add seam samples,
# duplicate across panels, blend images, or change projection/CUDA behavior.
# Task10I-G baseline hygiene: keep this patch available only as an explicit
# ablation so the default baseline remains Task10H-clean.
EDGE_AWARE_GAUSSIAN_REPAIR_ENABLED = False
EDGE_AWARE_SCALE_MULTIPLIER = 1.45
EDGE_AWARE_SCALE_CAP_MULTIPLIER = 2.0
EDGE_AWARE_COLOR_GRADIENT_PERCENTILE = 88.0
EDGE_AWARE_DEPTH_GRADIENT_PERCENTILE = 90.0
EDGE_AWARE_CANNY_LOW = 32
EDGE_AWARE_CANNY_HIGH = 96
EDGE_AWARE_DILATE_KERNEL_PX = 3

# Task10I-I aggressive representation-contract repair prototype.
# Disabled by default and isolated from Task10E/Task10I-E. The prototype first
# assigns source primitives to representation categories, then adds same-owner
# support only for structural categories. It does not change projection,
# frustums, CUDA, DAV3, display profile, or image-domain blending.
REPRESENTATION_CONTRACT_PROTOTYPE_ENABLED = False
REP_CATEGORY_UNKNOWN = 0
REP_CATEGORY_FACE_INTERIOR = 1
REP_CATEGORY_TEXTURE_ONLY_EDGE = 2
REP_CATEGORY_DEPTH_DISCONTINUITY_EDGE = 3
REP_CATEGORY_SILHOUETTE_EDGE = 4
REP_CATEGORY_FOLD_NEAR_STRUCTURAL_EDGE = 5
REP_CATEGORY_BACKGROUND_CONTROL = 6
REP_CATEGORY_AMBIGUOUS_BOUNDARY = 7
REP_CATEGORY_NAMES = {
    REP_CATEGORY_UNKNOWN: "UNKNOWN",
    REP_CATEGORY_FACE_INTERIOR: "FACE_INTERIOR",
    REP_CATEGORY_TEXTURE_ONLY_EDGE: "TEXTURE_ONLY_EDGE",
    REP_CATEGORY_DEPTH_DISCONTINUITY_EDGE: "DEPTH_DISCONTINUITY_EDGE",
    REP_CATEGORY_SILHOUETTE_EDGE: "SILHOUETTE_EDGE",
    REP_CATEGORY_FOLD_NEAR_STRUCTURAL_EDGE: "FOLD_NEAR_STRUCTURAL_EDGE",
    REP_CATEGORY_BACKGROUND_CONTROL: "BACKGROUND_CONTROL",
    REP_CATEGORY_AMBIGUOUS_BOUNDARY: "AMBIGUOUS_BOUNDARY",
}
REP_CONTRACT_PROTO_MAX_SUPPORT_POINTS = 650_000
REP_CONTRACT_PROTO_FOLD_QUOTA = 0.45
REP_CONTRACT_PROTO_SILHOUETTE_QUOTA = 0.35
REP_CONTRACT_PROTO_DEPTH_QUOTA = 0.20
REP_CONTRACT_PROTO_SUPPORT_OFFSET_PIXELS = 0.55
REP_CONTRACT_PROTO_SUPPORT_SCALE_MULTIPLIER = 1.55
REP_CONTRACT_PROTO_SUPPORT_CAP_MULTIPLIER = 2.4

# Task12 surface / patch-level representation prototype.
# Disabled by default. This prototype appends aggregate patch-family primitives
# from coherent source/world surface tiles; it is not the Task10I-I point-level
# companion support route and it never changes projection/CUDA/DAV3/display
# profile or image-domain output composition.
SURFACE_PATCH_LEVEL_PROTOTYPE_ENABLED = False
SURFACE_PATCH_PROTO_TILE_SOURCE_PX = 14
SURFACE_PATCH_PROTO_MIN_POINTS_PER_PATCH = 6
SURFACE_PATCH_PROTO_MAX_PATCH_POINTS = 180_000
SURFACE_PATCH_PROTO_SCALE_MULTIPLIER = 0.85
SURFACE_PATCH_PROTO_SCALE_CAP_MULTIPLIER = 3.8
SURFACE_PATCH_PROTO_ELIGIBLE_CATEGORIES = (
    REP_CATEGORY_FACE_INTERIOR,
    REP_CATEGORY_TEXTURE_ONLY_EDGE,
    REP_CATEGORY_DEPTH_DISCONTINUITY_EDGE,
    REP_CATEGORY_SILHOUETTE_EDGE,
    REP_CATEGORY_FOLD_NEAR_STRUCTURAL_EDGE,
)

# Fold-zone scale boost: panel_owner==2 points have tighter NN spacing
# (smoothstep blend between two projection planes provides denser sampling).
FOLD_SCALE_BOOST = 1.1       # Increase fold-zone Gaussian scale to fill
                              # gaps from tighter NN spacing at fold.
                              # Reduced from 1.6 proportionally with overlap_factor (2.0→1.0).

# Fold-zone render split: instead of rendering ALL fold-zone Gaussians in
# BOTH frustums (which causes double-counting), each fold-zone Gaussian
# renders in only ONE frustum based on its world X coordinate.
# |X| <= FADE_THRESHOLD_M creates a narrow transition zone where Gaussians
# still render in both frustums to prevent a visible seam.
FADE_THRESHOLD_M = 0.010     # 10mm transition zone at fold split

#
SUPER_SAMPLE_SIZE = 4          # NxN sub-pixel grid per sparse pixel
LOCAL_SPACING_THRESHOLD = 2.0  # 3D spacing > pixel_pitch x this -> flagged as sparse

#          
MAX_GAUSSIANS = 5_000_000

#         
BG_COLOR = torch.tensor([0.05, 0.05, 0.05], device=DEVICE)

#              
OVERLAP_WORLD_M = 0.0    # Task07B: canonical raw panels use the physical fold, no overlap frustum.
BLEND_PIXELS_TARGET = 50  # 2D Alpha         (px),              

#         
OUT_DIR = os.path.join(_PROJ_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# 1   Depth Anything V3      (FP16/BF16, local source)
# ============================================================================
# Model loading delegated to harness_local_dav3_cv2.load_dav3().
# Depth estimation uses predict_depth_fast() which wraps DAV3 inference +
# optional cv2.ximgproc.jointBilateralFilter.


def _build_bg_wall_grid(rgb_fullres, fg_mask, content_w, content_h,
                         panel_px_W, panel_px_H, panel_W, panel_H,
                         sweet_spot, bg_grid_step):
    """
    Build background wall points with anamorphic colors via inverse perceptual mapping.

    Separates the input image into background regions (where fg_mask==False),
    maps them through the virtual box onto the panel walls, and returns a
    grid of wall points with correct anamorphic colors.

    Returns (points_world, colors, x_sign, panel_owner) for BG Gaussians.
    """
    H_img, W_img = rgb_fullres.shape[:2]

    # Build BG image by masking out foreground
    bg_image = rgb_fullres.copy()
    bg_image[fg_mask] = 0  # zero out FG — will be filled by FG Gaussians

    # Anamorphic mapping: background -> panel wall textures
    bg_left_tex, bg_right_tex = map_background_to_panels(
        bg_image, panel_W, panel_H, panel_px_W, panel_px_H, sweet_spot)

    # Subsample grids on each wall
    xs = np.arange(0, panel_px_W, bg_grid_step, dtype=np.float32)
    zs = np.arange(0, panel_px_H, bg_grid_step, dtype=np.float32)
    n_x, n_z = len(xs), len(zs)

    xx, zz = np.meshgrid(xs, zs)
    xx_flat = xx.ravel()
    zz_flat = zz.ravel()

    # World positions on walls (depth = 0 on wall surface)
    u_norm = (xx_flat + 0.5) / float(panel_px_W)
    v_norm = (zz_flat + 0.5) / float(panel_px_H)

    # Left wall: y=0, x in [-panel_W, 0], z in [-panel_H/2, +panel_H/2]
    wx_L = -panel_W + u_norm * panel_W
    wy_L = np.zeros(n_x * n_z, dtype=np.float32)
    wz_L = (0.5 - v_norm) * panel_H
    pts_L = np.stack([wx_L, wy_L, wz_L], axis=-1)
    xs_L = -np.ones(n_x * n_z, dtype=np.int8)
    po_L = np.zeros(n_x * n_z, dtype=np.uint8)  # panel_owner=0 (left only)

    # Right wall: x=0, y in [-panel_W, 0], z in [-panel_H/2, +panel_H/2]
    wx_R = np.zeros(n_x * n_z, dtype=np.float32)
    wy_R = -panel_W + u_norm * panel_W
    wz_R = (0.5 - v_norm) * panel_H
    pts_R = np.stack([wx_R, wy_R, wz_R], axis=-1)
    xs_R = np.ones(n_x * n_z, dtype=np.int8)
    po_R = np.ones(n_x * n_z, dtype=np.uint8)  # panel_owner=1 (right only)

    # Sample colors from the anamorphic textures
    # Map grid positions to texture pixel indices
    col_idx = np.clip(np.floor(xx_flat).astype(np.int32), 0, panel_px_W - 1)
    row_idx = np.clip(np.floor(zz_flat).astype(np.int32), 0, panel_px_H - 1)

    rgb_L = bg_left_tex[row_idx, col_idx, :].astype(np.float32) / 255.0
    rgb_R = bg_right_tex[row_idx, col_idx, :].astype(np.float32) / 255.0

    points_world = np.concatenate([pts_L, pts_R], axis=0)
    colors = np.concatenate([rgb_L, rgb_R], axis=0)
    x_sign = np.concatenate([xs_L, xs_R], axis=0)
    panel_owner = np.concatenate([po_L, po_R], axis=0)

    n_bg = len(points_world)
    print(f"    BG wall grid: {n_bg} points "
          f"(step={bg_grid_step}, {n_x}x{n_z} per panel)")

    return points_world, colors, x_sign, panel_owner


def _task16_surface_tangent_fields(world_x, world_y, world_z):
    """Per-pixel world-space u/v tangents for Task16C surfel Gaussians."""
    tangent_u = np.stack([
        np.gradient(world_x, axis=1),
        np.gradient(world_y, axis=1),
        np.gradient(world_z, axis=1),
    ], axis=-1).astype(np.float32)
    tangent_v = np.stack([
        np.gradient(world_x, axis=0),
        np.gradient(world_y, axis=0),
        np.gradient(world_z, axis=0),
    ], axis=-1).astype(np.float32)
    return tangent_u, tangent_v


def _task16_normalize_vectors_np(vectors, fallback):
    vectors = np.asarray(vectors, dtype=np.float32)
    out = vectors.copy()
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-8)
    out[valid] = out[valid] / norms[valid]
    out[~valid] = np.asarray(fallback, dtype=np.float32)
    return out, valid


def _task16_quaternions_from_rotation_matrices(rot):
    """Convert world-frame rotation matrices to [w, x, y, z] quaternions."""
    rot = np.asarray(rot, dtype=np.float32)
    q = np.empty((rot.shape[0], 4), dtype=np.float32)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]

    m = trace > 0.0
    s = np.sqrt(np.maximum(trace[m] + 1.0, 1e-12)) * 2.0
    q[m, 0] = 0.25 * s
    q[m, 1] = (rot[m, 2, 1] - rot[m, 1, 2]) / s
    q[m, 2] = (rot[m, 0, 2] - rot[m, 2, 0]) / s
    q[m, 3] = (rot[m, 1, 0] - rot[m, 0, 1]) / s

    m1 = (~m) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + rot[m1, 0, 0] - rot[m1, 1, 1] - rot[m1, 2, 2], 1e-12)) * 2.0
    q[m1, 0] = (rot[m1, 2, 1] - rot[m1, 1, 2]) / s
    q[m1, 1] = 0.25 * s
    q[m1, 2] = (rot[m1, 0, 1] + rot[m1, 1, 0]) / s
    q[m1, 3] = (rot[m1, 0, 2] + rot[m1, 2, 0]) / s

    m2 = (~m) & (~m1) & (rot[:, 1, 1] > rot[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + rot[m2, 1, 1] - rot[m2, 0, 0] - rot[m2, 2, 2], 1e-12)) * 2.0
    q[m2, 0] = (rot[m2, 0, 2] - rot[m2, 2, 0]) / s
    q[m2, 1] = (rot[m2, 0, 1] + rot[m2, 1, 0]) / s
    q[m2, 2] = 0.25 * s
    q[m2, 3] = (rot[m2, 1, 2] + rot[m2, 2, 1]) / s

    m3 = (~m) & (~m1) & (~m2)
    s = np.sqrt(np.maximum(1.0 + rot[m3, 2, 2] - rot[m3, 0, 0] - rot[m3, 1, 1], 1e-12)) * 2.0
    q[m3, 0] = (rot[m3, 1, 0] - rot[m3, 0, 1]) / s
    q[m3, 1] = (rot[m3, 0, 2] + rot[m3, 2, 0]) / s
    q[m3, 2] = (rot[m3, 1, 2] + rot[m3, 2, 1]) / s
    q[m3, 3] = 0.25 * s

    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-8)
    q[~np.isfinite(q).all(axis=1)] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q.astype(np.float32)


def _task16_build_surfel_scale_rotation(
        geometry_metadata, per_point_scale, base_scale, max_scale_m,
        overlap_factor, is_super_sampled=None, super_scale_cap=None):
    """Build Task16C anisotropic surfel scales and rotations."""
    stats = {
        "enabled": True,
        "schema": "task16_surfel_gaussian_representation_v1",
        "fallback_used": False,
    }
    if geometry_metadata is None:
        stats["enabled"] = False
        stats["reason"] = "geometry_metadata_missing"
        return None, None, stats

    try:
        tangent_u = np.asarray(geometry_metadata["tangent_u"], dtype=np.float32)
        tangent_v = np.asarray(geometry_metadata["tangent_v"], dtype=np.float32)
        hard_edge = np.asarray(geometry_metadata.get("hard_edge", []), dtype=bool)
    except Exception as exc:
        stats["enabled"] = False
        stats["reason"] = f"invalid_geometry_metadata: {exc}"
        return None, None, stats

    n = int(per_point_scale.shape[0])
    if tangent_u.shape != (n, 3) or tangent_v.shape != (n, 3):
        stats["enabled"] = False
        stats["reason"] = (
            f"metadata_shape_mismatch: tangent_u={tangent_u.shape}, "
            f"tangent_v={tangent_v.shape}, N={n}"
        )
        return None, None, stats
    if hard_edge.shape[0] != n:
        hard_edge = np.zeros(n, dtype=bool)

    u_axis, u_valid = _task16_normalize_vectors_np(tangent_u, [1.0, 0.0, 0.0])
    v_axis_raw = tangent_v - np.sum(tangent_v * u_axis, axis=1, keepdims=True) * u_axis
    v_axis, v_valid = _task16_normalize_vectors_np(v_axis_raw, [0.0, 0.0, 1.0])
    n_axis = np.cross(u_axis, v_axis)
    n_axis, n_valid = _task16_normalize_vectors_np(n_axis, [0.0, 1.0, 0.0])
    v_axis = np.cross(n_axis, u_axis)
    v_axis, v_valid_2 = _task16_normalize_vectors_np(v_axis, [0.0, 0.0, 1.0])

    valid_frame = u_valid & v_valid & n_valid & v_valid_2
    rot = np.stack([u_axis, v_axis, n_axis], axis=2).astype(np.float32)
    quat = _task16_quaternions_from_rotation_matrices(rot)

    tu_len = np.linalg.norm(tangent_u, axis=1)
    tv_len = np.linalg.norm(tangent_v, axis=1)
    tu_len = np.where(np.isfinite(tu_len) & (tu_len > 1e-8), tu_len, base_scale)
    tv_len = np.where(np.isfinite(tv_len) & (tv_len > 1e-8), tv_len, base_scale)
    edge_mul = np.where(hard_edge, 0.72, 1.0).astype(np.float32)

    per_np = per_point_scale.detach().cpu().numpy().astype(np.float32)
    scale_u = np.maximum(tu_len * float(overlap_factor) * 0.78, per_np * 0.58) * edge_mul
    scale_v = np.maximum(tv_len * float(overlap_factor) * 0.78, per_np * 0.58) * edge_mul
    scale_u = np.clip(scale_u, per_np * 0.42, max_scale_m)
    scale_v = np.clip(scale_v, per_np * 0.42, max_scale_m)
    super_cap_stats = {"surfel_super_sampled_tangent_cap_applied": False}
    if is_super_sampled is not None and super_scale_cap is not None:
        super_np = np.asarray(is_super_sampled, dtype=bool).ravel()
        if super_np.shape[0] == n and bool(super_np.any()):
            pre_u = float(scale_u[super_np].mean())
            pre_v = float(scale_v[super_np].mean())
            cap = float(super_scale_cap)
            scale_u[super_np] = np.minimum(scale_u[super_np], cap)
            scale_v[super_np] = np.minimum(scale_v[super_np], cap)
            super_cap_stats = {
                "surfel_super_sampled_tangent_cap_applied": True,
                "surfel_super_sampled_count": int(np.count_nonzero(super_np)),
                "surfel_super_tangent_cap_mm": round(cap * 1000.0, 6),
                "surfel_super_pre_scale_u_mean_mm": round(pre_u * 1000.0, 6),
                "surfel_super_pre_scale_v_mean_mm": round(pre_v * 1000.0, 6),
                "surfel_super_post_scale_u_mean_mm": round(float(scale_u[super_np].mean()) * 1000.0, 6),
                "surfel_super_post_scale_v_mean_mm": round(float(scale_v[super_np].mean()) * 1000.0, 6),
            }
    normal_thickness = float(np.clip(base_scale * 0.12, 0.00008, 0.00075))
    scale_n = np.full(n, normal_thickness, dtype=np.float32)
    scale_vec_np = np.stack([scale_u, scale_v, scale_n], axis=1).astype(np.float32)

    scale_vec = torch.from_numpy(scale_vec_np).to(DEVICE)
    rotation = torch.from_numpy(quat).to(DEVICE)
    stats.update({
        "point_count": n,
        "valid_frame_fraction": round(float(valid_frame.mean()), 6),
        "hard_edge_count": int(hard_edge.sum()),
        "normal_thickness_mm": round(normal_thickness * 1000.0, 6),
        "scale_u_mean_mm": round(float(scale_vec_np[:, 0].mean() * 1000.0), 6),
        "scale_v_mean_mm": round(float(scale_vec_np[:, 1].mean() * 1000.0), 6),
        "scale_n_mean_mm": round(float(scale_vec_np[:, 2].mean() * 1000.0), 6),
        "anisotropy_ratio_mean": round(float(
            np.maximum(scale_vec_np[:, 0], scale_vec_np[:, 1]).mean()
            / max(float(scale_vec_np[:, 2].mean()), 1e-8)), 6),
    })
    stats.update(super_cap_stats)
    return scale_vec, rotation, stats


def _normalize_seam_boundary_support_mode(mode):
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("on", "off"):
        raise ValueError(
            "SEAM_BOUNDARY_SUPPORT_MODE must be 'on' or 'off', "
            f"got {mode!r}")
    return mode_norm


def _active_seam_boundary_support_mode():
    if not bool(SEAM_BOUNDARY_SUPPORT_ENABLED):
        return "off"
    return _normalize_seam_boundary_support_mode(SEAM_BOUNDARY_SUPPORT_MODE)


def _project_points_to_owner_panel_px(points, owners):
    """Project world points to their owning physical panel from SWEET_SPOT."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    owners = np.asarray(owners).reshape(-1)
    n = min(pts.shape[0], owners.shape[0])
    pts = pts[:n]
    owners = owners[:n]
    u = np.full(n, np.nan, dtype=np.float64)
    v = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    eye = np.asarray(SWEET_SPOT, dtype=np.float64).reshape(1, 3)
    ray = pts - eye

    left = owners == 0
    if np.any(left):
        denom = ray[left, 1]
        denom_ok = np.abs(denom) > 1e-9
        t = (0.0 - float(SWEET_SPOT[1])) / np.where(denom_ok, denom, 1.0)
        hit = eye + t[:, None] * ray[left]
        idx = np.nonzero(left)[0]
        u[idx] = (hit[:, 0] + PANEL_PHYS_W_M) / PANEL_PHYS_W_M * float(PANEL_PIXEL_W - 1)
        v[idx] = (hit[:, 2] + PANEL_PHYS_H_M * 0.5) / PANEL_PHYS_H_M * float(PANEL_PIXEL_H - 1)
        valid[idx] = denom_ok & (t > 0.0) & np.isfinite(u[idx]) & np.isfinite(v[idx])

    right = owners == 1
    if np.any(right):
        denom = ray[right, 0]
        denom_ok = np.abs(denom) > 1e-9
        t = (0.0 - float(SWEET_SPOT[0])) / np.where(denom_ok, denom, 1.0)
        hit = eye + t[:, None] * ray[right]
        idx = np.nonzero(right)[0]
        u[idx] = (-hit[:, 1]) / PANEL_PHYS_W_M * float(PANEL_PIXEL_W - 1)
        v[idx] = (hit[:, 2] + PANEL_PHYS_H_M * 0.5) / PANEL_PHYS_H_M * float(PANEL_PIXEL_H - 1)
        valid[idx] = denom_ok & (t > 0.0) & np.isfinite(u[idx]) & np.isfinite(v[idx])

    return u, v, valid


def _observer_panel_tangent_coordinate(panel_coord, normal_disp,
                                       eye_tangent, eye_normal):
    """World tangent coordinate whose observer ray hits panel_coord.

    depth_to_unified_pointcloud stores foreground depth as a normal displacement
    away from the physical panel.  If the unchanged panel tangent coordinate is
    used after that displacement, the sweet-spot ray intersects the physical
    LED plane inward from the intended panel coordinate near the fold.  Solve
    the inverse ray/plane relation instead, preserving point count, colors,
    depth values, opacity, scale, and output resolution.
    """
    panel_coord = np.asarray(panel_coord, dtype=np.float32)
    normal_disp = np.asarray(normal_disp, dtype=np.float32)
    eye_tangent = np.float32(eye_tangent)
    eye_normal = np.float32(eye_normal)
    if abs(float(eye_normal)) < 1e-8:
        return panel_coord
    return eye_tangent + (panel_coord - eye_tangent) * (
        (eye_normal - normal_disp) / eye_normal)


def _row_gap_summary(rows, u_px, side):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    u_px = np.asarray(u_px, dtype=np.float64).reshape(-1)
    if rows.size == 0:
        return {
            "valid_row_count": 0,
            "max_gap_px": float("inf"),
            "p95_gap_px": float("inf"),
            "rows_over_threshold": 0,
            "row_over_threshold_fraction": 1.0,
            "u_min": None,
            "u_max": None,
        }
    if side == "left":
        edge = float(PANEL_PIXEL_W - 1)
        by_row = np.full(PANEL_PIXEL_H, -np.inf, dtype=np.float64)
        np.maximum.at(by_row, np.clip(rows, 0, PANEL_PIXEL_H - 1), u_px)
        valid_rows = np.isfinite(by_row)
        gaps = edge - by_row[valid_rows]
    elif side == "right":
        edge = 0.0
        by_row = np.full(PANEL_PIXEL_H, np.inf, dtype=np.float64)
        np.minimum.at(by_row, np.clip(rows, 0, PANEL_PIXEL_H - 1), u_px)
        valid_rows = np.isfinite(by_row)
        gaps = by_row[valid_rows] - edge
    else:
        raise ValueError(f"unknown side {side!r}")
    if gaps.size == 0:
        return {
            "valid_row_count": 0,
            "max_gap_px": float("inf"),
            "p95_gap_px": float("inf"),
            "rows_over_threshold": 0,
            "row_over_threshold_fraction": 1.0,
            "u_min": None,
            "u_max": None,
        }
    threshold = 1.0
    gaps = np.maximum(gaps, 0.0)
    return {
        "valid_row_count": int(gaps.size),
        "max_gap_px": float(np.max(gaps)),
        "p95_gap_px": float(np.percentile(gaps, 95)),
        "rows_over_threshold": int(np.count_nonzero(gaps > threshold)),
        "row_over_threshold_fraction": float(np.count_nonzero(gaps > threshold) / max(gaps.size, 1)),
        "u_min": float(np.nanmin(u_px)),
        "u_max": float(np.nanmax(u_px)),
    }


def _manual_seam_support_gate(
        world_x_full, world_y_full, world_z_full, panel_owner_map,
        band_px=None):
    """Report manual seam-support state and diagnostic edge gaps."""
    mode = _active_seam_boundary_support_mode()
    band_px = int(SEAM_BOUNDARY_SUPPORT_BAND_PX if band_px is None else band_px)
    base = {
        "schema": "task29r_manual_seam_support_gate_v1",
        "mode": mode,
        "legacy_enabled": bool(SEAM_BOUNDARY_SUPPORT_ENABLED),
        "band_px": int(band_px),
        "threshold_gap_px": 1.0,
        "enabled": False,
        "reason": "not_evaluated",
    }
    if mode == "off" or band_px <= 0:
        base["reason"] = "support_mode_off_or_empty_band"
        return base

    h_img, w_img = world_x_full.shape[:2]
    split_u = w_img // 2
    cols_grid = np.arange(w_img, dtype=np.int32)[None, :]
    seam_mask = (
        ((panel_owner_map == 0) & (cols_grid >= split_u - band_px) & (cols_grid < split_u))
        | ((panel_owner_map == 1) & (cols_grid >= split_u) & (cols_grid < split_u + band_px))
    )
    finite = (
        np.isfinite(world_x_full)
        & np.isfinite(world_y_full)
        & np.isfinite(world_z_full)
        & (np.abs(world_x_full) < 10.0)
        & (np.abs(world_y_full) < 10.0)
        & (np.abs(world_z_full) < 10.0)
    )
    seam_mask &= finite
    rows, cols = np.where(seam_mask)
    if rows.size == 0:
        base.update({
            "enabled": mode == "on",
            "reason": "manual_on_no_valid_seam_samples" if mode == "on" else "no_valid_seam_samples",
            "sample_count": 0,
        })
        return base

    points = np.stack([
        world_x_full[rows, cols],
        world_y_full[rows, cols],
        world_z_full[rows, cols],
    ], axis=-1)
    owners = panel_owner_map[rows, cols].astype(np.uint8)
    u_px, v_px, valid = _project_points_to_owner_panel_px(points, owners)
    del v_px
    left_valid = valid & (owners == 0)
    right_valid = valid & (owners == 1)
    left_summary = _row_gap_summary(rows[left_valid], u_px[left_valid], "left")
    right_summary = _row_gap_summary(rows[right_valid], u_px[right_valid], "right")
    max_gap = max(float(left_summary["max_gap_px"]), float(right_summary["max_gap_px"]))
    p95_gap = max(float(left_summary["p95_gap_px"]), float(right_summary["p95_gap_px"]))
    needs_support = bool(max_gap > 1.0)
    enabled = bool(mode == "on")
    reason = "manual_on" if mode == "on" else "support_mode_off"
    base.update({
        "enabled": enabled,
        "reason": reason,
        "sample_count": int(rows.size),
        "valid_projected_count": int(np.count_nonzero(valid)),
        "support_needed": needs_support,
        "max_gap_px": max_gap,
        "p95_gap_px": p95_gap,
        "left": left_summary,
        "right": right_summary,
        "projection": "sweet_spot_owner_panel_ray_plane_center_gap",
        "radius_proxy_px": 0.0,
    })
    return base


def _build_seam_boundary_support_points(
        rgb_fullres, world_x_full, world_y_full, world_z_full,
        panel_owner_map, fg_mask=None):
    """
    Build owner-preserving support samples on the physical fold boundary.

    Inputs:
        rgb_fullres: RGB uint8 image, shape (H,W,3).
        world_x_full/world_y_full/world_z_full: Per-source-pixel world
            positions in metres, shape (H,W).
        panel_owner_map: uint8 owner ids, shape (H,W), 0=left and 1=right.
        fg_mask: Optional bool foreground mask, shape (H,W). If present,
            support follows the existing foreground selection and does not add
            new segmentation behavior.
    Outputs:
        (points, colors, x_sign, panel_owner, metadata). Points are float32
        metres, colors are float32 RGB in [0,1], and owner ids remain 0 or 1.
    Formula:
        Left-owned support keeps source x/z and sets y=0 on the left panel
        surface. Right-owned support keeps source y/z and sets x=0 on the
        right panel surface. No sample is duplicated into the opposite owner.
    Failure modes:
        Assumes the active seam-exclusive source split and two-panel physical
        fold. Empty arrays are returned when support is disabled.
    """
    empty_points = np.zeros((0, 3), dtype=np.float32)
    empty_colors = np.zeros((0, 3), dtype=np.float32)
    empty_sign = np.zeros((0,), dtype=np.int8)
    empty_owner = np.zeros((0,), dtype=np.uint8)
    empty_source_uv = np.zeros((0, 2), dtype=np.float32)
    support_gate = _manual_seam_support_gate(
        world_x_full, world_y_full, world_z_full, panel_owner_map)
    if not support_gate.get("enabled", False):
        return empty_points, empty_colors, empty_sign, empty_owner, empty_source_uv, {
            "enabled": False,
            "mode": support_gate.get("mode", _active_seam_boundary_support_mode()),
            "manual_support_gate": support_gate,
            "support_count": 0,
        }

    h_img, w_img = rgb_fullres.shape[:2]
    split_u = w_img // 2
    band_px = int(SEAM_BOUNDARY_SUPPORT_BAND_PX)
    cols = np.arange(w_img, dtype=np.int32)[None, :]
    left_support = (
        (panel_owner_map == 0)
        & (cols >= split_u - band_px)
        & (cols < split_u)
    )
    right_support = (
        (panel_owner_map == 1)
        & (cols >= split_u)
        & (cols < split_u + band_px)
    )
    support_mask = left_support | right_support
    if fg_mask is not None:
        support_mask &= fg_mask.astype(bool)

    finite = (
        np.isfinite(world_x_full)
        & np.isfinite(world_y_full)
        & np.isfinite(world_z_full)
        & (np.abs(world_x_full) < 10.0)
        & (np.abs(world_y_full) < 10.0)
        & (np.abs(world_z_full) < 10.0)
    )
    support_mask &= finite
    rows, cols = np.where(support_mask)
    if rows.size == 0:
        return empty_points, empty_colors, empty_sign, empty_owner, empty_source_uv, {
            "enabled": True,
            "mode": support_gate.get("mode", _active_seam_boundary_support_mode()),
            "band_px": band_px,
            "support_count": 0,
            "manual_support_gate": support_gate,
        }

    points = np.stack(
        [
            world_x_full[rows, cols],
            world_y_full[rows, cols],
            world_z_full[rows, cols],
        ],
        axis=-1,
    ).astype(np.float32)
    owners = panel_owner_map[rows, cols].astype(np.uint8)
    left_owner = owners == 0
    right_owner = owners == 1
    points[left_owner, 1] = 0.0
    points[right_owner, 0] = 0.0
    colors = rgb_fullres[rows, cols].astype(np.float32) / 255.0
    x_sign = np.where(
        points[:, 0] > 1e-6,
        1,
        np.where(points[:, 0] < -1e-6, -1, 0),
    ).astype(np.int8)
    source_uv = np.stack([cols, rows], axis=-1).astype(np.float32)
    metadata = {
        "enabled": True,
        "mode": support_gate.get("mode", _active_seam_boundary_support_mode()),
        "band_px": band_px,
        "support_count": int(rows.size),
        "left_support_count": int(np.count_nonzero(left_owner)),
        "right_support_count": int(np.count_nonzero(right_owner)),
        "source_rows": int(h_img),
        "source_cols": int(w_img),
        "owner_preserving": True,
        "cross_panel_duplication": False,
        "surface_rule": "left support y=0; right support x=0",
        "manual_support_gate": support_gate,
    }
    return points, colors, x_sign, owners, source_uv, metadata


def _compute_edge_aware_gaussian_mask(rgb_fullres, depth_norm):
    """Return a source-pixel mask for local edge-aware Gaussian repair."""
    h_img, w_img = rgb_fullres.shape[:2]
    if not EDGE_AWARE_GAUSSIAN_REPAIR_ENABLED:
        return np.zeros((h_img, w_img), dtype=bool), {
            "enabled": False,
            "reason": "EDGE_AWARE_GAUSSIAN_REPAIR_ENABLED is false",
        }

    gray = cv2.cvtColor(rgb_fullres, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    color_grad = np.sqrt(gx * gx + gy * gy)
    finite_color = color_grad[np.isfinite(color_grad)]
    if finite_color.size:
        color_thresh = float(np.percentile(
            finite_color, EDGE_AWARE_COLOR_GRADIENT_PERCENTILE))
    else:
        color_thresh = float("inf")
    color_edge = color_grad >= max(color_thresh, 8.0)

    canny = cv2.Canny(
        gray,
        int(EDGE_AWARE_CANNY_LOW),
        int(EDGE_AWARE_CANNY_HIGH),
    ).astype(bool)

    depth_edge = np.zeros((h_img, w_img), dtype=bool)
    depth_thresh = None
    if depth_norm is not None:
        dn = np.asarray(depth_norm, dtype=np.float32)
        if dn.shape == (h_img, w_img):
            dgx = cv2.Sobel(dn, cv2.CV_32F, 1, 0, ksize=3)
            dgy = cv2.Sobel(dn, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad = np.sqrt(dgx * dgx + dgy * dgy)
            finite_depth = depth_grad[np.isfinite(depth_grad)]
            if finite_depth.size:
                depth_thresh = float(np.percentile(
                    finite_depth, EDGE_AWARE_DEPTH_GRADIENT_PERCENTILE))
                depth_edge = depth_grad >= max(depth_thresh, 0.015)

    edge_mask = color_edge | canny | depth_edge
    kernel_px = max(1, int(EDGE_AWARE_DILATE_KERNEL_PX))
    if kernel_px > 1:
        kernel = np.ones((kernel_px, kernel_px), dtype=np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    edge_count = int(edge_mask.sum())
    total = int(edge_mask.size)
    stats = {
        "enabled": True,
        "family": "edge_aware_existing_gaussian_footprint_density",
        "source_edge_pixels": edge_count,
        "source_pixel_count": total,
        "source_edge_fraction": round(edge_count / max(total, 1), 6),
        "color_gradient_percentile": float(EDGE_AWARE_COLOR_GRADIENT_PERCENTILE),
        "color_gradient_threshold": round(color_thresh, 6)
        if np.isfinite(color_thresh)
        else None,
        "depth_gradient_percentile": float(EDGE_AWARE_DEPTH_GRADIENT_PERCENTILE),
        "depth_gradient_threshold": round(depth_thresh, 6)
        if depth_thresh is not None and np.isfinite(depth_thresh)
        else None,
        "canny_low": int(EDGE_AWARE_CANNY_LOW),
        "canny_high": int(EDGE_AWARE_CANNY_HIGH),
        "dilate_kernel_px": kernel_px,
        "adds_points": False,
        "changes_projection": False,
        "image_domain_repair": False,
    }
    print(f"    Edge-aware repair mask: {edge_count}/{total} source px "
          f"({100.0 * edge_count / max(total, 1):.2f}%)")
    return edge_mask, stats


def _category_count_dict(category_ids):
    """Return JSON-safe counts keyed by representation category name."""
    arr = np.asarray(category_ids, dtype=np.uint8).ravel()
    counts = {}
    for category_id, name in REP_CATEGORY_NAMES.items():
        counts[name] = int(np.count_nonzero(arr == category_id))
    counts["TOTAL"] = int(arr.size)
    return counts


def _compute_representation_category_map(
        rgb_fullres, depth_norm, panel_owner_map, fg_mask=None):
    """
    Classify source pixels into Task10I-H representation categories.

    The classifier is deliberately conservative: it separates face interiors,
    texture-only RGB edges, depth discontinuities, silhouettes, fold-near
    structural edges, and background controls before any prototype support is
    added. Only structural categories are eligible for repair.
    """
    h_img, w_img = rgb_fullres.shape[:2]
    rgb_u8 = np.asarray(rgb_fullres, dtype=np.uint8)
    rgb_f = rgb_u8.astype(np.float32)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    luma = (
        0.2126 * rgb_f[..., 0]
        + 0.7152 * rgb_f[..., 1]
        + 0.0722 * rgb_f[..., 2]
    )
    chroma = rgb_f.max(axis=2) - rgb_f.min(axis=2)

    background_control = (luma >= 218.0) & (chroma <= 36.0) & (sat <= 65.0)
    object_seed = ((sat >= 72.0) | (chroma >= 58.0)) & (luma >= 20.0)
    if fg_mask is not None:
        object_seed |= np.asarray(fg_mask, dtype=bool)
    object_seed &= ~background_control

    close_kernel = np.ones((3, 3), dtype=np.uint8)
    object_mask = cv2.morphologyEx(
        object_seed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    ).astype(bool)
    object_mask = cv2.dilate(
        object_mask.astype(np.uint8),
        close_kernel,
        iterations=1,
    ).astype(bool)
    object_mask &= ~background_control

    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    color_grad = np.sqrt(gx * gx + gy * gy)
    color_samples = color_grad[object_mask & np.isfinite(color_grad)]
    color_thresh = 18.0
    if color_samples.size:
        color_thresh = max(18.0, float(np.percentile(color_samples, 82.0)))
    color_edge = object_mask & (color_grad >= color_thresh)
    canny = cv2.Canny(gray, 36, 108).astype(bool) & object_mask

    depth_edge = np.zeros((h_img, w_img), dtype=bool)
    depth_thresh = None
    if depth_norm is not None:
        dn = np.asarray(depth_norm, dtype=np.float32)
        if dn.shape == (h_img, w_img):
            dgx = cv2.Sobel(dn, cv2.CV_32F, 1, 0, ksize=3)
            dgy = cv2.Sobel(dn, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad = np.sqrt(dgx * dgx + dgy * dgy)
            depth_samples = depth_grad[object_mask & np.isfinite(depth_grad)]
            if depth_samples.size:
                depth_thresh = max(0.018, float(np.percentile(depth_samples, 84.0)))
                depth_edge = object_mask & (depth_grad >= depth_thresh)

    morph_grad = cv2.morphologyEx(
        object_mask.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    silhouette_edge = morph_grad & object_mask

    cols = np.arange(w_img, dtype=np.int32)[None, :]
    split_u = w_img // 2
    fold_band_px = max(4, int(round(0.035 * w_img)))
    fold_near = (
        (np.abs(cols - split_u) <= fold_band_px)
        & object_mask
        & (color_edge | canny | depth_edge | silhouette_edge)
    )

    categories = np.full((h_img, w_img), REP_CATEGORY_UNKNOWN, dtype=np.uint8)
    categories[background_control] = REP_CATEGORY_BACKGROUND_CONTROL

    edge_like = color_edge | canny | depth_edge | silhouette_edge | fold_near
    face_interior = object_mask & ~edge_like
    categories[face_interior] = REP_CATEGORY_FACE_INTERIOR

    texture_only = object_mask & (color_edge | canny) & ~depth_edge & ~silhouette_edge & ~fold_near
    categories[texture_only] = REP_CATEGORY_TEXTURE_ONLY_EDGE

    depth_discontinuity = object_mask & depth_edge & ~silhouette_edge & ~fold_near
    categories[depth_discontinuity] = REP_CATEGORY_DEPTH_DISCONTINUITY_EDGE

    categories[silhouette_edge & object_mask & ~fold_near] = REP_CATEGORY_SILHOUETTE_EDGE
    categories[fold_near] = REP_CATEGORY_FOLD_NEAR_STRUCTURAL_EDGE

    ambiguous = (
        object_mask
        & edge_like
        & (categories == REP_CATEGORY_UNKNOWN)
    )
    categories[ambiguous] = REP_CATEGORY_AMBIGUOUS_BOUNDARY

    unknown_object = object_mask & (categories == REP_CATEGORY_UNKNOWN)
    categories[unknown_object] = REP_CATEGORY_FACE_INTERIOR

    owner_counts = {}
    if panel_owner_map is not None:
        owner_arr = np.asarray(panel_owner_map)
        owner_counts = {
            "left": _category_count_dict(categories[owner_arr == 0]),
            "right": _category_count_dict(categories[owner_arr == 1]),
        }

    stats = {
        "enabled": True,
        "family": "task10i_i_representation_contract_category_map",
        "source_shape": [int(h_img), int(w_img), 3],
        "category_counts": _category_count_dict(categories),
        "panel_owner_category_counts": owner_counts,
        "object_seed_pixels": int(np.count_nonzero(object_seed)),
        "object_mask_pixels": int(np.count_nonzero(object_mask)),
        "background_control_pixels": int(np.count_nonzero(background_control)),
        "fold_band_px": int(fold_band_px),
        "color_gradient_threshold": round(float(color_thresh), 6),
        "depth_gradient_threshold": round(float(depth_thresh), 6)
        if depth_thresh is not None
        else None,
        "changes_projection": False,
        "changes_cuda": False,
        "image_domain_repair": False,
    }
    print("    Representation category map: "
          + ", ".join(
              f"{name}={count}"
              for name, count in stats["category_counts"].items()
              if name != "TOTAL" and count > 0
          ))
    return categories, stats


def _select_representation_support_indices(
        representation_categories, is_bg_point, max_support_points):
    """Select structural source primitives for the Task10I-I prototype."""
    categories = np.asarray(representation_categories, dtype=np.uint8).ravel()
    if is_bg_point is None:
        bg = np.zeros(categories.shape[0], dtype=bool)
    else:
        bg = np.asarray(is_bg_point, dtype=bool).ravel()
        if bg.shape[0] != categories.shape[0]:
            bg = np.zeros(categories.shape[0], dtype=bool)

    quotas = [
        (
            REP_CATEGORY_FOLD_NEAR_STRUCTURAL_EDGE,
            int(round(max_support_points * REP_CONTRACT_PROTO_FOLD_QUOTA)),
        ),
        (
            REP_CATEGORY_SILHOUETTE_EDGE,
            int(round(max_support_points * REP_CONTRACT_PROTO_SILHOUETTE_QUOTA)),
        ),
        (
            REP_CATEGORY_DEPTH_DISCONTINUITY_EDGE,
            int(round(max_support_points * REP_CONTRACT_PROTO_DEPTH_QUOTA)),
        ),
    ]
    selected_parts = []
    selected_counts = {}
    for category_id, quota in quotas:
        candidates = np.flatnonzero((categories == category_id) & ~bg)
        quota = max(0, int(quota))
        if candidates.size > quota > 0:
            pick = np.linspace(0, candidates.size - 1, quota, dtype=np.int64)
            chosen = candidates[pick]
        else:
            chosen = candidates
        selected_parts.append(chosen.astype(np.int64, copy=False))
        selected_counts[REP_CATEGORY_NAMES[category_id]] = int(chosen.size)

    if not selected_parts:
        return np.zeros((0,), dtype=np.int64), selected_counts
    selected = np.concatenate(selected_parts, axis=0)
    if selected.size > max_support_points:
        pick = np.linspace(0, selected.size - 1, max_support_points, dtype=np.int64)
        selected = selected[pick]
    return selected.astype(np.int64, copy=False), selected_counts


def _append_representation_contract_support(
        points_world, colors, is_super_sampled, panel_owner, is_bg_point,
        representation_categories, img_width, content_w,
        prototype_enabled=None):
    """
    Append same-owner structural support primitives for the Task10I-I prototype.

    This is intentionally representation-space only. It adds no cross-panel
    duplicates, samples no pixels from rendered images, and never selects
    FACE_INTERIOR, TEXTURE_ONLY_EDGE, or BACKGROUND_CONTROL categories.
    """
    enabled = (
        bool(REPRESENTATION_CONTRACT_PROTOTYPE_ENABLED)
        if prototype_enabled is None
        else bool(prototype_enabled)
    )
    n_base = int(points_world.shape[0])
    empty_support_mask = np.zeros(n_base, dtype=bool)
    base_stats = {
        "enabled": enabled,
        "family": "task10i_i_aggressive_representation_contract_repair_prototype",
        "adds_points": False,
        "changes_projection": False,
        "changes_frustum": False,
        "changes_cuda": False,
        "image_domain_repair": False,
        "task10e_support_reused": False,
        "task10i_e_edge_scale_reused": False,
        "base_point_count": n_base,
        "final_point_count": n_base,
        "support_point_count": 0,
        "selected_source_points": 0,
        "support_counts_by_category": {},
        "texture_only_edge_support_count": 0,
        "face_interior_support_count": 0,
        "background_support_count": 0,
    }
    if not enabled:
        base_stats["reason"] = "prototype flag disabled"
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_support_mask, base_stats,
        )
    if representation_categories is None:
        base_stats["enabled"] = False
        base_stats["reason"] = "representation categories not provided"
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_support_mask, base_stats,
        )
    if panel_owner is None or img_width is None or content_w is None:
        base_stats["enabled"] = False
        base_stats["reason"] = "panel_owner/img_width/content_w unavailable"
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_support_mask, base_stats,
        )

    categories = np.asarray(representation_categories, dtype=np.uint8).ravel()
    owners = np.asarray(panel_owner, dtype=np.uint8).ravel()
    if categories.shape[0] != n_base or owners.shape[0] != n_base:
        base_stats["enabled"] = False
        base_stats["reason"] = (
            f"category/owner lengths {categories.shape[0]}/{owners.shape[0]} "
            f"do not match points {n_base}"
        )
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_support_mask, base_stats,
        )

    selected, quota_counts = _select_representation_support_indices(
        categories, is_bg_point, int(REP_CONTRACT_PROTO_MAX_SUPPORT_POINTS))
    if selected.size == 0:
        base_stats["enabled"] = True
        base_stats["reason"] = "no eligible structural categories selected"
        base_stats["category_counts"] = _category_count_dict(categories)
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_support_mask, base_stats,
        )

    support_points = points_world[selected].astype(np.float32, copy=True)
    support_colors = colors[selected].astype(np.float32, copy=True)
    support_owners = owners[selected].astype(np.uint8, copy=True)
    support_categories = categories[selected].astype(np.uint8, copy=True)

    pixel_pitch_m = float(content_w) / float(img_width)
    offset_m = pixel_pitch_m * float(REP_CONTRACT_PROTO_SUPPORT_OFFSET_PIXELS)
    support_index = np.arange(selected.size, dtype=np.int64)
    direction = np.where((support_index % 2) == 0, -1.0, 1.0).astype(np.float32)

    fold = support_categories == REP_CATEGORY_FOLD_NEAR_STRUCTURAL_EDGE
    left = support_owners == 0
    right = support_owners == 1
    support_points[fold & left, 0] -= offset_m
    support_points[fold & right, 1] -= offset_m

    non_fold = ~fold
    left_non_fold = non_fold & left
    right_non_fold = non_fold & right
    support_points[left_non_fold, 0] += direction[left_non_fold] * offset_m * 0.45
    support_points[right_non_fold, 1] += direction[right_non_fold] * offset_m * 0.45

    points_out = np.concatenate([points_world, support_points], axis=0)
    colors_out = np.concatenate([colors, support_colors], axis=0)
    panel_owner_out = np.concatenate([owners, support_owners], axis=0)
    categories_out = np.concatenate([categories, support_categories], axis=0)

    if is_super_sampled is None:
        is_super_out = None
    else:
        is_super_out = np.concatenate([
            np.asarray(is_super_sampled, dtype=bool).ravel(),
            np.zeros(selected.size, dtype=bool),
        ])
    if is_bg_point is None:
        is_bg_out = None
    else:
        is_bg_out = np.concatenate([
            np.asarray(is_bg_point, dtype=bool).ravel(),
            np.zeros(selected.size, dtype=bool),
        ])

    support_mask = np.concatenate([
        np.zeros(n_base, dtype=bool),
        np.ones(selected.size, dtype=bool),
    ])
    support_counts = _category_count_dict(support_categories)
    base_stats.update({
        "enabled": True,
        "adds_points": True,
        "base_point_count": n_base,
        "final_point_count": int(points_out.shape[0]),
        "selected_source_points": int(selected.size),
        "support_point_count": int(selected.size),
        "support_counts_by_category": {
            key: value
            for key, value in support_counts.items()
            if key != "TOTAL" and value > 0
        },
        "selection_quota_counts": quota_counts,
        "category_counts_before_support": _category_count_dict(categories),
        "category_counts_after_support": _category_count_dict(categories_out),
        "support_offset_m": round(offset_m, 9),
        "support_offset_source_pixels": float(REP_CONTRACT_PROTO_SUPPORT_OFFSET_PIXELS),
        "owner_preserving": True,
        "cross_panel_duplication": False,
        "texture_only_edge_support_count": int(
            np.count_nonzero(support_categories == REP_CATEGORY_TEXTURE_ONLY_EDGE)),
        "face_interior_support_count": int(
            np.count_nonzero(support_categories == REP_CATEGORY_FACE_INTERIOR)),
        "background_support_count": int(
            np.count_nonzero(support_categories == REP_CATEGORY_BACKGROUND_CONTROL)),
    })
    print("    Representation-contract prototype support: "
          f"{selected.size} same-owner structural pts "
          f"(offset={offset_m * 1000.0:.3f}mm)")
    return (
        points_out, colors_out, is_super_out, panel_owner_out, is_bg_out,
        categories_out, support_mask, base_stats,
    )


def _append_surface_patch_level_prototype(
        points_world, colors, is_super_sampled, panel_owner, is_bg_point,
        representation_categories, img_width, content_w, exclude_mask=None,
        prototype_enabled=None):
    """
    Append aggregate surface/patch primitives for Task12.

    A patch is a coherent source/world surface tile with a shared owner and
    representation category. The generated primitive is the aggregate centroid
    and mean color of that tile. This deliberately differs from Task10I-I:
    it does not add companion points near selected edge samples and does not
    duplicate across panels.
    """
    enabled = (
        bool(SURFACE_PATCH_LEVEL_PROTOTYPE_ENABLED)
        if prototype_enabled is None
        else bool(prototype_enabled)
    )
    n_base = int(points_world.shape[0])
    empty_patch_mask = np.zeros(n_base, dtype=bool)
    empty_patch_ids = np.full(n_base, -1, dtype=np.int32)
    stats = {
        "enabled": enabled,
        "family": "task12_surface_patch_level_representation_prototype",
        "strategy_families": [
            "A_face_patch_preservation_prototype",
            "B_surface_region_primitive_family_prototype",
        ],
        "representation_unit": "owner_category_surface_tile_patch",
        "adds_points": False,
        "changes_projection": False,
        "changes_frustum": False,
        "changes_cuda": False,
        "changes_dav3": False,
        "changes_display_profile": False,
        "image_domain_repair": False,
        "task10e_support_reused": False,
        "task10i_i_point_companion_support_reused": False,
        "task10i_e_edge_scale_reused": False,
        "base_point_count": n_base,
        "final_point_count": n_base,
        "patch_point_count": 0,
        "patch_counts_by_category": {},
        "patch_counts_by_owner": {},
        "texture_only_patch_count": 0,
        "face_interior_patch_count": 0,
        "background_patch_count": 0,
    }
    if not enabled:
        stats["reason"] = "prototype flag disabled"
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_patch_mask, empty_patch_ids, stats,
        )
    if representation_categories is None:
        stats["enabled"] = False
        stats["reason"] = "representation categories not provided"
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_patch_mask, empty_patch_ids, stats,
        )
    if panel_owner is None or img_width is None or content_w is None:
        stats["enabled"] = False
        stats["reason"] = "panel_owner/img_width/content_w unavailable"
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_patch_mask, empty_patch_ids, stats,
        )

    categories = np.asarray(representation_categories, dtype=np.uint8).ravel()
    owners = np.asarray(panel_owner, dtype=np.uint8).ravel()
    if categories.shape[0] != n_base or owners.shape[0] != n_base:
        stats["enabled"] = False
        stats["reason"] = (
            f"category/owner lengths {categories.shape[0]}/{owners.shape[0]} "
            f"do not match points {n_base}"
        )
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_patch_mask, empty_patch_ids, stats,
        )

    bg = np.zeros(n_base, dtype=bool)
    if is_bg_point is not None:
        bg_np = np.asarray(is_bg_point, dtype=bool).ravel()
        if bg_np.shape[0] == n_base:
            bg = bg_np
    super_mask = np.zeros(n_base, dtype=bool)
    if is_super_sampled is not None:
        super_np = np.asarray(is_super_sampled, dtype=bool).ravel()
        if super_np.shape[0] == n_base:
            super_mask = super_np
    excluded = np.zeros(n_base, dtype=bool)
    if exclude_mask is not None:
        ex_np = np.asarray(exclude_mask, dtype=bool).ravel()
        if ex_np.shape[0] == n_base:
            excluded = ex_np

    eligible_category = np.isin(
        categories,
        np.asarray(SURFACE_PATCH_PROTO_ELIGIBLE_CATEGORIES, dtype=np.uint8),
    )
    eligible = eligible_category & ~bg & ~super_mask & ~excluded & (
        (owners == 0) | (owners == 1)
    )
    eligible_indices = np.flatnonzero(eligible)
    stats["eligible_source_points"] = int(eligible_indices.size)
    if eligible_indices.size == 0:
        stats["reason"] = "no eligible surface/category points"
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_patch_mask, empty_patch_ids, stats,
        )

    pixel_pitch_m = float(content_w) / float(max(img_width, 1))
    tile_m = pixel_pitch_m * float(SURFACE_PATCH_PROTO_TILE_SOURCE_PX)
    tile_m = max(tile_m, 1e-6)
    pts = np.asarray(points_world, dtype=np.float32)
    elig_pts = pts[eligible_indices]
    elig_owners = owners[eligible_indices].astype(np.int64)
    elig_categories = categories[eligible_indices].astype(np.int64)

    # Owner-local panel coordinates: left uses X, right uses Y, both paired
    # with Z. These coordinates define a surface-region tile without changing
    # the frozen geometry mapping.
    surface_u = np.where(
        elig_owners == 0,
        elig_pts[:, 0] + float(content_w),
        -elig_pts[:, 1],
    )
    surface_v = elig_pts[:, 2] + float(PANEL_PHYS_H_M) * 0.5
    u_bin = np.floor(surface_u / tile_m).astype(np.int64)
    v_bin = np.floor(surface_v / tile_m).astype(np.int64)
    u_bin = np.clip(u_bin, 0, (1 << 20) - 1)
    v_bin = np.clip(v_bin, 0, (1 << 20) - 1)

    keys = (
        (elig_categories << 56)
        | (elig_owners << 52)
        | (u_bin << 26)
        | v_bin
    )
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    valid_groups = np.flatnonzero(
        counts >= int(SURFACE_PATCH_PROTO_MIN_POINTS_PER_PATCH))
    if valid_groups.size == 0:
        stats["reason"] = "no surface tiles met min_points_per_patch"
        stats["candidate_patch_count"] = int(unique_keys.size)
        stats["tile_source_px"] = int(SURFACE_PATCH_PROTO_TILE_SOURCE_PX)
        return (
            points_world, colors, is_super_sampled, panel_owner, is_bg_point,
            representation_categories, empty_patch_mask, empty_patch_ids, stats,
        )

    if valid_groups.size > int(SURFACE_PATCH_PROTO_MAX_PATCH_POINTS):
        pick = np.linspace(
            0, valid_groups.size - 1,
            int(SURFACE_PATCH_PROTO_MAX_PATCH_POINTS),
            dtype=np.int64,
        )
        valid_groups = valid_groups[pick]
    group_selected = np.zeros(unique_keys.shape[0], dtype=bool)
    group_selected[valid_groups] = True
    selected_point_mask = group_selected[inverse]
    selected_indices = eligible_indices[selected_point_mask]
    selected_inverse = inverse[selected_point_mask]
    compact_group = np.full(unique_keys.shape[0], -1, dtype=np.int64)
    compact_group[valid_groups] = np.arange(valid_groups.size, dtype=np.int64)
    compact_ids = compact_group[selected_inverse]
    patch_count = int(valid_groups.size)

    patch_counts = np.bincount(compact_ids, minlength=patch_count).astype(np.float32)
    sum_xyz = np.zeros((patch_count, 3), dtype=np.float64)
    sum_rgb = np.zeros((patch_count, 3), dtype=np.float64)
    np.add.at(sum_xyz, compact_ids, pts[selected_indices].astype(np.float64))
    np.add.at(sum_rgb, compact_ids, np.asarray(colors, dtype=np.float32)[selected_indices])
    patch_points = (sum_xyz / patch_counts[:, None]).astype(np.float32)
    patch_colors = (sum_rgb / patch_counts[:, None]).astype(np.float32)

    selected_keys = unique_keys[valid_groups]
    patch_categories = ((selected_keys >> 56) & 0xF).astype(np.uint8)
    patch_owners = ((selected_keys >> 52) & 0xF).astype(np.uint8)
    patch_ids_new = np.arange(patch_count, dtype=np.int32)

    points_out = np.concatenate([points_world, patch_points], axis=0)
    colors_out = np.concatenate([colors, patch_colors], axis=0)
    panel_owner_out = np.concatenate([owners, patch_owners], axis=0)
    categories_out = np.concatenate([categories, patch_categories], axis=0)
    patch_mask = np.concatenate([
        np.zeros(n_base, dtype=bool),
        np.ones(patch_count, dtype=bool),
    ])
    patch_ids = np.concatenate([empty_patch_ids, patch_ids_new], axis=0)

    if is_super_sampled is None:
        is_super_out = None
    else:
        is_super_out = np.concatenate([
            np.asarray(is_super_sampled, dtype=bool).ravel(),
            np.zeros(patch_count, dtype=bool),
        ])
    if is_bg_point is None:
        is_bg_out = None
    else:
        is_bg_out = np.concatenate([
            np.asarray(is_bg_point, dtype=bool).ravel(),
            np.zeros(patch_count, dtype=bool),
        ])

    category_counts = _category_count_dict(patch_categories)
    owner_counts = {
        "left": int(np.count_nonzero(patch_owners == 0)),
        "right": int(np.count_nonzero(patch_owners == 1)),
    }
    stats.update({
        "enabled": True,
        "adds_points": True,
        "base_point_count": n_base,
        "final_point_count": int(points_out.shape[0]),
        "candidate_patch_count": int(unique_keys.size),
        "patch_point_count": patch_count,
        "selected_source_points": int(selected_indices.size),
        "tile_source_px": int(SURFACE_PATCH_PROTO_TILE_SOURCE_PX),
        "tile_size_m": round(float(tile_m), 9),
        "min_points_per_patch": int(SURFACE_PATCH_PROTO_MIN_POINTS_PER_PATCH),
        "patch_counts_by_category": {
            key: value
            for key, value in category_counts.items()
            if key != "TOTAL" and value > 0
        },
        "patch_counts_by_owner": owner_counts,
        "mean_source_points_per_patch": round(
            float(np.mean(patch_counts)) if patch_counts.size else 0.0, 6),
        "min_source_points_per_patch": int(np.min(patch_counts)) if patch_counts.size else 0,
        "max_source_points_per_patch": int(np.max(patch_counts)) if patch_counts.size else 0,
        "texture_only_patch_count": int(
            np.count_nonzero(patch_categories == REP_CATEGORY_TEXTURE_ONLY_EDGE)),
        "face_interior_patch_count": int(
            np.count_nonzero(patch_categories == REP_CATEGORY_FACE_INTERIOR)),
        "background_patch_count": int(
            np.count_nonzero(patch_categories == REP_CATEGORY_BACKGROUND_CONTROL)),
        "owner_preserving": True,
        "cross_panel_duplication": False,
    })
    print("    Surface/patch prototype: "
          f"{patch_count} patch-family pts from {selected_indices.size} "
          f"source pts (tile={SURFACE_PATCH_PROTO_TILE_SOURCE_PX}px)")
    return (
        points_out, colors_out, is_super_out, panel_owner_out, is_bg_out,
        categories_out, patch_mask, patch_ids, stats,
    )


# ============================================================================
# 2          +      ->    3D
# ============================================================================
# Box projection geometry (2026-05-21):
# Image split at W/2. Each half projected to its panel-parallel plane.
#   Left half:  Y=depth plane (parallel to left panel Y=0), X in [-panel_w, 0]
#   Right half: X=depth plane (parallel to right panel X=0), Y in [-panel_w, 0]
# Blend zone: smooth transition across the fold (20% of image width).
# Depth displacement: per-panel normal direction (Y for left, X for right).


def _finite_bounded_xyz_mask(x, y, z, limit=10.0):
    """Return the existing finite and strict XYZ-range validity predicate.

    Inputs are same-shaped numeric arrays. The output is a same-shaped boolean
    mask equivalent to ``isfinite(stack).all() & (abs(axis) < limit)``.
    Two reusable boolean buffers avoid the temporary float ``(..., 3)`` stack.
    Shape mismatches raise ``ValueError``; inputs are never modified.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    z_arr = np.asarray(z)
    if x_arr.shape != y_arr.shape or x_arr.shape != z_arr.shape:
        raise ValueError(
            "x, y, and z must have identical shapes, got "
            f"{x_arr.shape}, {y_arr.shape}, and {z_arr.shape}")

    valid = np.empty(x_arr.shape, dtype=bool)
    scratch = np.empty(x_arr.shape, dtype=bool)
    np.isfinite(x_arr, out=valid)
    for coordinate in (y_arr, z_arr):
        np.isfinite(coordinate, out=scratch)
        np.logical_and(valid, scratch, out=valid)
    for coordinate in (x_arr, y_arr, z_arr):
        np.greater(coordinate, -limit, out=scratch)
        np.logical_and(valid, scratch, out=valid)
        np.less(coordinate, limit, out=scratch)
        np.logical_and(valid, scratch, out=valid)
    return valid


def depth_to_unified_pointcloud(rgb_fullres, depth_fullres,
                                 depth_near, depth_far,
                                 content_w, content_h,
                                 depth_norm=None,
                                 depth_power=1.0,
                                 fg_mask=None,
                                 bg_grid_step=4,
                                 hard_edge_mask=None,
                                 return_edge_aware_mask=False,
                                 return_representation_categories=False,
                                 return_geometry_metadata=False):
    """
        RGB image + depth map -> unified 3D point cloud (Z-up world coords)

         (Fix 3):
      content_w / content_h == W_img / H_img
            X-Y span ratio equals image aspect ratio.

    depth_norm: optional pre-computed normalized depth [0,1] (from DepthAnchorStream)
                          When provided, skips per-frame Min-Max normalization.

    fg_mask: optional (H,W) bool array, True=foreground. When provided:
        - FG pixels use hybrid 45-degree plane projection (DAV3 depth).
        - BG pixels are placed on panel walls with anamorphic colors from
          inverse perceptual mapping (virtual box B_virtual).

    bg_grid_step: subsampling step for BG wall point grid (default 4).
    return_representation_categories: when True, append a per-point
        Task10I-H representation category id as the final return value.
    return_geometry_metadata: when True, append Task16C per-point source uv,
        local u/v tangent vectors, hard-edge flags, and panel ownership.  This
        is metadata for Gaussian representation only; it does not alter the
        projection formula.
    """
    global SEAM_BOUNDARY_SUPPORT_LAST_META
    H_img, W_img = rgb_fullres.shape[:2]
    print(f"\n[2/7] Back-projecting {W_img}x{H_img} image -> unified 3D point cloud")
    print(f"    Content plane: {content_w:.3f}x{content_h:.3f} m")
    print(f"    Aspect check: content_w/content_h = {content_w/content_h:.5f}")
    print(f"                  W_img/H_img       = {W_img/H_img:.5f}")

    #             
    #        : depth_norm   [0,1], 0=   1=  
    #        : depth_power > 1        (0.01->0.0001, 0.1->0.01)
    #               3D    ,        Scale

    if depth_norm is not None:
        # Pre-anchored normalized depth   skip per-frame min-max to avoid
        # "breathing jelly effect" in video sequences.
        depth_norm_compressed = np.power(depth_norm, depth_power)
        disp_phys = depth_far + depth_norm_compressed * (depth_near - depth_far)
        print(f"    Depth (anchored): norm=[{depth_norm.min():.3f}, {depth_norm.max():.3f}]"
              f"  ->  compressed=[{depth_norm_compressed.min():.3f}, {depth_norm_compressed.max():.3f}]"
              f"  ->  physical: [{disp_phys.min():.3f}, {disp_phys.max():.3f}] m")
    else:
        # Legacy per-frame min-max (fine for single images).
        d_min = float(depth_fullres.min())
        d_max = float(depth_fullres.max())
        if d_max - d_min < 1e-6:
            depth_norm = np.zeros_like(depth_fullres)
        else:
            depth_norm = (depth_fullres - d_min) / (d_max - d_min)   # 0=far, 1=near
        #      :           
        depth_norm_compressed = np.power(depth_norm, depth_power)
        disp_phys = depth_far + depth_norm_compressed * (depth_near - depth_far)  # far->near
        print(f"    Depth raw: [{d_min:.2f}, {d_max:.2f}]  "
              f"->  compressed norm=[{depth_norm_compressed.min():.3f}, {depth_norm_compressed.max():.3f}]"
              f"  ->  physical: [{disp_phys.min():.3f}, {disp_phys.max():.3f}] m")

    #           
    u = np.arange(W_img, dtype=np.float32)
    v = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    # Box projection: split at fold, project halves to panel-parallel planes
    # Left half -> Y=depth plane (parallel to left panel Y=0)
    # Right half -> X=depth plane (parallel to right panel X=0)
    v_norm = vv / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * content_h

    # Seam-exclusive box projection.
    # The fold is a physical boundary between two panels, not a blended 3D
    # strip. Each source half maps only to its own panel coordinate system.
    split_u = W_img // 2
    blend_px = 0

    world_x = np.zeros((H_img, W_img), dtype=np.float32)
    world_y = np.zeros((H_img, W_img), dtype=np.float32)
    # Canonical row/Z contract: image row 0 is the physical top (+Z).
    world_z = vert_pos
    left_mask = uu < split_u

    u_left_all = np.clip(uu.astype(np.float32) / max(split_u - 1, 1), 0.0, 1.0)
    x_left_panel = -content_w + u_left_all * content_w
    y_left = +disp_phys
    x_left = _observer_panel_tangent_coordinate(
        x_left_panel, y_left, SWEET_SPOT[0], SWEET_SPOT[1])

    u_right_all_n = max(W_img - split_u - 1, 1)
    u_right_all = np.clip((uu.astype(np.float32) - split_u) / u_right_all_n, 0.0, 1.0)
    x_right = +disp_phys
    y_right_panel = -u_right_all * content_w
    y_right = _observer_panel_tangent_coordinate(
        y_right_panel, x_right, SWEET_SPOT[1], SWEET_SPOT[0])

    world_x = np.where(left_mask, x_left, x_right).astype(np.float32)
    world_y = np.where(left_mask, y_left, y_right).astype(np.float32)

    # Pure box projection — content placed on panel-parallel planes.
    # No 45-degree plane or hybrid blending.
    # Symmetric depth range (±0.15m) gives balanced X distribution:
    # left half → X ∈ [-1.28, 0], right half → X crosses zero at dn=0.5.

    print("    Fold mapping: seam-exclusive physical panel halves (0px projection blend)")

    # Panel ownership based on input image half (NOT world_x).
    # Each Gaussian renders in exactly one frustum — the panel corresponding
    # to its original image half. This keeps the fold as an exclusive
    # physical boundary instead of double-owning or blending seam pixels.
    panel_owner = np.where(uu < split_u, 0, 1).astype(np.uint8)
    # panel_owner: 0=left_only, 1=right_only (never both)

    if return_edge_aware_mask:
        edge_aware_full, edge_aware_meta = _compute_edge_aware_gaussian_mask(
            rgb_fullres, depth_norm)
    else:
        edge_aware_full = None
        edge_aware_meta = {"enabled": False, "reason": "not requested"}

    if return_representation_categories:
        representation_category_full, representation_category_meta = (
            _compute_representation_category_map(
                rgb_fullres, depth_norm, panel_owner, fg_mask=fg_mask)
        )
    else:
        representation_category_full = None
        representation_category_meta = {
            "enabled": False,
            "reason": "not requested",
        }

    # X    (        X=0            )
    x_sign_map = np.where(world_x > 1e-6, 1, np.where(world_x < -1e-6, -1, 0)).astype(np.int8)

    #       :      (step=1,          )   
    total_pixels = H_img * W_img
    step = 1
    assert total_pixels <= MAX_GAUSSIANS, \
        f"MAX_GAUSSIANS ({MAX_GAUSSIANS}) < total pixels ({total_pixels}). Increase MAX_GAUSSIANS."
    print(f"    Full resolution: step=1 ({total_pixels} pixels -> {total_pixels} gaussians)")

    # Save full-resolution 2D arrays for adaptive super-sampling
    world_x_full = world_x.copy()
    world_y_full = world_y.copy()
    world_z_full = world_z.copy()
    x_sign_full = x_sign_map.copy()
    if return_geometry_metadata:
        tangent_u_full, tangent_v_full = _task16_surface_tangent_fields(
            world_x_full, world_y_full, world_z_full)
        if hard_edge_mask is None:
            hard_edge_full = np.zeros((H_img, W_img), dtype=bool)
        else:
            hard_edge_arr = np.asarray(hard_edge_mask, dtype=bool)
            if hard_edge_arr.shape != (H_img, W_img):
                hard_edge_arr = cv2.resize(
                    hard_edge_arr.astype(np.uint8), (W_img, H_img),
                    interpolation=cv2.INTER_NEAREST) > 0
            hard_edge_full = hard_edge_arr
        src_u_full = uu.astype(np.float32)
        src_v_full = vv.astype(np.float32)

    wx = world_x[::step, ::step].ravel()
    wy = world_y[::step, ::step].ravel()
    wz = world_z[::step, ::step].ravel()
    rgb_s = rgb_fullres[::step, ::step].astype(np.float32).reshape(-1, 3) / 255.0
    xs = x_sign_map[::step, ::step].ravel()
    po = panel_owner[::step, ::step].ravel()
    edge_s = None
    if return_edge_aware_mask:
        edge_s = edge_aware_full[::step, ::step].ravel()
    category_s = None
    if return_representation_categories:
        category_s = representation_category_full[::step, ::step].ravel()

    valid = _finite_bounded_xyz_mask(wx, wy, wz)
    points_world = np.stack([wx[valid], wy[valid], wz[valid]], axis=-1)
    colors = rgb_s[valid]
    x_sign_flat = xs[valid]
    panel_owner_flat = po[valid]
    if return_edge_aware_mask:
        edge_aware_flat = edge_s[valid].astype(bool)
    else:
        edge_aware_flat = None
    if return_representation_categories:
        representation_categories_flat = category_s[valid].astype(np.uint8)
    else:
        representation_categories_flat = None
    is_super_sampled = np.zeros(points_world.shape[0], dtype=bool)
    if return_geometry_metadata:
        source_uv_flat = np.stack([
            src_u_full[::step, ::step].ravel()[valid],
            src_v_full[::step, ::step].ravel()[valid],
        ], axis=-1).astype(np.float32)
        tangent_u_flat = tangent_u_full[::step, ::step].reshape(-1, 3)[valid].astype(np.float32)
        tangent_v_flat = tangent_v_full[::step, ::step].reshape(-1, 3)[valid].astype(np.float32)
        hard_edge_flat = hard_edge_full[::step, ::step].ravel()[valid].astype(bool)

    #    Adaptive Super-sampling:      3D               
    #                3D            
    #     3D      pixel_pitch x LOCAL_SPACING_THRESHOLD            
    #                
    #                           
    n_sub_offsets = SUPER_SAMPLE_SIZE * SUPER_SAMPLE_SIZE  # 16

    # Compute per-pixel local 3D spacing: max of forward differences in u,v directions
    dx_du = np.abs(np.diff(world_x_full, axis=1))
    dx_dv = np.abs(np.diff(world_x_full, axis=0))
    dy_du = np.abs(np.diff(world_y_full, axis=1))
    dy_dv = np.abs(np.diff(world_y_full, axis=0))
    dz_du = np.abs(np.diff(world_z_full, axis=1))
    dz_dv = np.abs(np.diff(world_z_full, axis=0))
    # Pad to restore full (H,W) shape
    spacing_u = np.sqrt(dx_du**2 + dy_du**2 + dz_du**2)  # (H, W-1)
    spacing_v = np.sqrt(dx_dv**2 + dy_dv**2 + dz_dv**2)  # (H-1, W)
    spacing_u = np.pad(spacing_u, ((0, 0), (0, 1)), mode="edge")  # (H, W)
    spacing_v = np.pad(spacing_v, ((0, 1), (0, 0)), mode="edge")  # (H, W)
    spacing_local = np.maximum(spacing_u, spacing_v)  # (H, W)

    # Adaptive threshold: pixel_pitch x LOCAL_SPACING_THRESHOLD
    pixel_pitch_3d = content_w / float(W_img)
    spacing_threshold = pixel_pitch_3d * LOCAL_SPACING_THRESHOLD
    adaptive_mask = spacing_local > spacing_threshold

    # Exclude blend-zone pixels from adaptive super-sampling.
    # The 90-degree fold creates artificially large 3D spacing between
    # adjacent pixels on opposite projection planes. Super-sampling
    # these pixels creates many same-color Gaussians at slightly different
    # 3D positions, causing transition inflation during rasterization.
    split_u = W_img // 2
    seam_exclusion_px = 1
    seam_zone_mask = np.abs(uu - split_u) <= seam_exclusion_px
    adaptive_mask = adaptive_mask & ~seam_zone_mask

    n_masked = int(adaptive_mask.sum())

    if n_masked > 0:
        mask_rows, mask_cols = np.where(adaptive_mask)  # (n_masked,) arrays

        sub_offsets_1d = np.linspace(-0.375, 0.375, SUPER_SAMPLE_SIZE, dtype=np.float32)
        du_grid, dv_grid = np.meshgrid(sub_offsets_1d, sub_offsets_1d)
        du_flat = du_grid.ravel()
        dv_flat = dv_grid.ravel()

        n_super = n_masked * n_sub_offsets

        super_wx = np.empty(n_super, dtype=np.float32)
        super_wy = np.empty(n_super, dtype=np.float32)
        super_wz = np.empty(n_super, dtype=np.float32)
        super_rgb = np.empty((n_super, 3), dtype=np.float32)
        super_xs = np.empty(n_super, dtype=np.int8)
        super_po = np.empty(n_super, dtype=np.uint8)
        if return_geometry_metadata:
            super_source_uv = np.empty((n_super, 2), dtype=np.float32)
            super_tangent_u = np.empty((n_super, 3), dtype=np.float32)
            super_tangent_v = np.empty((n_super, 3), dtype=np.float32)
            super_hard_edge = np.empty(n_super, dtype=bool)
        if return_edge_aware_mask:
            super_edge = np.empty(n_super, dtype=bool)
        if return_representation_categories:
            super_category = np.empty(n_super, dtype=np.uint8)

        mask_cols_f = mask_cols.astype(np.float32)
        mask_rows_f = mask_rows.astype(np.float32)
        # Metadata below is invariant across sub-pixel offsets. Gather it once
        # and broadcast into the existing offset-major output layout instead
        # of repeating the same fancy-index operation 16 times.
        source_xs = x_sign_full[mask_rows, mask_cols]
        source_po = panel_owner[mask_rows, mask_cols]
        super_xs.reshape(n_sub_offsets, n_masked)[:] = source_xs
        super_po.reshape(n_sub_offsets, n_masked)[:] = source_po
        if return_geometry_metadata:
            source_tangent_u = tangent_u_full[mask_rows, mask_cols]
            source_tangent_v = tangent_v_full[mask_rows, mask_cols]
            source_hard_edge = hard_edge_full[mask_rows, mask_cols]
            super_tangent_u.reshape(n_sub_offsets, n_masked, 3)[:] = source_tangent_u
            super_tangent_v.reshape(n_sub_offsets, n_masked, 3)[:] = source_tangent_v
            super_hard_edge.reshape(n_sub_offsets, n_masked)[:] = source_hard_edge
        if return_edge_aware_mask:
            source_edge = edge_aware_full[mask_rows, mask_cols]
            super_edge.reshape(n_sub_offsets, n_masked)[:] = source_edge
        if return_representation_categories:
            source_category = representation_category_full[mask_rows, mask_cols]
            super_category.reshape(n_sub_offsets, n_masked)[:] = source_category

        for k in range(n_sub_offsets):
            sub_u = np.clip(mask_cols_f + du_flat[k], 0.0, W_img - 1.001)
            sub_v = np.clip(mask_rows_f + dv_flat[k], 0.0, H_img - 1.001)

            u0 = np.floor(sub_u).astype(np.int32)
            v0 = np.floor(sub_v).astype(np.int32)
            u1 = np.clip(u0 + 1, 0, W_img - 1)
            v1 = np.clip(v0 + 1, 0, H_img - 1)

            wu = sub_u - u0.astype(np.float32)
            wv = sub_v - v0.astype(np.float32)
            w00 = (1.0 - wu) * (1.0 - wv)
            w10 = wu * (1.0 - wv)
            w01 = (1.0 - wu) * wv
            w11 = wu * wv

            sl = slice(k * n_masked, (k + 1) * n_masked)

            super_wx[sl] = (w00 * world_x_full[v0, mask_cols] +
                            w10 * world_x_full[v0, u1] +
                            w01 * world_x_full[v1, mask_cols] +
                            w11 * world_x_full[v1, u1])
            super_wy[sl] = (w00 * world_y_full[v0, mask_cols] +
                            w10 * world_y_full[v0, u1] +
                            w01 * world_y_full[v1, mask_cols] +
                            w11 * world_y_full[v1, u1])
            super_wz[sl] = (w00 * world_z_full[v0, mask_cols] +
                            w10 * world_z_full[v0, u1] +
                            w01 * world_z_full[v1, mask_cols] +
                            w11 * world_z_full[v1, u1])

            # Nearest-neighbor color sampling: preserves binary colors at
            # checkerboard boundaries, eliminating gray-intermediate Gaussians
            # that cause transition inflation (TIR 1.9x → ~1.3x at 128p).
            # 3D positions remain bilinear-interpolated for smooth distribution.
            sub_u_nn = np.clip(np.round(sub_u).astype(np.int32), 0, W_img - 1)
            sub_v_nn = np.clip(np.round(sub_v).astype(np.int32), 0, H_img - 1)
            super_rgb[sl] = rgb_fullres[sub_v_nn, sub_u_nn]

            if return_geometry_metadata:
                super_source_uv[sl, 0] = sub_u
                super_source_uv[sl, 1] = sub_v

        super_valid = _finite_bounded_xyz_mask(
            super_wx, super_wy, super_wz)
        n_super_valid = int(super_valid.sum())

        # Per-pixel spacing stats for the masked region
        mask_p99 = float(np.percentile(spacing_local[adaptive_mask], 99))

        if n_super_valid > 0:
            total_before = points_world.shape[0]
            expected_total = total_before + n_super_valid
            if expected_total <= MAX_GAUSSIANS:
                points_world = np.concatenate([
                    points_world,
                    np.stack([super_wx[super_valid], super_wy[super_valid], super_wz[super_valid]], axis=-1),
                ], axis=0)
                colors = np.concatenate([colors, super_rgb[super_valid] / 255.0], axis=0)
                x_sign_flat = np.concatenate([x_sign_flat, super_xs[super_valid]], axis=0)
                panel_owner_flat = np.concatenate([panel_owner_flat, super_po[super_valid]], axis=0)
                if return_geometry_metadata:
                    source_uv_flat = np.concatenate(
                        [source_uv_flat, super_source_uv[super_valid]], axis=0)
                    tangent_u_flat = np.concatenate(
                        [tangent_u_flat, super_tangent_u[super_valid]], axis=0)
                    tangent_v_flat = np.concatenate(
                        [tangent_v_flat, super_tangent_v[super_valid]], axis=0)
                    hard_edge_flat = np.concatenate(
                        [hard_edge_flat, super_hard_edge[super_valid]], axis=0)
                if return_edge_aware_mask:
                    edge_aware_flat = np.concatenate(
                        [edge_aware_flat, super_edge[super_valid]], axis=0)
                if return_representation_categories:
                    representation_categories_flat = np.concatenate(
                        [
                            representation_categories_flat,
                            super_category[super_valid],
                        ],
                        axis=0,
                    )
                is_super_sampled = np.concatenate([
                    is_super_sampled,
                    np.ones(n_super_valid, dtype=bool)])
                print(f"    [Adaptive Super-sampling] {n_masked}/{H_img*W_img} pixels "
                      f"({n_masked/H_img/W_img*100:.1f}%) masked "
                      f"(local 3D spacing > {spacing_threshold*1000:.2f}mm, "
                      f"mask p99={mask_p99*1000:.2f}mm)")
                print(f"    {n_masked} pixels x {n_sub_offsets} sub-pixel -> {n_super} candidates, "
                      f"{n_super_valid} valid")
                print(f"    Total after super-sampling: {points_world.shape[0]} "
                      f"({total_before} base + {n_super_valid} super-sampled)")
            else:
                print(f"    [Adaptive Super-sampling] SKIPPED: would exceed MAX_GAUSSIANS "
                      f"({expected_total} > {MAX_GAUSSIANS})")
        else:
            print(f"    [Adaptive Super-sampling] No valid super-sampled points (all {n_super} filtered)")
    else:
        print(f"    [Adaptive Super-sampling] No sparse pixels detected "
              f"(spacing_threshold={spacing_threshold*1000:.2f}mm)")

    n_total = points_world.shape[0]
    n_left = int(np.sum(x_sign_flat < 0))
    n_right = int(np.sum(x_sign_flat > 0))

    print(f"    Total points: {n_total}")
    print(f"    X<0 (left half-space):  {n_left}  ({n_left/max(n_total,1)*100:.1f}%)")
    print(f"    X>0 (right half-space): {n_right}  ({n_right/max(n_total,1)*100:.1f}%)")
    print(f"    X range: [{points_world[:,0].min():.3f}, {points_world[:,0].max():.3f}] m")
    print(f"    Y range: [{points_world[:,1].min():.3f}, {points_world[:,1].max():.3f}] m")
    print(f"    Z range: [{points_world[:,2].min():.3f}, {points_world[:,2].max():.3f}] m")

    # ── FG/BG split: filter FG + add BG wall grid ──
    if fg_mask is not None:
        fg_flat = fg_mask[::step, ::step].ravel()
        fg_valid_base = fg_flat[valid]
        # Extend fg_valid for super-sampled points (always foreground)
        n_super = points_world.shape[0] - len(fg_valid_base)
        if n_super > 0:
            fg_valid = np.concatenate([fg_valid_base, np.ones(n_super, dtype=bool)])
        else:
            fg_valid = fg_valid_base
        n_fg_before = points_world.shape[0]
        points_world = points_world[fg_valid]
        colors = colors[fg_valid]
        x_sign_flat = x_sign_flat[fg_valid]
        panel_owner_flat = panel_owner_flat[fg_valid]
        if return_geometry_metadata:
            source_uv_flat = source_uv_flat[fg_valid]
            tangent_u_flat = tangent_u_flat[fg_valid]
            tangent_v_flat = tangent_v_flat[fg_valid]
            hard_edge_flat = hard_edge_flat[fg_valid]
        if return_edge_aware_mask:
            edge_aware_flat = edge_aware_flat[fg_valid]
        if return_representation_categories:
            representation_categories_flat = representation_categories_flat[fg_valid]
        is_super_sampled = is_super_sampled[fg_valid]
        n_fg = points_world.shape[0]
        print(f"    FG/BG split: {n_fg}/{n_fg_before} points are foreground "
              f"({100.*n_fg/max(n_fg_before,1):.1f}%)")

        # Build BG wall grid with anamorphic colors
        bg_pts, bg_col, bg_xs, bg_po = _build_bg_wall_grid(
            rgb_fullres, fg_mask, content_w, content_h,
            PANEL_PIXEL_W, PANEL_PIXEL_H, PANEL_PHYS_W_M, PANEL_PHYS_H_M,
            SWEET_SPOT, bg_grid_step)

        # Merge FG + BG
        points_world = np.concatenate([points_world, bg_pts], axis=0)
        colors = np.concatenate([colors, bg_col], axis=0)
        x_sign_flat = np.concatenate([x_sign_flat, bg_xs], axis=0)
        panel_owner_flat = np.concatenate([panel_owner_flat, bg_po], axis=0)
        if return_geometry_metadata:
            source_uv_flat = np.concatenate(
                [source_uv_flat, np.full((len(bg_pts), 2), -1.0, dtype=np.float32)], axis=0)
            tangent_u_flat = np.concatenate(
                [tangent_u_flat, np.zeros((len(bg_pts), 3), dtype=np.float32)], axis=0)
            tangent_v_flat = np.concatenate(
                [tangent_v_flat, np.zeros((len(bg_pts), 3), dtype=np.float32)], axis=0)
            hard_edge_flat = np.concatenate(
                [hard_edge_flat, np.zeros(len(bg_pts), dtype=bool)], axis=0)
        if return_edge_aware_mask:
            edge_aware_flat = np.concatenate(
                [edge_aware_flat, np.zeros(len(bg_pts), dtype=bool)])
        if return_representation_categories:
            representation_categories_flat = np.concatenate(
                [
                    representation_categories_flat,
                    np.full(
                        len(bg_pts),
                        REP_CATEGORY_BACKGROUND_CONTROL,
                        dtype=np.uint8,
                    ),
                ])
        is_super_sampled = np.concatenate(
            [is_super_sampled, np.zeros(len(bg_pts), dtype=bool)])

        n_total = points_world.shape[0]
        n_fg = n_fg  # already set above
        n_bg = len(bg_pts)
        print(f"    Merged: {n_total} total ({n_fg} FG + {n_bg} BG wall)")
    else:
        n_bg = 0

    # Build BG point mask (needed by create_gaussian_model for scale)
    n_total = points_world.shape[0]
    if fg_mask is not None:
        is_bg_point = np.zeros(n_total, dtype=bool)
        is_bg_point[-n_bg:] = True
    else:
        is_bg_point = np.zeros(n_total, dtype=bool)

    support_pts, support_col, support_xs, support_po, support_uv, support_meta = (
        _build_seam_boundary_support_points(
            rgb_fullres,
            world_x_full,
            world_y_full,
            world_z_full,
            panel_owner,
            fg_mask=fg_mask,
        )
    )
    SEAM_BOUNDARY_SUPPORT_LAST_META = dict(support_meta)
    n_support = int(support_pts.shape[0])
    if n_support > 0:
        points_world = np.concatenate([points_world, support_pts], axis=0)
        colors = np.concatenate([colors, support_col], axis=0)
        x_sign_flat = np.concatenate([x_sign_flat, support_xs], axis=0)
        panel_owner_flat = np.concatenate([panel_owner_flat, support_po], axis=0)
        if return_geometry_metadata:
            source_uv_flat = np.concatenate(
                [source_uv_flat, support_uv.astype(np.float32)], axis=0)
            tangent_u_flat = np.concatenate(
                [tangent_u_flat, np.zeros((n_support, 3), dtype=np.float32)], axis=0)
            tangent_v_flat = np.concatenate(
                [tangent_v_flat, np.zeros((n_support, 3), dtype=np.float32)], axis=0)
            hard_edge_flat = np.concatenate(
                [hard_edge_flat, np.zeros(n_support, dtype=bool)], axis=0)
        if return_edge_aware_mask:
            edge_aware_flat = np.concatenate(
                [edge_aware_flat, np.zeros(n_support, dtype=bool)])
        if return_representation_categories:
            representation_categories_flat = np.concatenate(
                [
                    representation_categories_flat,
                    np.full(n_support, REP_CATEGORY_UNKNOWN, dtype=np.uint8),
                ])
        is_super_sampled = np.concatenate(
            [is_super_sampled, np.zeros(n_support, dtype=bool)])
        is_bg_point = np.concatenate(
            [is_bg_point, np.zeros(n_support, dtype=bool)])
        print(f"    Seam boundary support: {n_support} owner-preserving pts "
              f"(band={support_meta.get('band_px', 0)} source px, "
              f"left={support_meta.get('left_support_count', 0)}, "
              f"right={support_meta.get('right_support_count', 0)})")
    else:
        gate = support_meta.get("manual_support_gate", {})
        if gate:
            print("    Seam boundary support: none "
                  f"(mode={support_meta.get('mode', 'unknown')}, "
                  f"reason={gate.get('reason', 'unknown')}, "
                  f"max_gap={gate.get('max_gap_px', 0.0):.2f}px)")

    if return_edge_aware_mask:
        edge_aware_meta["point_edge_count"] = int(edge_aware_flat.sum())
        edge_aware_meta["point_count"] = int(edge_aware_flat.size)
        print(f"    Edge-aware point mask: {edge_aware_meta['point_edge_count']}/"
              f"{edge_aware_meta['point_count']} pts "
              f"({100.0 * edge_aware_meta['point_edge_count'] / max(edge_aware_meta['point_count'], 1):.2f}%)")
    if return_representation_categories:
        representation_category_meta["point_category_counts"] = (
            _category_count_dict(representation_categories_flat)
        )
        print("    Representation point categories: "
              + ", ".join(
                  f"{name}={count}"
                  for name, count in representation_category_meta["point_category_counts"].items()
                  if name != "TOTAL" and count > 0
              ))
    result = [
        points_world,
        colors,
        x_sign_flat,
        is_super_sampled,
        panel_owner_flat,
        is_bg_point,
    ]
    if return_edge_aware_mask:
        result.append(edge_aware_flat)
    if return_representation_categories:
        result.append(representation_categories_flat)
    if return_geometry_metadata:
        geometry_metadata = {
            "schema": "task16_geometry_metadata_v1",
            "source_uv": source_uv_flat.astype(np.float32),
            "tangent_u": tangent_u_flat.astype(np.float32),
            "tangent_v": tangent_v_flat.astype(np.float32),
            "hard_edge": hard_edge_flat.astype(bool),
            "panel_owner": panel_owner_flat.astype(np.uint8),
            "point_count": int(points_world.shape[0]),
            "hard_edge_count": int(hard_edge_flat.sum()),
            "seam_boundary_support": support_meta,
        }
        result.append(geometry_metadata)
    return tuple(result)

# ============================================================================
# 3      -> Gaussian Model
# ============================================================================


def _as_device_tensor(value, *, dtype=None):
    """Return a detached tensor on ``DEVICE`` without redundant copies.

    Inputs are either a NumPy array or Torch tensor and an optional target
    dtype.  Exact-device/dtype tensors retain their storage, while NumPy inputs
    preserve the historical host-to-device conversion.  Invalid array-like
    values and device transfer failures propagate to the caller.
    """
    if torch.is_tensor(value):
        tensor = value.detach()
        if dtype is None:
            return tensor.to(device=DEVICE)
        return tensor.to(device=DEVICE, dtype=dtype)
    tensor = torch.from_numpy(np.asarray(value))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(DEVICE)


_GAUSSIAN_REUSE_PARAMETER_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)
_GAUSSIAN_REUSE_ROUTING_NAMES = (
    "_panel_owner",
    "_is_super_sampled",
)
_GAUSSIAN_REUSE_OPTIONAL_ATTRIBUTES = (
    "_source_uv",
    "_representation_categories",
    "_representation_contract_support_mask",
    "_surface_patch_level_patch_mask",
    "_surface_patch_level_patch_ids",
)


class GaussianFrameReuseState:
    """Own persistent GPU storage for consecutive video-frame Gaussians.

    Inputs are supplied later through :meth:`bind`; construction has no device
    side effect. The state adopts the first compatible frame's six Gaussian
    tensors and optional routing tensors as its capacity-sized backing storage.
    Later frames with the same SH degree, representation, device, dtype, tensor
    tails, and no more than the stored point capacity are copied into the active
    prefix in place. A larger or incompatible frame is adopted as a new backing
    allocation. The returned ``GaussianModel`` remains the same object whenever
    SH degree is unchanged, while active tensor views expose exactly the current
    point count. This changes storage lifecycle only; it does not cache or alter
    geometry, scale, opacity, or rendering calculations.

    Non-CUDA tensors, missing parameter tensors, inconsistent leading dimensions,
    and non-positive point counts fail explicitly before storage is rebound.
    Counters distinguish initial allocation, compatible reuse, and reallocation.
    """

    def __init__(self):
        self.model = None
        self.capacity = 0
        self.active_count = 0
        self.sh_degree = None
        self.gaussian_representation = None
        self.allocation_count = 0
        self.reuse_count = 0
        self.reallocation_count = 0
        self.last_action = "uninitialized"
        self._storage = {}

    def prepare_model(self, sh_degree, gaussian_representation):
        """Return a reusable model and clear optional per-frame attributes."""
        requested_degree = int(sh_degree)
        if self.model is None or self.model.max_sh_degree != requested_degree:
            self.model = GaussianModel(sh_degree=requested_degree)
        for name in _GAUSSIAN_REUSE_OPTIONAL_ATTRIBUTES:
            if hasattr(self.model, name):
                delattr(self.model, name)
        return self.model

    @staticmethod
    def _validate(values, routing):
        """Validate a complete CUDA tensor set and return its point count."""
        missing = [
            name for name in _GAUSSIAN_REUSE_PARAMETER_NAMES
            if name not in values
        ]
        if missing:
            raise ValueError(f"missing Gaussian reuse tensors: {missing}")
        xyz = values["_xyz"]
        if not torch.is_tensor(xyz):
            raise TypeError("_xyz must be a torch.Tensor")
        if xyz.ndim < 1:
            raise ValueError("_xyz must have a leading point dimension")
        point_count = int(xyz.shape[0])
        if point_count <= 0:
            raise ValueError("Gaussian reuse requires at least one point")
        for name, tensor in {**values, **routing}.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if tensor.ndim < 1 or int(tensor.shape[0]) != point_count:
                raise ValueError(
                    f"{name} leading dimension must equal {point_count}, "
                    f"got {tuple(tensor.shape)}")
        return point_count

    def _is_compatible(
            self, values, routing, point_count, sh_degree,
            gaussian_representation):
        """Return whether supplied tensors fit the current backing store."""
        if not self._storage or point_count > self.capacity:
            return False
        if self.sh_degree != int(sh_degree):
            return False
        if self.gaussian_representation != gaussian_representation:
            return False
        expected_names = set(values) | set(routing)
        if set(self._storage) != expected_names:
            return False
        for name, source in {**values, **routing}.items():
            backing = self._storage[name]
            if backing.device != source.device or backing.dtype != source.dtype:
                return False
            if tuple(backing.shape[1:]) != tuple(source.shape[1:]):
                return False
        return True

    @staticmethod
    def _parameter_from_tensor(tensor):
        """Wrap tensor storage as an inference parameter without a device copy."""
        return nn.Parameter(tensor.detach().contiguous().requires_grad_(False))

    def _adopt(self, values, routing, point_count):
        """Adopt current CUDA tensors as capacity-sized backing storage."""
        storage = {
            name: self._parameter_from_tensor(tensor)
            for name, tensor in values.items()
        }
        storage.update({
            name: tensor.detach().contiguous()
            for name, tensor in routing.items()
        })
        self._storage = storage
        self.capacity = point_count

    def _copy_in_place(self, values, routing, point_count):
        """Copy one frame into current backing prefixes without reallocating."""
        with torch.no_grad():
            for name, source in {**values, **routing}.items():
                destination = self._storage[name][:point_count]
                same_view = (
                    destination.data_ptr() == source.data_ptr()
                    and destination.storage_offset() == source.storage_offset()
                    and tuple(destination.shape) == tuple(source.shape)
                )
                if not same_view:
                    destination.copy_(source)

    def _bind_active_views(self, model, point_count, routing_names):
        """Expose exact active-prefix views on the reusable model."""
        for name in _GAUSSIAN_REUSE_PARAMETER_NAMES:
            backing = self._storage[name]
            active = backing if point_count == self.capacity else backing[:point_count]
            current = getattr(model, name, None)
            if not (
                    torch.is_tensor(current)
                    and tuple(current.shape) == tuple(active.shape)
                    and current.data_ptr() == active.data_ptr()):
                setattr(model, name, self._parameter_from_tensor(active))
        for name in _GAUSSIAN_REUSE_ROUTING_NAMES:
            if name not in routing_names:
                if hasattr(model, name):
                    delattr(model, name)
                continue
            backing = self._storage[name]
            active = backing if point_count == self.capacity else backing[:point_count]
            setattr(model, name, active)

    def bind(
            self, model, values, routing, sh_degree,
            gaussian_representation):
        """Bind computed tensors to persistent storage and return the model.

        ``values`` contains the six canonical parameters and ``routing`` may
        contain panel/super-sample masks. Compatible inputs copy into stored
        prefixes; initial or incompatible inputs replace the backing set after
        validation. The active view never exposes stale capacity-tail elements.
        """
        point_count = self._validate(values, routing)
        compatible = self._is_compatible(
            values,
            routing,
            point_count,
            sh_degree,
            gaussian_representation,
        )
        had_storage = bool(self._storage)
        if compatible:
            self._copy_in_place(values, routing, point_count)
            self.reuse_count += 1
            self.last_action = "reuse"
        else:
            self._adopt(values, routing, point_count)
            self.allocation_count += 1
            if had_storage:
                self.reallocation_count += 1
                self.last_action = "reallocate"
            else:
                self.last_action = "allocate"
        self.model = model
        self.active_count = point_count
        self.sh_degree = int(sh_degree)
        self.gaussian_representation = gaussian_representation
        self._bind_active_views(model, point_count, set(routing))
        return model

    def diagnostics(self):
        """Return JSON-safe storage lifecycle counters for frame metadata."""
        return {
            "enabled": True,
            "storage_reuse_only": True,
            "gaussian_calculations_reused": False,
            "last_action": self.last_action,
            "capacity": int(self.capacity),
            "active_count": int(self.active_count),
            "allocation_count": int(self.allocation_count),
            "reuse_count": int(self.reuse_count),
            "reallocation_count": int(self.reallocation_count),
            "sh_degree": self.sh_degree,
            "gaussian_representation": self.gaussian_representation,
        }


def create_gaussian_model(points_world, colors, sh_degree=0,
                          img_width=None, content_w=None,
                          overlap_factor=2.0,
                          is_super_sampled=None,
                          panel_owner=None,
                          target_opacity=0.99,
                          depth_clamp_min=2.0,
                          depth_clamp_range=8.0,
                          far_shrink_gamma=2.0,
                          far_shrink_max=0.0,  # disabled
                          shrink_floor_ratio=0.5,
                          max_scale_mm=3.2,
                          fade_k=12.0, fade_threshold=0.10,
                          is_bg_point=None, bg_grid_step=4,
                          edge_aware_mask=None,
                          edge_aware_repair_enabled=None,
                          representation_categories=None,
                          representation_contract_prototype_enabled=None,
                          surface_patch_level_prototype_enabled=None,
                          geometry_metadata=None,
                          gaussian_representation="isotropic",
                          reuse_state=None):
    """
       -> 3DGS Gaussian    

    Scale = max(pixel_pitch x overlap_factor, local_NN_dist x 0.7) 
    pixel_pitch           , KNN                  
    is_super_sampled: bool array, True for super-sampled points -> tighter overlap.
    Opacity = inv_sigmoid(target_opacity)            
    """
    if img_width is None or content_w is None:
        raise ValueError("img_width and content_w are required")
    (
        points_world,
        colors,
        is_super_sampled,
        panel_owner,
        is_bg_point,
        representation_categories,
        representation_support_mask,
        representation_contract_stats,
    ) = _append_representation_contract_support(
        points_world,
        colors,
        is_super_sampled,
        panel_owner,
        is_bg_point,
        representation_categories,
        img_width,
        content_w,
        prototype_enabled=representation_contract_prototype_enabled,
    )
    (
        points_world,
        colors,
        is_super_sampled,
        panel_owner,
        is_bg_point,
        representation_categories,
        surface_patch_mask,
        surface_patch_ids,
        surface_patch_stats,
    ) = _append_surface_patch_level_prototype(
        points_world,
        colors,
        is_super_sampled,
        panel_owner,
        is_bg_point,
        representation_categories,
        img_width,
        content_w,
        exclude_mask=representation_support_mask,
        prototype_enabled=surface_patch_level_prototype_enabled,
    )
    if edge_aware_mask is not None:
        if torch.is_tensor(edge_aware_mask):
            edge_tensor = edge_aware_mask.detach().reshape(-1).to(dtype=torch.bool)
            if edge_tensor.numel() < points_world.shape[0]:
                edge_tensor = torch.cat([
                    edge_tensor,
                    torch.zeros(
                        points_world.shape[0] - edge_tensor.numel(),
                        device=edge_tensor.device,
                        dtype=torch.bool,
                    ),
                ])
            edge_aware_mask = edge_tensor
        else:
            edge_np = np.asarray(edge_aware_mask, dtype=bool).ravel()
            if edge_np.size < points_world.shape[0]:
                edge_aware_mask = np.concatenate([
                    edge_np,
                    np.zeros(points_world.shape[0] - edge_np.size, dtype=bool),
                ])

    N = points_world.shape[0]
    print(f"\n[3/7] Creating GaussianModel from {N} points...")
    if reuse_state is None:
        print(">>> REBUILDING GAUSSIANS FROM SCRATCH <<<")
    else:
        print(">>> UPDATING PREALLOCATED GAUSSIAN STORAGE <<<")
    if gaussian_representation not in ("isotropic", "surfel"):
        raise ValueError(
            "gaussian_representation must be 'isotropic' or 'surfel', "
            f"got {gaussian_representation!r}")

    if reuse_state is not None and not isinstance(
            reuse_state, GaussianFrameReuseState):
        raise TypeError("reuse_state must be GaussianFrameReuseState or None")
    model = (
        GaussianModel(sh_degree=sh_degree)
        if reuse_state is None
        else reuse_state.prepare_model(sh_degree, gaussian_representation)
    )
    xyz = _as_device_tensor(points_world, dtype=torch.float32)
    source_uv_for_model = None
    if geometry_metadata is not None and "source_uv" in geometry_metadata:
        source_uv_np = np.asarray(geometry_metadata["source_uv"], dtype=np.float32)
        if source_uv_np.ndim == 2 and source_uv_np.shape[1] == 2:
            if source_uv_np.shape[0] < N:
                pad = np.full((N - source_uv_np.shape[0], 2), -1.0, dtype=np.float32)
                source_uv_np = np.concatenate([source_uv_np, pad], axis=0)
            elif source_uv_np.shape[0] > N:
                source_uv_np = source_uv_np[:N]
            source_uv_for_model = source_uv_np.astype(np.float32, copy=False)

    # RGB -> SH DC
    if torch.is_tensor(colors):
        colors_t = _as_device_tensor(
            colors, dtype=torch.float32).clamp(min=0.0, max=1.0)
    else:
        colors_t = torch.from_numpy(np.clip(colors, 0, 1)).float().to(DEVICE)
    sh_dc = RGB2SH(colors_t)
    features_dc = sh_dc.unsqueeze(1)

    n_rest = (model.max_sh_degree + 1) ** 2 - 1
    features_rest = torch.zeros((N, n_rest, 3), device=DEVICE, dtype=torch.float32)

    #    Uniform Scale (box projection = uniform pixel spacing, no KNN needed)
    pixel_pitch_m = content_w / float(img_width)
    base_scale = pixel_pitch_m * overlap_factor
    per_point_scale = torch.full((N,), base_scale, device=DEVICE)
    print(f"    Uniform scale: {base_scale*1000:.3f}mm (pixel_pitch={pixel_pitch_m*1000:.3f}mm x {overlap_factor})")

    repair_enabled = (
        bool(EDGE_AWARE_GAUSSIAN_REPAIR_ENABLED)
        if edge_aware_repair_enabled is None
        else bool(edge_aware_repair_enabled)
    )

    # BG wall grid points: scale by grid step to fill gaps between sparse wall points
    bg_mask = None
    if is_bg_point is not None:
        bg_mask = _as_device_tensor(is_bg_point, dtype=torch.bool).reshape(-1)
        n_bg = int(bg_mask.sum().item())
        if n_bg > 0:
            # Detect grid step from point density (each panel gets ~same # of BG pts)
            bg_scale = base_scale * float(bg_grid_step)
            per_point_scale[bg_mask] = bg_scale
            print(f"    BG wall scale: {bg_scale*1000:.3f}mm "
                  f"(base={base_scale*1000:.3f}mm x step={bg_grid_step}, {n_bg} pts)")

    # Depth-aware clamp:                   
    # d_phys = (x+y)/sqrt2,   (+0.08m)->    (10xbase),   (-0.12m)->    (2xbase)
    inv_sqrt2 = 1.0 / math.sqrt(2)
    d_phys = (xyz[:, 0] + xyz[:, 1]) * inv_sqrt2
    d_min, d_max = d_phys.min(), d_phys.max()
    d_norm = ((d_phys - d_min) / max(d_max - d_min, 1e-8)).clamp(0.0, 1.0)
    max_per_pt = per_point_scale * (depth_clamp_min + depth_clamp_range * d_norm)
    # per_point_scale already set

    # Far-field 3D scale shrinking:           3D        KNN    Scale
    # ->    2D clamp   85px,            "  "    
    #   :                  ,          ,
    #              ,        ,        
    # decay = 1.0 - ((1-d_norm)^gamma) * max_shrink
    #      d_norm=1: decay   1.0 (    )
    #      d_norm=0: decay   1-max_shrink (  0.15 ->    15%)
    # far_shrink_gamma: >1      ,         1.0
    # far_shrink_max:          (0.85 =    15%)
    farness = 1.0 - d_norm       # 1=  , 0=
    farness_pow = farness.pow(far_shrink_gamma)
    decay_factor = 1.0 - farness_pow * far_shrink_max
    shrink_floor = pixel_pitch_m * shrink_floor_ratio
    per_point_scale_faded = torch.maximum(per_point_scale * decay_factor,
                                           torch.full((N,), shrink_floor, device=DEVICE))
    # BG wall points keep their grid-step scale, not depth-based shrink
    if is_bg_point is not None and bg_mask.any():
        per_point_scale = torch.where(bg_mask, per_point_scale, per_point_scale_faded)
    else:
        per_point_scale = per_point_scale_faded
    #          clamp    (       )
    # per_point_scale already set

    # Adaptive cap: fixed hard cap (3.2mm) OR pitch-adaptive floor,
    # whichever is larger.  For low-resolution inputs (large pixel_pitch),
    # the fixed cap creates coverage gaps (overlap_ratio < 0.5).
    # Adaptive floor ensures >= 80% overlap per axis at all resolutions.
    fixed_max_m = max_scale_mm / 1000.0
    adaptive_max_m = pixel_pitch_m * GAUSSIAN_OVERLAP_FACTOR * 0.8
    max_scale_m = max(fixed_max_m, adaptive_max_m)
    per_point_scale = per_point_scale.clamp(max=max_scale_m)

    # Fold-zone scale boost: panel_owner==2 points need larger scales because
    # smoothstep blend between two projection planes produces tighter NN spacing.
    # Apply a region-specific multiplier without affecting panel-only Gaussians.
    po_t = None
    if panel_owner is not None:
        po_t = _as_device_tensor(panel_owner, dtype=torch.uint8).reshape(-1)
        fold_mask = (po_t == 2)
        n_fold = int(fold_mask.sum().item())
        if n_fold > 0:
            pre_mean = per_point_scale[fold_mask].mean().item()
            per_point_scale = torch.where(
                fold_mask,
                per_point_scale * FOLD_SCALE_BOOST,
                per_point_scale,
            )
            post_mean = per_point_scale[fold_mask].mean().item()
            print(f"    Fold-zone scale boost: {n_fold} pts, "
                  f"{pre_mean*1000:.3f}mm -> {post_mean*1000:.3f}mm "
                  f"(x{FOLD_SCALE_BOOST})")
    else:
        n_fold = 0

    # Scale adaptation for super-sampled points.
    # Super-sampled points are packed ~SUPER_SAMPLE_SIZEx denser per dimension,
    # so their footprint must be capped to the sub-pixel pitch. Otherwise the
    # adaptive sampler creates 16x more primitives with the same footprint as
    # base pixels, saturating alpha before source-neighbor primitives contribute.
    super_sampled_mask = None
    super_scale_cap = None
    if is_super_sampled is not None:
        super_sampled_mask = _as_device_tensor(
            is_super_sampled, dtype=torch.bool).reshape(-1)
        n_super = int(super_sampled_mask.sum().item())
        if n_super > 0:
            effective_pixel_pitch = pixel_pitch_m / float(SUPER_SAMPLE_SIZE)
            super_scale_cap = effective_pixel_pitch * float(overlap_factor)
            pre_mean = per_point_scale[super_sampled_mask].mean().item()
            per_point_scale[super_sampled_mask] = torch.minimum(
                per_point_scale[super_sampled_mask],
                torch.full((n_super,), super_scale_cap, device=DEVICE))
            post_mean = per_point_scale[super_sampled_mask].mean().item()
            print(f"    Adaptive super-sample scale: {n_super} pts, "
                  f"{pre_mean*1000:.3f}mm -> {post_mean*1000:.3f}mm "
                  f"(effective pixel_pitch={effective_pixel_pitch*1000:.3f}mm, "
                  f"cap={super_scale_cap*1000:.3f}mm)")

    seam_support_scale_active = bool(
        SEAM_BOUNDARY_SUPPORT_LAST_META.get("enabled", False)
        and SEAM_BOUNDARY_SUPPORT_LAST_META.get("support_count", 0) > 0
    )
    if panel_owner is not None and seam_support_scale_active:
        source_half_pitch_m = content_w / float(max((img_width // 2) - 1, 1))
        support_band_m = source_half_pitch_m * float(SEAM_BOUNDARY_SUPPORT_BAND_PX)
        support_floor_m = (
            PANEL_PHYS_W_M / float(max(PANEL_PIXEL_W - 1, 1))
            * float(SEAM_BOUNDARY_SUPPORT_SCALE_FLOOR_PANEL_PITCH)
        )
        surface_eps = float(SEAM_BOUNDARY_SUPPORT_SURFACE_EPS_M)
        left_support = (
            (po_t == 0)
            & (xyz[:, 0] >= -support_band_m - surface_eps)
            & (xyz[:, 0] <= surface_eps)
            & (xyz[:, 1].abs() <= surface_eps)
        )
        right_support = (
            (po_t == 1)
            & (xyz[:, 1] >= -support_band_m - surface_eps)
            & (xyz[:, 1] <= surface_eps)
            & (xyz[:, 0].abs() <= surface_eps)
        )
        support_mask = left_support | right_support
        if bg_mask is not None:
            support_mask = support_mask & ~bg_mask
        n_support = int(support_mask.sum().item())
        if n_support > 0:
            pre_mean = per_point_scale[support_mask].mean().item()
            per_point_scale[support_mask] = torch.maximum(
                per_point_scale[support_mask],
                torch.full((n_support,), support_floor_m, device=DEVICE))
            post_mean = per_point_scale[support_mask].mean().item()
            print(f"    Seam boundary support scale floor: {n_support} pts, "
                  f"{pre_mean*1000:.3f}mm -> {post_mean*1000:.3f}mm "
                  f"(floor={support_floor_m*1000:.3f}mm, "
                  f"band={SEAM_BOUNDARY_SUPPORT_BAND_PX}px)")

    if representation_contract_stats.get("enabled") and representation_support_mask is not None:
        support_np = np.asarray(representation_support_mask, dtype=bool).ravel()
        if support_np.shape[0] == N:
            support_t = torch.from_numpy(support_np).to(DEVICE)
            if bg_mask is not None:
                support_t = support_t & ~bg_mask
            n_contract_support = int(support_t.sum().item())
            representation_contract_stats["scale_support_point_count"] = n_contract_support
            if n_contract_support > 0:
                local_cap_m = min(
                    max_scale_m * float(REP_CONTRACT_PROTO_SUPPORT_CAP_MULTIPLIER),
                    base_scale * float(REP_CONTRACT_PROTO_SUPPORT_CAP_MULTIPLIER),
                )
                support_floor_m = min(
                    local_cap_m,
                    base_scale * float(REP_CONTRACT_PROTO_SUPPORT_SCALE_MULTIPLIER),
                )
                pre_mean = per_point_scale[support_t].mean().item()
                per_point_scale[support_t] = torch.maximum(
                    per_point_scale[support_t],
                    torch.full((n_contract_support,), support_floor_m, device=DEVICE),
                )
                post_mean = per_point_scale[support_t].mean().item()
                representation_contract_stats["support_scale_floor_mm"] = round(
                    support_floor_m * 1000.0, 6)
                representation_contract_stats["support_local_cap_mm"] = round(
                    local_cap_m * 1000.0, 6)
                representation_contract_stats["pre_mean_support_scale_mm"] = round(
                    pre_mean * 1000.0, 6)
                representation_contract_stats["post_mean_support_scale_mm"] = round(
                    post_mean * 1000.0, 6)
                print(f"    Representation-contract support scale floor: "
                      f"{n_contract_support} pts, "
                      f"{pre_mean*1000:.3f}mm -> {post_mean*1000:.3f}mm")
        else:
            representation_contract_stats["enabled"] = False
            representation_contract_stats["reason"] = (
                f"support mask length {support_np.shape[0]} does not match point count {N}"
            )

    if surface_patch_stats.get("enabled") and surface_patch_mask is not None:
        patch_np = np.asarray(surface_patch_mask, dtype=bool).ravel()
        if patch_np.shape[0] == N:
            patch_t = torch.from_numpy(patch_np).to(DEVICE)
            if bg_mask is not None:
                patch_t = patch_t & ~bg_mask
            n_patch = int(patch_t.sum().item())
            surface_patch_stats["scale_patch_point_count"] = n_patch
            if n_patch > 0:
                tile_scale_m = (
                    pixel_pitch_m
                    * float(SURFACE_PATCH_PROTO_TILE_SOURCE_PX)
                    * float(SURFACE_PATCH_PROTO_SCALE_MULTIPLIER)
                )
                local_cap_m = (
                    pixel_pitch_m
                    * float(SURFACE_PATCH_PROTO_TILE_SOURCE_PX)
                    * float(SURFACE_PATCH_PROTO_SCALE_CAP_MULTIPLIER)
                )
                patch_floor_m = min(local_cap_m, tile_scale_m)
                pre_mean = per_point_scale[patch_t].mean().item()
                per_point_scale[patch_t] = torch.maximum(
                    per_point_scale[patch_t],
                    torch.full((n_patch,), patch_floor_m, device=DEVICE),
                )
                post_mean = per_point_scale[patch_t].mean().item()
                surface_patch_stats["patch_scale_floor_mm"] = round(
                    patch_floor_m * 1000.0, 6)
                surface_patch_stats["patch_local_cap_mm"] = round(
                    local_cap_m * 1000.0, 6)
                surface_patch_stats["pre_mean_patch_scale_mm"] = round(
                    pre_mean * 1000.0, 6)
                surface_patch_stats["post_mean_patch_scale_mm"] = round(
                    post_mean * 1000.0, 6)
                print(f"    Surface/patch prototype scale floor: "
                      f"{n_patch} pts, "
                      f"{pre_mean*1000:.3f}mm -> {post_mean*1000:.3f}mm")
        else:
            surface_patch_stats["enabled"] = False
            surface_patch_stats["reason"] = (
                f"patch mask length {patch_np.shape[0]} does not match point count {N}"
            )

    edge_repair_stats = {
        "enabled": bool(repair_enabled),
        "family": "edge_aware_existing_gaussian_footprint_density",
        "mask_available": edge_aware_mask is not None,
        "adds_points": False,
        "changes_projection": False,
        "image_domain_repair": False,
        "scale_multiplier": float(EDGE_AWARE_SCALE_MULTIPLIER),
        "scale_cap_multiplier": float(EDGE_AWARE_SCALE_CAP_MULTIPLIER),
        "point_count": int(N),
        "edge_point_count": 0,
    }
    if repair_enabled and edge_aware_mask is not None:
        edge_mask_t = _as_device_tensor(
            edge_aware_mask, dtype=torch.bool).reshape(-1)
        if edge_mask_t.shape[0] == N:
            if bg_mask is not None:
                edge_mask_t = edge_mask_t & ~bg_mask
            n_edge = int(edge_mask_t.sum().item())
            edge_repair_stats["edge_point_count"] = n_edge
            edge_repair_stats["edge_point_fraction"] = round(n_edge / max(N, 1), 6)
            local_cap_m = min(
                max_scale_m * float(EDGE_AWARE_SCALE_CAP_MULTIPLIER),
                base_scale * float(EDGE_AWARE_SCALE_CAP_MULTIPLIER),
            )
            edge_repair_stats["local_scale_cap_mm"] = round(local_cap_m * 1000.0, 6)
            if n_edge > 0:
                pre_mean = per_point_scale[edge_mask_t].mean().item()
                boosted = torch.minimum(
                    per_point_scale[edge_mask_t] * float(EDGE_AWARE_SCALE_MULTIPLIER),
                    torch.full((n_edge,), local_cap_m, device=DEVICE),
                )
                per_point_scale[edge_mask_t] = torch.maximum(
                    per_point_scale[edge_mask_t],
                    boosted,
                )
                post_mean = per_point_scale[edge_mask_t].mean().item()
                edge_repair_stats["pre_mean_scale_mm"] = round(pre_mean * 1000.0, 6)
                edge_repair_stats["post_mean_scale_mm"] = round(post_mean * 1000.0, 6)
                print(f"    Edge-aware Gaussian repair: {n_edge} pts, "
                      f"{pre_mean*1000:.3f}mm -> {post_mean*1000:.3f}mm "
                      f"(x{EDGE_AWARE_SCALE_MULTIPLIER}, "
                      f"cap={local_cap_m*1000:.3f}mm)")
            else:
                print("    Edge-aware Gaussian repair: no edge points selected")
        else:
            edge_repair_stats["enabled"] = False
            edge_repair_stats["reason"] = (
                f"mask length {edge_mask_t.shape[0]} does not match point count {N}"
            )
            print(f"    Edge-aware Gaussian repair skipped: {edge_repair_stats['reason']}")
    elif repair_enabled:
        edge_repair_stats["enabled"] = False
        edge_repair_stats["reason"] = "edge-aware mask not provided"
        print("    Edge-aware Gaussian repair skipped: mask not provided")

    scale_mean = per_point_scale.mean().item()
    scale_median = per_point_scale.median().item()
    print(f"    Final scale: mean={scale_mean*1000:.3f}mm  median={scale_median*1000:.3f}mm  "
          f"range=[{per_point_scale.min().item()*1000:.3f}, {per_point_scale.max().item()*1000:.3f}]mm")

    if panel_owner is not None and n_fold > 0:
        panel_mask = ~fold_mask
        if panel_mask.any():
            panel_mean = per_point_scale[panel_mask].mean().item() * 1000
            fold_mean = per_point_scale[fold_mask].mean().item() * 1000
            print(f"      Panel vs Fold scale: panel_mean={panel_mean:.3f}mm  "
                  f"fold_mean={fold_mean:.3f}mm  ratio={fold_mean/max(panel_mean,1e-8):.2f}x")

    task16_surfel_stats = {
        "enabled": False,
        "schema": "task16_surfel_gaussian_representation_v1",
        "representation": gaussian_representation,
    }
    rotation = torch.zeros((N, 4), device=DEVICE, dtype=torch.float32)
    rotation[:, 0] = 1.0
    if gaussian_representation == "surfel":
        surfel_scales, surfel_rotation, task16_surfel_stats = (
            _task16_build_surfel_scale_rotation(
                geometry_metadata,
                per_point_scale,
                base_scale,
                max_scale_m,
                overlap_factor,
                is_super_sampled=is_super_sampled,
                super_scale_cap=super_scale_cap,
            )
        )
        if surfel_scales is not None and surfel_rotation is not None:
            scaling = torch.log(torch.clamp(surfel_scales, min=1e-8))
            rotation = surfel_rotation
            print("    Task16C surfel Gaussian representation: "
                  f"anisotropy_mean={task16_surfel_stats.get('anisotropy_ratio_mean')} "
                  f"hard_edges={task16_surfel_stats.get('hard_edge_count')}")
        else:
            task16_surfel_stats["fallback_used"] = True
            scaling = torch.log(per_point_scale.unsqueeze(1).expand(-1, 3))
            print("    Task16C surfel fallback to isotropic: "
                  f"{task16_surfel_stats.get('reason', 'unknown')}")
    else:
        scaling = torch.log(per_point_scale.unsqueeze(1).expand(-1, 3))

    #              
    inv_sig = math.log(target_opacity / (1.0 - target_opacity))
    opacity = torch.full((N, 1), inv_sig, device=DEVICE, dtype=torch.float32)

    #    Far-field opacity fading: sigmoid smooth fade for oil-painting blur
    # BG wall points skip fading — they are always fully opaque.
    fade_factor = torch.sigmoid(fade_k * (d_norm - fade_threshold))
    if is_bg_point is not None and bg_mask.any():
        fade_factor[bg_mask] = 1.0  # BG walls stay fully opaque
    alpha_linear = torch.sigmoid(opacity)
    alpha_faded = alpha_linear * fade_factor.unsqueeze(1)
    alpha_faded = alpha_faded.clamp(min=0.001)
    opacity = torch.logit(alpha_faded)
    n_faded = int((d_norm < fade_threshold).sum().item())
    print(f"    Opacity: inv_sigmoid({target_opacity}) = {inv_sig:.3f}")
    print(f"    Opacity Fade: sigmoid(k={fade_k}, th={fade_threshold}), "
          f"{n_faded}/{N} far gaussians ({100.*n_faded/N:.1f}%), "
          f"fade range=[{fade_factor.min().item():.4f}, {fade_factor.max().item():.4f}]")

    # Store super-sampled mask for diagnostic probes
    if super_sampled_mask is not None and reuse_state is None:
        model._is_super_sampled = super_sampled_mask
    if po_t is not None and reuse_state is None:
        model._panel_owner = po_t
    if source_uv_for_model is not None:
        model._source_uv = torch.from_numpy(source_uv_for_model).to(DEVICE)
    model._edge_aware_repair_stats = edge_repair_stats
    model._representation_contract_prototype_stats = representation_contract_stats
    model._surface_patch_level_prototype_stats = surface_patch_stats
    model._task16_surfel_stats = task16_surfel_stats
    model._gaussian_representation = gaussian_representation
    if representation_categories is not None:
        model._representation_categories = torch.from_numpy(
            np.asarray(representation_categories, dtype=np.uint8)).to(DEVICE)
    if representation_support_mask is not None:
        model._representation_contract_support_mask = torch.from_numpy(
            np.asarray(representation_support_mask, dtype=bool)).to(DEVICE)
    if surface_patch_mask is not None:
        model._surface_patch_level_patch_mask = torch.from_numpy(
            np.asarray(surface_patch_mask, dtype=bool)).to(DEVICE)
    if surface_patch_ids is not None:
        model._surface_patch_level_patch_ids = torch.from_numpy(
            np.asarray(surface_patch_ids, dtype=np.int32)).to(DEVICE)

    parameter_values = {
        "_xyz": xyz,
        "_features_dc": features_dc,
        "_features_rest": features_rest,
        "_scaling": scaling,
        "_rotation": rotation,
        "_opacity": opacity,
    }
    routing_values = {}
    if po_t is not None:
        routing_values["_panel_owner"] = po_t
    if super_sampled_mask is not None:
        routing_values["_is_super_sampled"] = super_sampled_mask
    if reuse_state is None:
        for name, value in parameter_values.items():
            setattr(model, name, nn.Parameter(value.requires_grad_(False)))
    else:
        model = reuse_state.bind(
            model,
            parameter_values,
            routing_values,
            sh_degree,
            gaussian_representation,
        )
        print(
            "    Gaussian storage: "
            f"action={reuse_state.last_action}, "
            f"active={reuse_state.active_count}, "
            f"capacity={reuse_state.capacity}, "
            f"reuse_count={reuse_state.reuse_count}, "
            f"reallocations={reuse_state.reallocation_count}"
        )
    model.active_sh_degree = sh_degree

    print(f"    Created {N} gaussians")
    return model


# ============================================================================
# 5 — Build dual-frustum camera
# ============================================================================

def build_dual_camera(mirror_left=False, mirror_right=False):
    """Build DualFrustumCamera in Z-up coordinates."""
    print(f"\n[5/7] Building dual-frustum camera...")
    print(f"    Panel: {PANEL_PHYS_W_M:.3f}x{PANEL_PHYS_H_M:.3f} m  ({PANEL_PIXEL_W}x{PANEL_PIXEL_H} px)")
    print(f"    Aspect (H/W): {PANEL_PHYS_H_M/PANEL_PHYS_W_M:.3f} = {PANEL_PIXEL_H}/{PANEL_PIXEL_W} = {PANEL_PIXEL_H/PANEL_PIXEL_W:.3f}")
    print(f"    Sweet Spot: {tuple(SWEET_SPOT)}")
    if mirror_left or mirror_right:
        print(f"    Mirror: left={mirror_left} right={mirror_right}")

    left_corners, right_corners = build_l_screen_corners(PANEL_PHYS_W_M, PANEL_PHYS_H_M)

    camera = DualFrustumCamera(
        viewer_pos=SWEET_SPOT,
        screen_left_corners=left_corners,
        screen_right_corners=right_corners,
        znear=0.01,
        zfar=100.0,
        overlap_world_m=OVERLAP_WORLD_M,
        image_width=PANEL_PIXEL_W,
        image_height=PANEL_PIXEL_H,
        mirror_left=mirror_left,
        mirror_right=mirror_right,
        device=str(DEVICE),
    )
    return camera


# ============================================================================
# 6     
# ============================================================================

def render_dual(camera, model, bg_color):
    """Render from two asymmetric frustums (left/right panels)."""
    print(f"\n[6/7] Rendering dual frustums ({PANEL_PIXEL_W}x{PANEL_PIXEL_H} each)...")
    pipe = GroupParams()
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.debug = False
    pipe.antialiasing = False

    xyz = model.get_xyz
    n_current = xyz.shape[0]
    panel_owner = getattr(model, '_panel_owner', None)
    if panel_owner is not None and len(panel_owner) == n_current:
        # Split fold zone by world X coordinate instead of rendering all
        # fold Gaussians in BOTH frustums.  This eliminates the topological
        # density increase (extra checkerboard cells) at the fold seam.
        fold_mask = (panel_owner == 2)
        n_fold = int(fold_mask.sum().item())
        if n_fold > 0:
            fold_left  = fold_mask & (xyz[:, 0] < -FADE_THRESHOLD_M)
            fold_right = fold_mask & (xyz[:, 0] > +FADE_THRESHOLD_M)
            fold_fade  = fold_mask & (xyz[:, 0].abs() <= FADE_THRESHOLD_M)
            n_fl = int(fold_left.sum().item())
            n_fr = int(fold_right.sum().item())
            n_ff = int(fold_fade.sum().item())
            model._panel_left_mask  = (panel_owner == 0) | fold_left | fold_fade
            model._panel_right_mask = (panel_owner == 1) | fold_right | fold_fade
            print(f"    Fold zone split: {n_fl} left, {n_fr} right, "
                  f"{n_ff} fade ({FADE_THRESHOLD_M*1000:.0f}mm transition)")
        else:
            model._panel_left_mask = (panel_owner == 0)
            model._panel_right_mask = (panel_owner == 1)
    else:
        model._panel_left_mask = xyz[:, 0] < 0.001
        model._panel_right_mask = xyz[:, 0] > -0.001

    t0 = time.time()
    out = render_dual_frustum(camera, model, pipe, bg_color)
    elapsed = time.time() - t0

    for side in ["left", "right"]:
        key = f"render_{side}"
        img = out[key]
        print(f"    {side.capitalize()} render: {img.shape}  "
              f"range=[{img.min():.3f}, {img.max():.3f}]  mean={img.mean():.3f}")

        # Diagnostic: radii distribution
        r = out[f"radii_{side}"]
        r_pos = r[r > 0]
        print(f"    {side.capitalize()} radii: {r_pos.numel()}/{r.numel()} visible "
              f"(median={r_pos.float().median():.0f}px, "
              f"max={r.max().item()}px, "
              f"pct_gt5={100.*(r>5).sum().item()/r.numel():.1f}%)")

        # What fraction of pixels are pure background?
        T_key = f"render_{side}"
        bg_mask = (img.mean(dim=0) < 0.051)  # R+G+B < 0.153 -> mean < 0.051
        n_bg = bg_mask.sum().item()
        print(f"    {side.capitalize()} bg pixels: {n_bg}/{img.shape[1]*img.shape[2]} "
              f"({100.*n_bg/(img.shape[1]*img.shape[2]):.1f}%)")

    print(f"    Render time: {elapsed:.2f}s")
    return out


# ============================================================================
# 7         (   resize      )
# ============================================================================

def soft_blend_seam(img_left, img_right, overlap_pixels):
    """
    Blend two images along the vertical fold seam using a linear alpha ramp.

    The fold is at the right edge of img_left and left edge of img_right.
    Overlap zone spans `overlap_pixels` columns from each side, cross-faded:
      alpha_L(col) = (P-1 - col) / (P-1)   1->0
      alpha_R(col) = col / (P-1)            0->1

    Returns (H, 2*W - overlap_pixels, C) blended image.
    """
    H, W, C = img_left.shape
    P = int(overlap_pixels)
    assert 0 < P < W, f"overlap_pixels ({P}) must be in (0, {W})"

    ramp_down = np.linspace(1.0, 0.0, P, dtype=np.float64)[None, :, None]
    ramp_up   = np.linspace(0.0, 1.0, P, dtype=np.float64)[None, :, None]

    zone_L = img_left[:, -P:, :].astype(np.float64)
    zone_R = img_right[:, :P, :].astype(np.float64)
    blended = zone_L * ramp_down + zone_R * ramp_up

    stitched = np.concatenate([
        img_left[:, :-P, :].astype(np.float64),
        blended,
        img_right[:, P:, :].astype(np.float64),
    ], axis=1)

    return stitched.astype(img_left.dtype)


STITCHED_OUTPUT_Z_AXIS_FLIP_ENABLED = True


def _apply_stitched_output_z_axis_flip(img):
    """Legacy adapter from patent raw rows to a conventional upright preview.

    Canonical raw panel and raw-combined files must not call this helper.
    It remains available for older diagnostic harnesses that import it.
    """
    if not STITCHED_OUTPUT_Z_AXIS_FLIP_ENABLED:
        return adapt_panel_row0_edge(img, SCREEN_BOTTOM)
    return adapt_panel_row0_edge(img, SCREEN_TOP)


def _cuda_status():
    """Return JSON-safe CUDA status for parameter provenance."""
    cuda_available = bool(torch.cuda.is_available())
    status = {
        "device": str(DEVICE),
        "torch_cuda_available": cuda_available,
    }
    if cuda_available:
        try:
            status["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            status["device_name"] = None
    return status


def _write_parameter_provenance(
        out_dir, input_rgb, overlap_px, blend_px,
        raw_combined_shape, blended_preview_shape, run_metadata=None,
        save_profile="diagnostic"):
    """Record the active display profile and output semantics for this run."""
    run_metadata = run_metadata or {}
    raw_w = int(raw_combined_shape[1])
    raw_h = int(raw_combined_shape[0])
    preview_w = int(blended_preview_shape[1])
    preview_h = int(blended_preview_shape[0])
    input_shape = None
    if input_rgb is not None:
        input_shape = [int(input_rgb.shape[0]), int(input_rgb.shape[1]), int(input_rgb.shape[2])]

    output_files = {
        "canonical_raw_panel_left": "render_left.png",
        "canonical_raw_panel_right": "render_right.png",
        "raw_combined_diagnostic": "render_raw_combined.png",
        "raw_combined_resolution": [raw_w, raw_h],
        "render_left_resolution": [int(PANEL_PIXEL_W), int(PANEL_PIXEL_H)],
        "render_right_resolution": [int(PANEL_PIXEL_W), int(PANEL_PIXEL_H)],
        "save_profile": save_profile,
    }
    row0_edges = {
        "render_left.png": PATENT_RAW_ROW0_EDGE,
        "render_right.png": PATENT_RAW_ROW0_EDGE,
        "render_raw_combined.png": PATENT_RAW_ROW0_EDGE,
    }
    if save_profile == "diagnostic":
        output_files.update({
            "raw_combined_compatibility_alias": "render_stitched.png",
            "diagnostic_blended_preview": "render_blended_preview.png",
            "raw_left_seam_crop": "seam_crop_left_edge.png",
            "raw_right_seam_crop": "seam_crop_right_edge.png",
            "raw_combined_seam_crop": "seam_crop_raw_combined.png",
            "blended_preview_resolution": [preview_w, preview_h],
        })
        row0_edges.update({
            "render_stitched.png": PATENT_RAW_ROW0_EDGE,
            "render_blended_preview.png": SCREEN_TOP,
            "seam_crop_left_edge.png": PATENT_RAW_ROW0_EDGE,
            "seam_crop_right_edge.png": PATENT_RAW_ROW0_EDGE,
            "seam_crop_raw_combined.png": PATENT_RAW_ROW0_EDGE,
        })
    output_files["panel_row_orientation"] = {
        "contract": "patent_0724_raw_raster_row0_v1",
        "raw_raster_row0_edge": PATENT_RAW_ROW0_EDGE,
        "per_file_row0_edge": row0_edges,
        "upright_preview_adapter_enabled": bool(
            STITCHED_OUTPUT_Z_AXIS_FLIP_ENABLED),
        "controller_rule": (
            "Apply one vertical flip only when the display controller expects "
            "file row 0 at the physical screen top."
        ),
    }
    output_files["semantics"] = {
        "canonical_display_outputs": ["render_left.png", "render_right.png"],
        "raw_combined_is_diagnostic": True,
        "blended_preview_is_canonical": False,
    }

    provenance = {
        "schema": "task07b_seam_exclusive_v1",
        "orientation_contract": "patent_0724_raw_raster_row0_v1",
        "run_id": run_metadata.get("run_id"),
        "mode": run_metadata.get("mode", "direct_process_frame"),
        "frame_idx": run_metadata.get("frame_idx"),
        "input_path": run_metadata.get("input_path"),
        "input_shape": input_shape,
        "output_dir": os.path.abspath(out_dir),
        "accepted_display_profile": {
            "panel_pixel_w": int(PANEL_PIXEL_W),
            "panel_pixel_h": int(PANEL_PIXEL_H),
            "raw_combined_pixel_w": int(2 * PANEL_PIXEL_W),
            "raw_combined_pixel_h": int(PANEL_PIXEL_H),
            "panel_phys_w_m": float(PANEL_PHYS_W_M),
            "panel_phys_h_m": float(PANEL_PHYS_H_M),
            "sweet_spot": [float(v) for v in SWEET_SPOT],
            "sweet_spot_profile": run_metadata.get(
                "sweet_spot_profile", SWEET_SPOT_PROFILE),
            "sweet_spot_horizontal_distance_m": float(
                math.sqrt(float(SWEET_SPOT[0]) ** 2 + float(SWEET_SPOT[1]) ** 2)),
            "fold_angle_deg": 90,
        },
        "seam": {
            "overlap_world_m": float(OVERLAP_WORLD_M),
            "geometric_overlap_px": int(overlap_px),
            "blend_pixels_target": int(BLEND_PIXELS_TARGET),
            "blended_preview_px": int(blend_px),
            "strategy": "seam_exclusive_physical_panels",
            "projection_blend_px": 0,
            "canonical_overlap_blending": False,
            "boundary_support": {
                "enabled": bool(SEAM_BOUNDARY_SUPPORT_ENABLED),
                "mode": str(SEAM_BOUNDARY_SUPPORT_MODE),
                "active_mode": _active_seam_boundary_support_mode(),
                "family": "manual_owner_preserving_boundary_support_ablation",
                "band_source_px": int(SEAM_BOUNDARY_SUPPORT_BAND_PX),
                "scale_floor_panel_pitch_multiplier": float(
                    SEAM_BOUNDARY_SUPPORT_SCALE_FLOOR_PANEL_PITCH),
                "cross_panel_duplication": False,
                "image_domain_repair": False,
                "default_production_path": False,
            },
        },
        "gaussian_representation": {
            "representation": run_metadata.get("gaussian_representation", "isotropic"),
            "overlap_factor": float(GAUSSIAN_OVERLAP_FACTOR),
            "target_opacity": float(GAUSSIAN_OPACITY_SIGMOID),
            "frame_reuse": run_metadata.get(
                "gaussian_frame_reuse", {"enabled": False}),
            "task16_surfel": run_metadata.get(
                "task16_surfel_gaussian",
                {
                    "enabled": False,
                    "representation": run_metadata.get(
                        "gaussian_representation", "isotropic"),
                    "reason": "not reported by caller",
                },
            ),
            "edge_aware_footprint_density_repair": run_metadata.get(
                "edge_aware_gaussian_repair",
                {
                    "enabled": bool(EDGE_AWARE_GAUSSIAN_REPAIR_ENABLED),
                    "mask_available": False,
                    "reason": "not reported by caller",
                },
            ),
            "representation_contract_prototype": run_metadata.get(
                "representation_contract_prototype",
                {
                    "enabled": bool(REPRESENTATION_CONTRACT_PROTOTYPE_ENABLED),
                    "support_point_count": 0,
                    "reason": "not reported by caller",
                },
            ),
            "surface_patch_level_prototype": run_metadata.get(
                "surface_patch_level_prototype",
                {
                    "enabled": bool(SURFACE_PATCH_LEVEL_PROTOTYPE_ENABLED),
                    "patch_point_count": 0,
                    "reason": "not reported by caller",
                },
            ),
        },
        "depth_structure": {
            "process_res": int(run_metadata.get("depth_process_res", 756)),
            "task16": run_metadata.get(
                "task16_depth_structure",
                {
                    "schema": "task16_depth_structure_result_v1",
                    "mode": "off",
                    "enabled": False,
                    "reason": "not reported by caller",
                },
            ),
        },
        "pointcloud_backend": run_metadata.get(
            "pointcloud_backend",
            {
                "requested": "unknown",
                "resolved": "cpu",
                "reason": "not reported by caller",
                "cuda": None,
            },
        ),
        "outputs": output_files,
        "cuda": _cuda_status(),
        "argv": run_metadata.get("argv"),
        "saved_at_unix": time.time(),
    }

    provenance_path = os.path.join(out_dir, "parameter_provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"    Parameter provenance: {provenance_path}")


def save_results(out, input_rgb=None, overlap_px=0, out_dir=None,
                 run_metadata=None, save_profile="diagnostic"):
    """Save canonical panel outputs and optional diagnostic artifacts.

    ``minimal`` is the production video profile: two physical-panel PNGs, one
    combined review frame, and parameter provenance. ``diagnostic`` preserves
    the historical full artifact set used by single-image and audit harnesses.
    """
    if save_profile not in {"minimal", "diagnostic"}:
        raise ValueError(
            "save_profile must be 'minimal' or 'diagnostic', "
            f"got {save_profile!r}")
    if out_dir is None:
        out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[7/7] Saving results to {out_dir}...")

    def to_uint8(tensor):
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    left_img = to_uint8(out["render_left"])
    right_img = to_uint8(out["render_right"])

    #     RGB PNG (cv2    BGR)
    cv2.imwrite(os.path.join(out_dir, "render_left.png"),
                cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "render_right.png"),
                cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))

    raw_combined = np.hstack([left_img, right_img])
    raw_combined_output = adapt_panel_row0_edge(raw_combined, PATENT_RAW_ROW0_EDGE)
    cv2.imwrite(os.path.join(out_dir, "render_raw_combined.png"),
                cv2.cvtColor(raw_combined_output, cv2.COLOR_RGB2BGR))

    print(f"    Raw combined diagnostic: {raw_combined_output.shape[1]}x{raw_combined_output.shape[0]} px")

    blend_px = 0
    blended_preview_output = raw_combined_output
    if save_profile == "diagnostic":
        # Compatibility alias retained only for audit consumers.
        cv2.imwrite(os.path.join(out_dir, "render_stitched.png"),
                    cv2.cvtColor(raw_combined_output, cv2.COLOR_RGB2BGR))

        seam_crop_w = min(96, PANEL_PIXEL_W)
        seam_left = left_img[:, -seam_crop_w:, :]
        seam_right = right_img[:, :seam_crop_w, :]
        seam_combined = np.hstack([seam_left, seam_right])
        seam_combined_output = adapt_panel_row0_edge(
            seam_combined, PATENT_RAW_ROW0_EDGE)
        cv2.imwrite(os.path.join(out_dir, "seam_crop_left_edge.png"),
                    cv2.cvtColor(seam_left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "seam_crop_right_edge.png"),
                    cv2.cvtColor(seam_right, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "seam_crop_raw_combined.png"),
                    cv2.cvtColor(seam_combined_output, cv2.COLOR_RGB2BGR))

        # The blended image is for fold inspection, not the display canvas.
        if overlap_px > 0:
            blend_px = min(overlap_px, BLEND_PIXELS_TARGET, PANEL_PIXEL_W - 1)
            print(f"    Diagnostic alpha preview: {blend_px} px overlap zone (geometric={overlap_px} px)")
            blended_preview = soft_blend_seam(left_img, right_img, blend_px)
        else:
            blended_preview = raw_combined
        blended_preview_output = _apply_stitched_output_z_axis_flip(blended_preview)
        cv2.imwrite(os.path.join(out_dir, "render_blended_preview.png"),
                    cv2.cvtColor(blended_preview_output, cv2.COLOR_RGB2BGR))
        print(f"    Blended preview (diagnostic): {blended_preview_output.shape[1]}x{blended_preview_output.shape[0]} px")

        # Depth diagnostics use percentile-based normalization.
        for side in ["left", "right"]:
            key = f"depth_{side}"
            if key in out and out[key] is not None:
                try:
                    d = out[key].detach().cpu().float().numpy()
                    if d.ndim == 1:
                        d = d.reshape(PANEL_PIXEL_H, PANEL_PIXEL_W)
                    elif d.ndim == 3:
                        d = d.squeeze(0)
                    if d.ndim != 2:
                        continue
                    valid = np.isfinite(d) & (np.abs(d) < 1e6)
                    if valid.sum() > 100:
                        dmin = np.percentile(d[valid], 1)
                        dmax = np.percentile(d[valid], 99)
                        dn = np.zeros_like(d)
                        if dmax - dmin > 1e-8:
                            dn[valid] = np.clip(
                                (d[valid] - dmin) / (dmax - dmin), 0, 1)
                        cv2.imwrite(os.path.join(out_dir, f"depth_{side}.png"),
                                    (dn * 255).astype(np.uint8))
                        print(f"    depth_{side}: saved (1%={dmin:.3f}, 99%={dmax:.3f})")
                    else:
                        print(f"    depth_{side}: too few valid pixels ({valid.sum()})")
                except Exception as e:
                    print(f"    Warning: depth_{side}: {e}")

        if input_rgb is not None:
            preview = cv2.resize(
                cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR),
                (min(input_rgb.shape[1], 800), min(input_rgb.shape[0], 600)))
            cv2.imwrite(os.path.join(out_dir, "input_preview.png"), preview)
    else:
        print("    Save profile: minimal (diagnostic PNGs skipped)")

    _write_parameter_provenance(
        out_dir, input_rgb, overlap_px, blend_px,
        raw_combined_output.shape, blended_preview_output.shape,
        run_metadata=run_metadata, save_profile=save_profile)

    print(f"\n  Output files:")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        print(f"    {f} ({os.path.getsize(fpath)/1024:.1f} KB)")


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Per-Frame Pipeline Core (extracted from main for video-loop reuse)
# ============================================================================


def _resolve_pointcloud_backend(
        requested,
        *,
        save_profile,
        fg_mask_present,
        depth_structure_mode,
        gaussian_representation,
        edge_aware_gaussian_repair,
        representation_contract_prototype,
        surface_patch_level_prototype,
        needs_representation_categories,
        needs_geometry_metadata,
        ):
    """Resolve the pointcloud backend with explicit, fail-closed semantics.

    Inputs are the requested ``auto/cpu/cuda`` mode and all features that alter
    the canonical pointcloud/model contract.  Output is ``(backend, reason)``.
    Explicit CPU always preserves the NumPy path; auto falls back with named
    blockers; strict CUDA raises when CUDA is unavailable or any unsupported
    feature is active.
    """
    mode = str(requested).strip().lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "pointcloud_backend must be 'auto', 'cpu', or 'cuda', "
            f"got {requested!r}")
    if mode == "cpu":
        return "cpu", "explicit CPU pointcloud backend requested"

    blockers = []
    if DEVICE.type != "cuda" or not torch.cuda.is_available():
        blockers.append("CUDA device unavailable")
    if abs(float(SWEET_SPOT[0])) < 1.0e-8 or abs(float(SWEET_SPOT[1])) < 1.0e-8:
        blockers.append("sweet_spot eye-normal coordinate is zero")
    if save_profile != "minimal":
        blockers.append(f"save_profile={save_profile!r}")
    if bool(fg_mask_present):
        blockers.append("fg_mask_present=True")
    if depth_structure_mode != "off":
        blockers.append(f"depth_structure_mode={depth_structure_mode!r}")
    if gaussian_representation != "isotropic":
        blockers.append(
            f"gaussian_representation={gaussian_representation!r}")
    if bool(edge_aware_gaussian_repair):
        blockers.append("edge_aware_gaussian_repair=True")
    if bool(representation_contract_prototype):
        blockers.append("representation_contract_prototype=True")
    if bool(surface_patch_level_prototype):
        blockers.append("surface_patch_level_prototype=True")
    if bool(needs_representation_categories):
        blockers.append("needs_representation_categories=True")
    if bool(needs_geometry_metadata):
        blockers.append("needs_geometry_metadata=True")
    if (
            _active_seam_boundary_support_mode() != "off"
            or bool(SEAM_BOUNDARY_SUPPORT_ENABLED)):
        blockers.append("seam_boundary_support is active")

    if blockers:
        details = ", ".join(blockers)
        if mode == "cuda":
            raise RuntimeError(
                "strict CUDA pointcloud backend is unsupported: " + details)
        return "cpu", "auto fallback to CPU: " + details
    return "cuda", "canonical CUDA pointcloud path is supported"


def process_frame(rgb_orig, model_dav3, camera, out_dir=None,
                  frame_idx=None, depth_map=None, depth_anchor=None,
                  jbf_ksize=9, jbf_sigma_s=5.0, jbf_sigma_r=0.03,
                  model_sam3=None, run_metadata=None,
                  edge_aware_gaussian_repair=None,
                  representation_contract_prototype=None,
                  surface_patch_level_prototype=None,
                  depth_structure_mode="off",
                  gaussian_representation="isotropic",
                  depth_process_res=756,
                  save_structure_debug=False,
                  save_profile="diagnostic",
                  pointcloud_backend="auto",
                  gaussian_reuse_state=None):
    """
    Run the full 3DGS pipeline (steps 1-7) on a single RGB frame.

    Depth Anything V3 model and dual-frustum camera are pre-loaded and reused
    across calls   this function is designed to be called in a video-consumption
    loop.

    Parameters
    ----------
    rgb_orig : np.ndarray
        (H, W, 3) uint8 RGB image.
    model_dav3 : DepthAnything3
        Pre-loaded Depth Anything V3 model (eval mode).
    camera : DualFrustumCamera
        Pre-built dual-frustum camera.
    out_dir : str | None
        Output directory for this frame's results.
    frame_idx : int | None
        Frame index for logging / sub-directory naming.
    depth_map : np.ndarray or None
        (H, W) float32 pre-computed depth map.  When provided, DAV3 inference
        is skipped   the caller is responsible for depth estimation and
        temporal smoothing (e.g. via VideoDepthStreamer EMA).
    depth_anchor : DepthAnchorStream or None
        When provided, EMA-smoothed percentile bounds are used for depth
        normalization instead of per-frame min-max.  Essential for video to
        prevent "breathing jelly effect" from temporal scale drift.
    pointcloud_backend : {"auto", "cpu", "cuda"}
        ``auto`` uses CUDA only for the canonical minimal video contract;
        unsupported modes visibly retain CPU. ``cuda`` is strict and raises
        instead of silently falling back.
    gaussian_reuse_state : GaussianFrameReuseState or None
        Optional caller-owned video state. Compatible frames update persistent
        GPU backing tensors in place; ``None`` preserves independent construction.

    Returns
    -------
    dict
        Rendered output dict from render_dual().
    """
    global SEAM_BOUNDARY_SUPPORT_LAST_META
    H_orig, W_orig = rgb_orig.shape[:2]
    tag = f"[frame {frame_idx:04d}] " if frame_idx is not None else ""
    if depth_structure_mode not in ("off", "edge-preserve"):
        raise ValueError(
            "depth_structure_mode must be 'off' or 'edge-preserve', "
            f"got {depth_structure_mode!r}")
    if gaussian_representation not in ("isotropic", "surfel"):
        raise ValueError(
            "gaussian_representation must be 'isotropic' or 'surfel', "
            f"got {gaussian_representation!r}")
    if save_profile not in ("minimal", "diagnostic"):
        raise ValueError(
            "save_profile must be 'minimal' or 'diagnostic', "
            f"got {save_profile!r}")
    frame_t0 = time.perf_counter()
    stage_timings = {}

    #    [1/7]        
    t_stage = time.perf_counter()
    if depth_map is not None:
        # Pre-computed depth (e.g. EMA-smoothed DAV3 from VideoDepthStreamer).
        # Apply cv2 C++ JBF for spatial edge preservation at original resolution.
        depth_input = depth_map.astype(np.float32)
        if depth_input.shape != (H_orig, W_orig):
            depth_input = cv2.resize(depth_input, (W_orig, H_orig))
        depth_filtered = cv2_jbf(
            rgb_orig.astype(np.float32) / 255.0, depth_input,
            ksize=jbf_ksize, sigma_s=jbf_sigma_s, sigma_r=jbf_sigma_r)
    else:
        # Single-shot: DAV3 inference + cv2 C++ JBF + resize to original
        depth_filtered = predict_depth_with_jbf(
            rgb_orig, model_dav3, process_res=depth_process_res,
            jbf_d=jbf_ksize, jbf_sigma_color=jbf_sigma_r, jbf_sigma_space=jbf_sigma_s)
    stage_timings["depth_sec"] = time.perf_counter() - t_stage

    repair_enabled = (
        bool(EDGE_AWARE_GAUSSIAN_REPAIR_ENABLED)
        if edge_aware_gaussian_repair is None
        else bool(edge_aware_gaussian_repair)
    )
    prototype_enabled = (
        bool(REPRESENTATION_CONTRACT_PROTOTYPE_ENABLED)
        if representation_contract_prototype is None
        else bool(representation_contract_prototype)
    )
    surface_patch_enabled = (
        bool(SURFACE_PATCH_LEVEL_PROTOTYPE_ENABLED)
        if surface_patch_level_prototype is None
        else bool(surface_patch_level_prototype)
    )
    needs_representation_categories = prototype_enabled or surface_patch_enabled

    # The traced production artifact used the no-segmentation path.  Keep
    # ``model_sam3`` as a compatibility keyword, but foreground masking is no
    # longer part of production frame processing.
    fg_mask = None
    stage_timings["segmentation_sec"] = 0.0

    structure_result = None
    hard_edge_mask = None
    depth_for_geometry = depth_filtered
    t_stage = time.perf_counter()
    if depth_structure_mode == "edge-preserve":
        structure_result = refine_depth_structure(
            rgb_orig,
            depth_filtered,
            mode=depth_structure_mode,
            fg_mask=fg_mask,
            jbf_diameter=jbf_ksize,
            jbf_sigma_color=jbf_sigma_r,
            jbf_sigma_space=jbf_sigma_s,
        )
        depth_for_geometry = structure_result.depth_refined
        hard_edge_mask = structure_result.hard_edge_mask
        print("  Task16C depth structure refine: "
              f"hard_edges={structure_result.metadata['hard_edge_fraction']:.3f} "
              f"confidence={structure_result.metadata['confidence_mean']:.3f}")
    stage_timings["depth_structure_sec"] = time.perf_counter() - t_stage

    #    Temporal depth anchoring (avoids "breathing jelly" in video)
    t_stage = time.perf_counter()
    if depth_anchor is not None:
        depth_norm, depth_phys = depth_anchor.process(depth_for_geometry)
        if structure_result is not None:
            structure_result.depth_norm = depth_norm.astype(np.float32)
        print(f"  Anchor: ema_bounds=[{depth_anchor.ema_min:.3f}, {depth_anchor.ema_max:.3f}], "
              f"frame={depth_anchor.n_frames}")
    elif structure_result is not None:
        depth_norm = structure_result.depth_norm
    else:
        depth_norm = None   # fall back to per-frame min-max in stage 2
    stage_timings["depth_anchor_sec"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if out_dir and save_profile == "diagnostic":
        d_vis = (depth_filtered - depth_filtered.min()) / max(
            depth_filtered.max() - depth_filtered.min(), 1e-8)
        cv2.imwrite(os.path.join(out_dir, "depth_raw.png"),
                    (d_vis * 255).astype(np.uint8))
        if structure_result is not None and save_structure_debug:
            for name, image in depth_debug_images(structure_result).items():
                cv2.imwrite(os.path.join(out_dir, name), image)
    stage_timings["debug_depth_save_sec"] = time.perf_counter() - t_stage

    #    [2/7]    3D
    t_stage = time.perf_counter()
    content_h = PANEL_PHYS_H_M  # fill full panel height (was: CONTENT_W_M * H/W)
    needs_geometry_metadata = gaussian_representation == "surfel"
    resolved_pointcloud_backend, pointcloud_backend_reason = (
        _resolve_pointcloud_backend(
            pointcloud_backend,
            save_profile=save_profile,
            fg_mask_present=fg_mask is not None,
            depth_structure_mode=depth_structure_mode,
            gaussian_representation=gaussian_representation,
            edge_aware_gaussian_repair=repair_enabled,
            representation_contract_prototype=prototype_enabled,
            surface_patch_level_prototype=surface_patch_enabled,
            needs_representation_categories=needs_representation_categories,
            needs_geometry_metadata=needs_geometry_metadata,
        )
    )
    cuda_pointcloud_metadata = None
    if resolved_pointcloud_backend == "cuda":
        print(f"\n[2/7] CUDA pointcloud backend: {pointcloud_backend_reason}")
        pointcloud_result, cuda_pointcloud_metadata = (
            depth_to_unified_pointcloud_cuda(
                rgb_orig,
                depth_for_geometry,
                DEPTH_FWD_M,
                DEPTH_BACK_M,
                CONTENT_W_M,
                content_h,
                sweet_spot=tuple(float(value) for value in SWEET_SPOT),
                super_sample_size=int(SUPER_SAMPLE_SIZE),
                local_spacing_threshold=float(LOCAL_SPACING_THRESHOLD),
                max_gaussians=int(MAX_GAUSSIANS),
                depth_norm=depth_norm,
                device=DEVICE,
            )
        )
        torch.cuda.synchronize()
        SEAM_BOUNDARY_SUPPORT_LAST_META = {
            "enabled": False,
            "support_count": 0,
            "reason": "CUDA pointcloud canonical path requires seam support off",
        }
        print(
            "    CUDA pointcloud: "
            f"{cuda_pointcloud_metadata['output_point_count']} points, "
            f"{cuda_pointcloud_metadata['super_sampled_count']} super-sampled, "
            "anchoring="
            f"{cuda_pointcloud_metadata['depth_normalization']}"
        )
        if cuda_pointcloud_metadata["supersampling_skipped_for_capacity"]:
            print(
                "    Adaptive super-sampling skipped for capacity: "
                f"{cuda_pointcloud_metadata['candidate_point_count']} > "
                f"{cuda_pointcloud_metadata['max_gaussians']}"
            )
    else:
        print(f"    Pointcloud backend: CPU ({pointcloud_backend_reason})")
        pointcloud_result = depth_to_unified_pointcloud(
            rgb_orig, depth_for_geometry,
            DEPTH_FWD_M, DEPTH_BACK_M,
            CONTENT_W_M, content_h,
            depth_norm=depth_norm,
            fg_mask=fg_mask,
            hard_edge_mask=hard_edge_mask,
            return_edge_aware_mask=True,
            return_representation_categories=needs_representation_categories,
            return_geometry_metadata=needs_geometry_metadata)
    geometry_metadata = None
    if needs_geometry_metadata:
        *pointcloud_result, geometry_metadata = pointcloud_result
    if needs_representation_categories:
        (points_world, colors, x_sign, is_super_sampled, panel_owner,
         is_bg, edge_aware_mask, representation_categories) = pointcloud_result
    else:
        (points_world, colors, x_sign, is_super_sampled, panel_owner,
         is_bg, edge_aware_mask) = pointcloud_result
        representation_categories = None
    stage_timings["pointcloud_sec"] = time.perf_counter() - t_stage

    #    [3/7] Gaussian Model
    t_stage = time.perf_counter()
    model = create_gaussian_model(points_world, colors, sh_degree=SH_DEGREE,
                                  img_width=W_orig, content_w=CONTENT_W_M,
                                  overlap_factor=GAUSSIAN_OVERLAP_FACTOR,
                                  is_super_sampled=is_super_sampled,
                                  panel_owner=panel_owner,
                                  target_opacity=GAUSSIAN_OPACITY_SIGMOID,
                                  is_bg_point=is_bg,
                                  edge_aware_mask=edge_aware_mask,
                                  edge_aware_repair_enabled=repair_enabled,
                                  representation_categories=representation_categories,
                                  representation_contract_prototype_enabled=prototype_enabled,
                                  surface_patch_level_prototype_enabled=surface_patch_enabled,
                                  geometry_metadata=geometry_metadata,
                                  gaussian_representation=gaussian_representation,
                                  reuse_state=gaussian_reuse_state)
    stage_timings["representation_build_sec"] = time.perf_counter() - t_stage

    #    [6/7]      
    t_stage = time.perf_counter()
    out = render_dual(camera, model, BG_COLOR)
    stage_timings["render_sec"] = time.perf_counter() - t_stage

    #    [7/7]      
    save_metadata = dict(run_metadata or {})
    save_metadata["edge_aware_gaussian_repair"] = getattr(
        model, "_edge_aware_repair_stats", {
            "enabled": bool(repair_enabled),
            "mask_available": False,
            "reason": "model stats unavailable",
        })
    save_metadata["representation_contract_prototype"] = getattr(
        model, "_representation_contract_prototype_stats", {
            "enabled": bool(prototype_enabled),
            "support_point_count": 0,
            "reason": "model stats unavailable",
        })
    save_metadata["surface_patch_level_prototype"] = getattr(
        model, "_surface_patch_level_prototype_stats", {
            "enabled": bool(surface_patch_enabled),
            "patch_point_count": 0,
            "reason": "model stats unavailable",
        })
    save_metadata["task16_depth_structure"] = (
        structure_result.metadata if structure_result is not None else {
            "schema": "task16_depth_structure_result_v1",
            "mode": depth_structure_mode,
            "enabled": False,
        }
    )
    save_metadata["task16_surfel_gaussian"] = getattr(
        model, "_task16_surfel_stats", {
            "enabled": False,
            "representation": gaussian_representation,
            "reason": "model stats unavailable",
        })
    save_metadata["gaussian_representation"] = gaussian_representation
    save_metadata["gaussian_frame_reuse"] = (
        gaussian_reuse_state.diagnostics()
        if gaussian_reuse_state is not None
        else {"enabled": False}
    )
    save_metadata["depth_process_res"] = int(depth_process_res)
    save_metadata["pointcloud_backend"] = {
        "requested": str(pointcloud_backend),
        "resolved": resolved_pointcloud_backend,
        "reason": pointcloud_backend_reason,
        "cuda": cuda_pointcloud_metadata,
    }
    save_metadata["stage_timings_partial"] = {
        k: round(float(v), 6) for k, v in stage_timings.items()
    }
    t_stage = time.perf_counter()
    if out_dir:
        save_results(out, rgb_orig, overlap_px=camera.overlap_px, out_dir=out_dir,
                     run_metadata=save_metadata, save_profile=save_profile)
    else:
        save_results(out, rgb_orig, overlap_px=camera.overlap_px,
                     run_metadata=save_metadata, save_profile=save_profile)
    stage_timings["save_sec"] = time.perf_counter() - t_stage
    stage_timings["total_sec"] = time.perf_counter() - frame_t0
    if out_dir:
        timing_path = os.path.join(out_dir, "stage_timings.json")
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema": "l_splat_stage_timings_v1",
                "frame_idx": frame_idx,
                "timings_sec": {
                    k: round(float(v), 6) for k, v in stage_timings.items()
                },
            }, f, indent=2)

    return out


# ============================================================================
# Main   single-image mode (backward compatible) + video mode (FastVideoExtractor)
# ============================================================================

def main():
    global PANEL_PHYS_W_M, PANEL_PHYS_H_M, PANEL_PIXEL_W, PANEL_PIXEL_H
    global CONTENT_W_M, SWEET_SPOT, SWEET_SPOT_PROFILE, OVERLAP_WORLD_M, BLEND_PIXELS_TARGET
    global OUT_DIR
    global SEAM_BOUNDARY_SUPPORT_MODE, SEAM_BOUNDARY_SUPPORT_ENABLED, SEAM_BOUNDARY_SUPPORT_BAND_PX
    global SEAM_BOUNDARY_SUPPORT_SCALE_FLOOR_PANEL_PITCH, SEAM_BOUNDARY_SUPPORT_SURFACE_EPS_M

    ap = argparse.ArgumentParser(
        description="E2E Pipeline v3   Dual-Frustum L-Screen 3DGS")
    ap.add_argument("--video", type=str, default=None,
                    help="Path to input video (enables video mode).")
    ap.add_argument("--max-frames", type=int, default=5,
                    help="Max frames to process in video mode (default: 5, 0=all).")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Frame stride in video mode (1=all, 2=every-other).")
    ap.add_argument("--ema-alpha", type=float, default=0.4,
                    help="EMA smoothing factor for temporal depth (0<alpha<=1, "
                         "0.4=strong smoothing, 1.0=raw depth).")
    ap.add_argument("--flow-confidence",
                    choices=["off", "auto", "on"], default="auto",
                    help="Video temporal mode: off=legacy EMA, auto=use cached "
                         "RAFT or fall back, on=require RAFT (default: auto).")
    ap.add_argument("--flow-max-side", type=int, default=1024,
                    help="Maximum RAFT working-image side in pixels "
                         "(default: 1024, quality-first).")
    ap.add_argument("--flow-history-weight", type=float, default=0.5,
                    help="Maximum aligned previous-raw-depth weight in [0,1] "
                         "(default: 0.5; support is exactly two frames).")
    ap.add_argument("--image", type=str, default=None,
                    help="Path to a single input image (overrides default hypercar.jpg).")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR,
                    help="Output directory root.")
    ap.add_argument("--mirror-left", action="store_true", default=False,
                    help="Apply FLIP_X horizontal mirror to left frustum.")
    ap.add_argument("--mirror-right", action="store_true", default=False,
                    help="Apply FLIP_X horizontal mirror to right frustum.")
    ap.add_argument("--no-segment", action="store_true", default=False,
                    help="Deprecated compatibility flag; production segmentation "
                         "is always disabled.")
    ap.add_argument("--save-diagnostics", action="store_true", default=False,
                    help="In video mode, save the full historical diagnostic PNG "
                         "set instead of the minimal production artifacts.")
    ap.add_argument(
        "--pointcloud-backend",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Pointcloud backend: auto selects CUDA for the canonical minimal "
             "video path, cpu forces NumPy, cuda is strict (default: auto).",
    )
    ap.add_argument("--enable-edge-aware-gaussian-repair",
                    action="store_true", default=False,
                    help="Explicitly enable the Task10I-E edge-aware Gaussian footprint repair for ablation.")
    ap.add_argument("--disable-edge-aware-gaussian-repair",
                    action="store_true", default=False,
                    help="Legacy no-op guard; the Task10I-E repair is disabled by default.")
    ap.add_argument("--enable-representation-contract-prototype",
                    action="store_true", default=False,
                    help="Explicitly enable the Task10I-I aggressive representation-contract prototype.")
    ap.add_argument("--disable-representation-contract-prototype",
                    action="store_true", default=False,
                    help="Rollback guard; the Task10I-I prototype is disabled by default.")
    ap.add_argument("--enable-surface-patch-prototype",
                    action="store_true", default=False,
                    help="Explicitly enable the Task12 surface/patch-level representation prototype.")
    ap.add_argument("--disable-surface-patch-prototype",
                    action="store_true", default=False,
                    help="Rollback guard; the Task12 prototype is disabled by default.")
    ap.add_argument("--enable-seam-boundary-support",
                    action="store_true", default=False,
                    help="Enable owner-preserving seam-boundary support (same as mode=on).")
    ap.add_argument("--disable-seam-boundary-support",
                    action="store_true", default=False,
                    help="Disable owner-preserving seam-boundary support (same as mode=off).")
    ap.add_argument("--seam-boundary-support-mode",
                    choices=["on", "off"],
                    default=SEAM_BOUNDARY_SUPPORT_MODE,
                    help="Manual seam-boundary support ablation mode. Default off keeps the display-aware baseline.")
    ap.add_argument("--seam-boundary-support-band-px", type=int,
                    default=SEAM_BOUNDARY_SUPPORT_BAND_PX,
                    help="Source columns per side used by seam-boundary support when enabled.")
    ap.add_argument("--seam-boundary-support-scale-floor-panel-pitch", type=float,
                    default=SEAM_BOUNDARY_SUPPORT_SCALE_FLOOR_PANEL_PITCH,
                    help="Scale floor as a panel-pixel-pitch multiplier for seam-boundary support.")
    ap.add_argument("--seam-boundary-support-surface-eps-mm", type=float,
                    default=SEAM_BOUNDARY_SUPPORT_SURFACE_EPS_M * 1000.0,
                    help="Surface matching epsilon in millimetres for seam-boundary support scale floor.")
    # ── Display configuration CLI overrides (Task02) ──
    ap.add_argument("--depth-structure-mode",
                    choices=["off", "edge-preserve"], default="off",
                    help="Task16C DAV3 post-process mode. Default off preserves legacy behavior.")
    ap.add_argument("--gaussian-representation",
                    choices=["isotropic", "surfel"], default="isotropic",
                    help="Task16C Gaussian representation mode. Default isotropic preserves legacy behavior.")
    ap.add_argument("--depth-process-res", type=int, default=756,
                    help="DAV3 process_res upper bound (default: 756).")
    ap.add_argument("--save-structure-debug", action="store_true", default=False,
                    help="Write Task16C depth/edge/confidence debug images when enabled.")
    ap.add_argument("--panel-width-m", type=float, default=1.28,
                    help="Physical panel width in meters (default: 1.28).")
    ap.add_argument("--panel-height-m", type=float, default=1.60,
                    help="Physical panel height in meters (default: 1.60).")
    ap.add_argument("--left-panel-px-w", type=int, default=936,
                    help="Left panel render pixel width (default: 936).")
    ap.add_argument("--right-panel-px-w", type=int, default=936,
                    help="Right panel render pixel width (default: 936).")
    ap.add_argument("--panel-px-h", type=int, default=1176,
                    help="Panel render pixel height (default: 1176).")
    ap.add_argument("--seam-width-m", type=float, default=0.0,
                    help="Legacy diagnostic seam overlap width in meters (default: 0.0).")
    ap.add_argument("--seam-px", type=int, default=50,
                    help="Seam blend target pixels (default: 50).")
    ap.add_argument("--sweet-spot-distance-cm", type=float,
                    default=DEFAULT_SWEET_SPOT_DISTANCE_CM,
                    help="Sweet-spot horizontal distance from fold in cm "
                         "(default: 300).")
    ap.add_argument("--sweet-spot-height-cm", type=float,
                    default=DEFAULT_SWEET_SPOT_HEIGHT_CM,
                    help="Sweet-spot height in cm (default: 170).")
    ap.add_argument("--sweet-spot-height-mode",
                    choices=["eye-from-floor", "world-z"],
                    default=DEFAULT_SWEET_SPOT_HEIGHT_MODE,
                    help="Interpret sweet-spot height as real eye height above "
                         "floor or as centered world Z.")
    ap.add_argument("--screen-bottom-height-cm", type=float,
                    default=DEFAULT_SCREEN_BOTTOM_HEIGHT_CM,
                    help="Physical floor height of the active screen bottom edge.")
    args = ap.parse_args()
    if args.flow_max_side < 8:
        ap.error("--flow-max-side must be >= 8")
    if not 0.0 <= args.flow_history_weight <= 1.0:
        ap.error("--flow-history-weight must be in [0,1]")
    OUT_DIR = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(OUT_DIR, exist_ok=True)

    # Apply CLI overrides to module-level display globals.
    PANEL_PHYS_W_M = args.panel_width_m
    PANEL_PHYS_H_M = args.panel_height_m
    PANEL_PIXEL_W = args.left_panel_px_w
    PANEL_PIXEL_H = args.panel_px_h
    CONTENT_W_M = PANEL_PHYS_W_M
    if args.right_panel_px_w != args.left_panel_px_w:
        print(f"  WARNING: --right-panel-px-w ({args.right_panel_px_w}) != "
              f"--left-panel-px-w ({args.left_panel_px_w}). "
              f"Per-panel pixel differentiation not yet supported; "
              f"using left value for both panels.")
    OVERLAP_WORLD_M = args.seam_width_m
    BLEND_PIXELS_TARGET = args.seam_px
    seam_support_mode = _normalize_seam_boundary_support_mode(
        args.seam_boundary_support_mode)
    if args.enable_seam_boundary_support:
        seam_support_mode = "on"
    if args.disable_seam_boundary_support:
        seam_support_mode = "off"
    SEAM_BOUNDARY_SUPPORT_MODE = seam_support_mode
    SEAM_BOUNDARY_SUPPORT_ENABLED = seam_support_mode != "off"
    SEAM_BOUNDARY_SUPPORT_BAND_PX = max(0, int(args.seam_boundary_support_band_px))
    SEAM_BOUNDARY_SUPPORT_SCALE_FLOOR_PANEL_PITCH = max(
        0.0, float(args.seam_boundary_support_scale_floor_panel_pitch))
    SEAM_BOUNDARY_SUPPORT_SURFACE_EPS_M = max(
        0.0, float(args.seam_boundary_support_surface_eps_mm) / 1000.0)
    SWEET_SPOT, SWEET_SPOT_PROFILE = resolve_sweet_spot(
        distance_cm=args.sweet_spot_distance_cm,
        height_cm=args.sweet_spot_height_cm,
        panel_height_m=PANEL_PHYS_H_M,
        height_mode=args.sweet_spot_height_mode,
        screen_bottom_height_cm=args.screen_bottom_height_cm)

    print("=" * 64)
    print("  E2E PIPELINE v3: Z-up, BGR->RGB, Aspect-Preserving")
    print("=" * 64)
    print(f"  Device: {DEVICE}")
    print(f"  Panel:  {PANEL_PHYS_W_M:.3f}x{PANEL_PHYS_H_M:.3f} m "
          f"({PANEL_PIXEL_W}x{PANEL_PIXEL_H} px)")
    print(f"  Sweet Spot: {tuple(SWEET_SPOT)} "
          f"({SWEET_SPOT_PROFILE['mode']}, "
          f"screen_bottom={SWEET_SPOT_PROFILE['screen_bottom_height_cm']:.1f}cm)")
    print(f"  Content: {CONTENT_W_M:.3f} m wide (height adapts to input aspect)")
    print(f"  Depth: [{DEPTH_BACK_M:+.2f}, {DEPTH_FWD_M:+.2f}] m")
    print(f"  Task16 depth structure: {args.depth_structure_mode} "
          f"(process_res={args.depth_process_res})")
    print(f"  Task16 Gaussian representation: {args.gaussian_representation}")
    print(f"  Pointcloud backend request: {args.pointcloud_backend}")
    edge_repair_enabled = bool(args.enable_edge_aware_gaussian_repair)
    if args.disable_edge_aware_gaussian_repair:
        edge_repair_enabled = False
    print(f"  Edge-aware Gaussian repair: {'ENABLED' if edge_repair_enabled else 'DISABLED'}")
    contract_proto_enabled = bool(args.enable_representation_contract_prototype)
    if args.disable_representation_contract_prototype:
        contract_proto_enabled = False
    print("  Representation-contract prototype: "
          f"{'ENABLED' if contract_proto_enabled else 'DISABLED'}")
    surface_patch_enabled = bool(args.enable_surface_patch_prototype)
    if args.disable_surface_patch_prototype:
        surface_patch_enabled = False
    print("  Surface/patch prototype: "
          f"{'ENABLED' if surface_patch_enabled else 'DISABLED'}")
    print("  Seam-boundary support: "
          f"{_active_seam_boundary_support_mode().upper()} "
          f"(band={SEAM_BOUNDARY_SUPPORT_BAND_PX}px, "
          f"floor={SEAM_BOUNDARY_SUPPORT_SCALE_FLOOR_PANEL_PITCH:.3f}x panel pitch)")

    runtime_save_profile = (
        "minimal" if args.video and not args.save_diagnostics else "diagnostic")
    runtime_pointcloud_backend, runtime_pointcloud_reason = (
        _resolve_pointcloud_backend(
            args.pointcloud_backend,
            save_profile=runtime_save_profile,
            fg_mask_present=False,
            depth_structure_mode=args.depth_structure_mode,
            gaussian_representation=args.gaussian_representation,
            edge_aware_gaussian_repair=edge_repair_enabled,
            representation_contract_prototype=contract_proto_enabled,
            surface_patch_level_prototype=surface_patch_enabled,
            needs_representation_categories=(
                contract_proto_enabled or surface_patch_enabled),
            needs_geometry_metadata=(
                args.gaussian_representation == "surfel"),
        )
    )
    print(
        f"  Pointcloud backend resolved: {runtime_pointcloud_backend} "
        f"({runtime_pointcloud_reason})"
    )

    #    Shared resources: DAV3 model + camera + temporal anchor
    model_dav3 = load_dav3("da3-small", device=str(DEVICE))
    camera = build_dual_camera(mirror_left=args.mirror_left,
                               mirror_right=args.mirror_right)
    print_frustum_diagnostics(camera)

    # ``最终播放.mp4`` was produced by the no-segmentation path.  This is now
    # the only production behavior; the compatibility keyword remains accepted
    # by process_frame() for older harness callers.
    model_sam3 = False
    print("  FG/BG split: DISABLED (canonical no-segmentation runtime)")

    # Temporal depth anchor   EMA-smoothed percentile bounds prevent
    # "breathing jelly" scale drift across video frames.
    depth_anchor = DepthAnchorStream(
        physical_near=DEPTH_FWD_M, physical_far=DEPTH_BACK_M, alpha=0.05,
        raw_depth_semantic="ray_depth_high_far")
    print(f"\n  Temporal anchor: alpha={depth_anchor.alpha}, "
          f"p_lo={depth_anchor.p_lo}, p_hi={depth_anchor.p_hi}, "
          f"raw_depth_semantic={depth_anchor.raw_depth_semantic}")

    #                                                                    
    # Video mode   VideoDepthStreamer + EMA drives the main loop
    #                                                                    
    if args.video:
        video_path = os.path.expanduser(args.video) if args.video.startswith("~") \
                     else args.video
        if not os.path.exists(video_path):
            print(f"\n  ERROR: Video not found: {video_path}")
            return

        # Build a depth_fn closure that wraps the pre-loaded DAV3 model.
        # Runs on the MAIN thread inside get_smoothed_frame()   the
        # background thread only handles cv2 I/O.
        def dav3_depth_fn(rgb_np):
            return predict_depth_fast(
                rgb_np, model_dav3, process_res=args.depth_process_res)

        flow_estimator = None
        if args.flow_confidence != "off":
            from l_splat.flow_confidence import FlowConfidenceEstimator

            checkpoint_cached = FlowConfidenceEstimator.checkpoint_is_cached(
                "small")
            if args.flow_confidence == "auto" and not checkpoint_cached:
                print(
                    "\n  [FlowConfidence] RAFT-small checkpoint is not "
                    "cached; using legacy EMA in auto mode.")
            else:
                try:
                    flow_estimator = FlowConfidenceEstimator(
                        model_size="small",
                        device=str(DEVICE),
                        fb_threshold=0.5,
                        max_side=args.flow_max_side,
                        num_flow_updates=12,
                    )
                except Exception as exc:
                    if args.flow_confidence == "on":
                        raise RuntimeError(
                            "FlowConfidence was required but RAFT initialisation "
                            f"failed: {exc}") from exc
                    print(
                        "\n  [FlowConfidence] initialisation failed; using "
                        f"legacy EMA: {exc}")

        temporal_label = (
            "FLOW TWO-FRAME" if flow_estimator is not None else "EMA")
        print(f"\n  [VIDEO MODE + {temporal_label}] {video_path}")
        streamer = VideoDepthStreamer(video_path, alpha=args.ema_alpha,
                                      frame_stride=args.frame_stride,
                                      depth_fn=dav3_depth_fn,
                                      flow_estimator=flow_estimator,
                                      flow_history_weight=args.flow_history_weight,
                                      flow_fail_open=(
                                          args.flow_confidence == "auto"))
        print(f"  Resolution:  {streamer.resolution[0]}x{streamer.resolution[1]}")
        print(f"  Total frames:{streamer.total_frames}")
        print(f"  Frame stride:{streamer._frame_stride}")
        print(f"  EMA alpha:   {streamer.alpha}")
        print(f"  Flow mode:   {args.flow_confidence} "
              f"({'active' if streamer.flow_enabled else 'EMA fallback'})")
        if streamer.flow_enabled:
            print(f"  Flow work:   max_side={args.flow_max_side}, "
                  f"updates=12, history_weight={args.flow_history_weight:.2f}, "
                  "support=2 raw frames")
        print(f"  Expected:    {streamer.expected_frames} output frames")
        max_frames = args.max_frames if args.max_frames > 0 else streamer.expected_frames
        print(f"  Limit:       {max_frames} frames")
        print()

        t_start = time.perf_counter()
        frame_idx = 0
        processed = 0
        gaussian_reuse_state = GaussianFrameReuseState()
        print(
            "  Gaussian storage: persistent GPU buffers + in-place updates "
            "(Gaussian calculations still run per frame)"
        )

        while processed < max_frames:
            result = streamer.get_smoothed_frame(device="cpu")
            if result is None:
                print(f"  Video exhausted after {processed} frames.")
                break

            rgb_tensor, depth_smoothed_tensor = result
            # Convert GPU/CPU tensors -> numpy for process_frame
            rgb_np = (rgb_tensor.cpu().numpy() * 255).astype(np.uint8)
            depth_np = depth_smoothed_tensor.cpu().numpy()
            H, W = rgb_np.shape[:2]
            temporal_metrics = streamer.last_temporal_metrics
            temporal_suffix = ""
            if temporal_metrics.get("mode") == "flow_two_frame":
                temporal_suffix = (
                    f"flow={temporal_metrics.get('flow_total_ms', 0.0):.1f}ms "
                    f"fuse={temporal_metrics.get('fusion_ms', 0.0):.1f}ms "
                    f"conf={temporal_metrics.get('mean_confidence', 0.0):.3f}")

            print(f"     Frame {frame_idx:04d} ({W}x{H}) "
                  f"depth=[{depth_np.min():.3f}, {depth_np.max():.3f}] "
                  f"{temporal_suffix}")

            frame_out_dir = os.path.join(OUT_DIR, f"frame_{frame_idx:04d}")
            os.makedirs(frame_out_dir, exist_ok=True)
            temporal_metrics_path = os.path.join(
                frame_out_dir, "temporal_metrics.json")
            with open(temporal_metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "schema": "l_splat_temporal_metrics_v1",
                    "frame_idx": frame_idx,
                    "requested_mode": args.flow_confidence,
                    "metrics": temporal_metrics,
                }, f, indent=2)

            try:
                # Pass EMA-smoothed depth -> process_frame skips its own DAV3.
                # DepthAnchorStream provides temporally stable normalization.
                process_frame(rgb_np, model_dav3, camera,
                              out_dir=frame_out_dir, frame_idx=frame_idx,
                              depth_map=depth_np, depth_anchor=depth_anchor,
                              model_sam3=model_sam3,
                              edge_aware_gaussian_repair=edge_repair_enabled,
                              representation_contract_prototype=contract_proto_enabled,
                              surface_patch_level_prototype=surface_patch_enabled,
                              depth_structure_mode=args.depth_structure_mode,
                              gaussian_representation=args.gaussian_representation,
                              depth_process_res=args.depth_process_res,
                              save_structure_debug=args.save_structure_debug,
                              save_profile=(
                                  "diagnostic" if args.save_diagnostics
                                  else "minimal"),
                              pointcloud_backend=args.pointcloud_backend,
                              gaussian_reuse_state=gaussian_reuse_state,
                              run_metadata={
                                  "mode": "video",
                                  "input_path": video_path,
                                  "frame_idx": frame_idx,
                                  "run_id": frame_out_dir,
                                  "argv": sys.argv[1:],
                                  "sweet_spot_profile": SWEET_SPOT_PROFILE,
                                  "temporal": temporal_metrics,
                              })
                processed += 1
            except Exception as e:
                print(f"  ERROR processing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                if runtime_pointcloud_backend == "cuda":
                    streamer.close()
                    raise

            frame_idx += 1

        elapsed = time.perf_counter() - t_start
        streamer.close()
        final_temporal_label = (
            "FLOW TWO-FRAME" if streamer.flow_enabled else "EMA")

        fps_pipeline = processed / elapsed if elapsed > 0 else 0
        print("\n" + "=" * 64)
        print(f"  VIDEO PIPELINE + {final_temporal_label} COMPLETE")
        print("=" * 64)
        print(f"  Frames processed:  {processed}")
        print(f"  EMA alpha:         {args.ema_alpha}")
        print(f"  Flow mode:         {args.flow_confidence}")
        print(f"  Flow active:       {streamer.flow_enabled}")
        print(
            "  Gaussian reuse:    "
            f"reuse={gaussian_reuse_state.reuse_count}, "
            f"realloc={gaussian_reuse_state.reallocation_count}, "
            f"capacity={gaussian_reuse_state.capacity}"
        )
        print(f"  Total elapsed:     {elapsed:.1f}s")
        print(f"  Pipeline FPS:      {fps_pipeline:.3f} FPS "
              f"({elapsed/processed*1000:.0f} ms/frame)" if processed else "")
        print(f"  Output:            {OUT_DIR}/frame_XXXX/")
        return

    #                                                                    
    # Image mode   single-image pipeline (backward compatible)
    #                                                                    
    if args.image:
        img_path = os.path.expanduser(args.image) if args.image.startswith("~") \
                   else args.image
    else:
        img_path = os.path.join(os.path.expanduser("~"), "Desktop", "hypercar.jpg")

    print(f"\n  [IMAGE MODE] Input: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  ERROR: Could not load {img_path}")
        return
    rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = rgb_orig.shape[:2]
    print(f"  Original: {W_orig}x{H_orig}  (RGB channels confirmed)")

    process_frame(rgb_orig, model_dav3, camera,
                  out_dir=OUT_DIR, frame_idx=None,
                  depth_anchor=depth_anchor,
                  model_sam3=model_sam3,
                  edge_aware_gaussian_repair=edge_repair_enabled,
                  representation_contract_prototype=contract_proto_enabled,
                  surface_patch_level_prototype=surface_patch_enabled,
                  depth_structure_mode=args.depth_structure_mode,
                  gaussian_representation=args.gaussian_representation,
                  depth_process_res=args.depth_process_res,
                  save_structure_debug=args.save_structure_debug,
                  save_profile="diagnostic",
                  pointcloud_backend=args.pointcloud_backend,
                  run_metadata={
                      "mode": "image",
                      "input_path": img_path,
                      "frame_idx": None,
                      "run_id": OUT_DIR,
                      "argv": sys.argv[1:],
                      "sweet_spot_profile": SWEET_SPOT_PROFILE,
                  })

    print("\n" + "=" * 64)
    print("  PIPELINE v3 COMPLETE")
    print("=" * 64)
    print(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
