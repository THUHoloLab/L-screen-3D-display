"""
modification.py
Dual Asymmetric Frustums — 3DGS Integration Module for L-shaped Screens.

Target: 90-degree outward-fold L-shaped LED screens for glasses-free 3D
(Anamorphic 3D) content generation.

This module is the thin Torch/CUDA adapter around the canonical float64 geometry
in ``l_splat.geometry.frustum``. Projection mathematics and two-pass rendering
orchestration must not be duplicated here.

Integration points with original 3DGS:
  - utils/graphics_utils.py:   replace getProjectionMatrix with asymmetric version
  - scene/cameras.py:          use DualFrustumCamera instead of Camera
  - gaussian_renderer/__init__.py: pass A_x/A_y to GaussianRasterizationSettings
  - diff-gaussian-rasterization:   use A_x/A_y in Jacobian (forward.cu / backward.cu)
"""

import numpy as np
import torch
import torch.nn as nn

from l_splat.geometry.frustum import (
    FLIP_Y,
    FLIP_X,
    look_at_offscreen,
    compute_frustum_bounds,
    build_asymmetric_proj_matrix,
)


# ===========================================================================
# Canonical geometry adapter
# ===========================================================================
# Pure geometry functions live in l_splat.geometry.frustum.


def get_asymmetric_frustum(viewer_pos, screen_corners, near, far, view_matrix=None):
    """
    Convenience: compute full (View, Proj) for a single screen.

    If view_matrix is provided, it is used as-is (the caller is responsible
    for ensuring it is consistent with the screen geometry). Otherwise the
    view matrix is computed via look_at toward the screen center.

    Returns view_matrix, proj_matrix, frustum_info.
    """
    (l, r, b, t), V = compute_frustum_bounds(viewer_pos, screen_corners, near, view_matrix)
    P, info = build_asymmetric_proj_matrix(near, far, l, r, b, t)
    return V, P, info


# ===========================================================================
# Section 3 — Dual-Frustum Camera
# ===========================================================================
# Wraps two asymmetric frustum definitions (left screen, right screen) sharing
# a single viewer position.  Compatible with 3DGS's Camera / MiniCam interface
# for drop-in use in gaussian_renderer.render().

class DualFrustumCamera(nn.Module):
    """
    Camera model holding two asymmetric frustums for an L-shaped screen pair.

    Each frustum has its own projection matrix (P_left, P_right) and combined
    view*proj matrix (full_proj_transform_left/right).

    The view matrix is shared if both screens are rendered from the same
    viewer position and orientation, but in practice each screen gets its
    own look-at direction (toward that screen's center), resulting in
    different view matrices.

    Compatible with the original 3DGS render() function — pass the relevant
    frustum's parameters when calling render().
    """

    def __init__(self,
                 viewer_pos,                 # (3,) world-space viewer position
                 screen_left_corners,        # (4, 3) left screen corners
                 screen_right_corners,       # (4, 3) right screen corners
                 znear=0.01,
                 zfar=100.0,
                 frustum_scale=1.0,          # scale corners around center to widen frustum
                 overlap_world_m=0.0,        # extend frustum past fold by this many metres
                 image_width=832,            # render target pixel width
                 image_height=936,           # render target pixel height
                 mirror_left=False,          # apply FLIP_X to left view matrix
                 mirror_right=False,         # apply FLIP_X to right view matrix
                 device="cuda"):
        super().__init__()

        viewer_pos = np.asarray(viewer_pos, dtype=np.float64)
        screen_left_corners = np.asarray(screen_left_corners, dtype=np.float64)
        screen_right_corners = np.asarray(screen_right_corners, dtype=np.float64)

        # Screen parameters — set early so overlap_px calculation can use them
        self.image_width = image_width
        self.image_height = image_height
        self.znear = znear
        self.zfar = zfar

        # Scale corners around their center to capture content beyond panel edges
        if frustum_scale != 1.0:
            left_center = screen_left_corners.mean(axis=0)
            screen_left_corners = left_center + (screen_left_corners - left_center) * frustum_scale
            right_center = screen_right_corners.mean(axis=0)
            screen_right_corners = right_center + (screen_right_corners - right_center) * frustum_scale

        # --- Overlap frustum: add virtual corners past the fold ---
        # Left camera extends +X past fold on Y=0 plane; right camera extends
        # +Y past fold on X=0 plane.  This creates a geometric overlap zone
        # where both cameras see content near the fold, enabling 2D alpha blending.
        if overlap_world_m > 0:
            Hz = (screen_left_corners[0, 2] - screen_left_corners[2, 2]) / 2.0
            left_overlap_corners = np.array([
                [overlap_world_m, 0.0,  Hz],
                [overlap_world_m, 0.0, -Hz],
            ], dtype=np.float64)
            right_overlap_corners = np.array([
                [0.0,  overlap_world_m,  Hz],
                [0.0,  overlap_world_m, -Hz],
            ], dtype=np.float64)
            screen_left_corners = np.vstack([screen_left_corners, left_overlap_corners])
            screen_right_corners = np.vstack([screen_right_corners, right_overlap_corners])

        # flip_y corrects the Y axis: look_at_offscreen produces det=1 with
        # correct fold alignment but up=(0,0,-1) → upside-down image.
        # flip_y = diag(1, -1, 1, 1) negates camera-space Y (row 1),
        # giving an upright image.  Trade-off: det becomes -1.
        flip_y = FLIP_Y

        # --- Left frustum ---
        left_target = viewer_pos + np.array([0.0, -1.0, 0.0])
        V_L_manual = look_at_offscreen(viewer_pos, left_target, np.array([0.0, 0.0, 1.0]))
        V_L_manual = flip_y @ V_L_manual
        if mirror_left:
            V_L_manual = FLIP_X @ V_L_manual
        V_L, P_L, info_L = get_asymmetric_frustum(
            viewer_pos, screen_left_corners, znear, zfar, V_L_manual)

        # --- Right frustum ---
        right_target = viewer_pos + np.array([-1.0, 0.0, 0.0])
        V_R_manual = look_at_offscreen(viewer_pos, right_target, np.array([0.0, 0.0, 1.0]))
        V_R_manual = flip_y @ V_R_manual
        if mirror_right:
            V_R_manual = FLIP_X @ V_R_manual
        V_R, P_R, info_R = get_asymmetric_frustum(
            viewer_pos, screen_right_corners, znear, zfar, V_R_manual)

        # --- Compute overlap in pixel units (for 2D alpha blending) ---
        if overlap_world_m > 0:
            # Project fold point through both frustums; how far inside is it?
            fold_pt = np.append(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)
            pL_view = V_L @ fold_pt
            pR_view = V_R @ fold_pt
            pL_clip = P_L @ pL_view
            pR_clip = P_R @ pR_view
            ndc_x_L = pL_clip[0] / pL_clip[3]
            ndc_x_R = pR_clip[0] / pR_clip[3]
            px_per_ndc = self.image_width / 2.0
            self.overlap_px_L = float((1.0 - ndc_x_L) * px_per_ndc)
            self.overlap_px_R = float((ndc_x_R - (-1.0)) * px_per_ndc)
            self.overlap_px = int(min(self.overlap_px_L, self.overlap_px_R))
        else:
            self.overlap_px = 0

        # Store as torch tensors (transposed to match 3DGS convention:
        # 3DGS stores matrices transposed for row-vector multiplication)
        self.V_left = torch.tensor(V_L, dtype=torch.float32, device=device).T
        self.P_left = torch.tensor(P_L, dtype=torch.float32, device=device).T
        self.full_proj_transform_left = self.V_left @ self.P_left
        self.camera_center_left = torch.linalg.inv(self.V_left)[3, :3]

        self.V_right = torch.tensor(V_R, dtype=torch.float32, device=device).T
        self.P_right = torch.tensor(P_R, dtype=torch.float32, device=device).T
        self.full_proj_transform_right = self.V_right @ self.P_right
        self.camera_center_right = torch.linalg.inv(self.V_right)[3, :3]

        # CUDA Jacobian parameters (A_x, A_y replace fx/fy from tan_fov)
        self.A_x_left = info_L["A_x"]
        self.A_y_left = info_L["A_y"]
        self.A_x_right = info_R["A_x"]
        self.A_y_right = info_R["A_y"]

        # Store frustum info for diagnostics
        self.info_left = info_L
        self.info_right = info_R

        # ═══ 探针: 视锥边界完整转储 (Phase 2) ═══
        # 云端架构师要求确认: top 是否过大? znear 是否过小?
        print(f"  [PROBE] Frustum Bounds Dump (znear={znear:.4f}, zfar={zfar:.1f}):")
        for side, info in [("LEFT ", info_L), ("RIGHT", info_R)]:
            print(f"    {side}: l={info['l']:.6f}  r={info['r']:.6f}  "
                  f"b={info['b']:.6f}  t={info['t']:.6f}")
            print(f"           A_x={info['A_x']:.4f}  A_y={info['A_y']:.4f}  "
                  f"offset_x={info['offset_x']:.4f}  offset_y={info['offset_y']:.4f}")
            # 关键: 检查投影矩阵除零风险
            # P[0,0] = 2n/(r-l), 若 r-l 极小则爆
            rl = info['r'] - info['l']
            tb = info['t'] - info['b']
            print(f"           r-l={rl:.6f}  t-b={tb:.6f}  "
                  f"2n/(r-l)={2*znear/rl:.4f}  2n/(t-b)={2*znear/tb:.4f}")
        print(f"  [/PROBE]")

# ===========================================================================
# Diagnostics
# ===========================================================================

def print_frustum_diagnostics(camera: DualFrustumCamera):
    """Print human-readable frustum parameters for a DualFrustumCamera."""
    print("=" * 60)
    print("  Dual-Frustum Camera Diagnostics")
    print("=" * 60)
    for side, info in [("Left", camera.info_left), ("Right", camera.info_right)]:
        print(f"\n  {side} Screen:")
        print(f"    Frustum bounds: l={info['l']:.4f} r={info['r']:.4f} "
              f"b={info['b']:.4f} t={info['t']:.4f}")
        print(f"    A_x (CUDA Jacobian): {info['A_x']:.4f}")
        print(f"    A_y (CUDA Jacobian): {info['A_y']:.4f}")
        print(f"    Off-axis offset_x:   {info['offset_x']:+.4f}")
        print(f"    Off-axis offset_y:   {info['offset_y']:+.4f}")
    if camera.overlap_px > 0:
        print(f"\n  Overlap frustum: {camera.overlap_px} px geometric overlap per side")
        print(f"    (left={camera.overlap_px_L:.1f}px, right={camera.overlap_px_R:.1f}px)")
    print(f"\n{'='*60}")
