#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixAsymmetric
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


# ---------------------------------------------------------------------------
# Dual-Frustum helper: OpenGL look-at (numpy), self-contained.
# ---------------------------------------------------------------------------

def _look_at_opengl_np(eye, center, up):
    """Build view matrix (world->view), camera looks down -Z (OpenGL convention)."""
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    u = up / np.linalg.norm(up)
    s = np.cross(fwd, u)
    s /= np.linalg.norm(s)
    u = np.cross(s, fwd)
    V = np.eye(4, dtype=np.float64)
    V[0, :3] = s
    V[1, :3] = u
    V[2, :3] = -fwd
    V[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(fwd, eye)]
    return V


def _build_asymmetric_frustum_np(viewer_pos, screen_corners, near, far):
    """
    Compute (V, P, A_x, A_y, off_x, off_y) for one screen panel.

    Returns:
        V:       (4,4) view matrix (OpenGL convention, camera looks -Z)
        P:       (4,4) projection matrix (z_sign=-1)
        A_x:     P[0,0] = 2n/(r-l)  -- replaces fx in CUDA Jacobian
        A_y:     P[1,1] = 2n/(t-b)  -- replaces fy in CUDA Jacobian
        off_x:   P[0,2] = (r+l)/(r-l)
        off_y:   P[1,2] = (t+b)/(t-b)
    """
    viewer_pos = np.asarray(viewer_pos, dtype=np.float64)
    screen_corners = np.asarray(screen_corners, dtype=np.float64)
    near, far = float(near), float(far)

    screen_center = np.mean(screen_corners, axis=0)
    screen_up = screen_corners[0] - screen_corners[2]
    screen_up /= np.linalg.norm(screen_up)

    V = _look_at_opengl_np(viewer_pos, screen_center, screen_up)

    corners_h = np.hstack([screen_corners, np.ones((4, 1), dtype=np.float64)])
    corners_view = (V @ corners_h.T).T
    z_vals = corners_view[:, 2]
    if np.any(z_vals >= 0):
        raise ValueError(f"Screen corners behind camera (z_view >= 0). Adjust viewer position.")

    t_vals = -near / z_vals
    x_near = corners_view[:, 0] * t_vals
    y_near = corners_view[:, 1] * t_vals

    l, r = float(np.min(x_near)), float(np.max(x_near))
    b, t = float(np.min(y_near)), float(np.max(y_near))

    if r - l < 1e-12 or t - b < 1e-12:
        raise ValueError(f"Degenerate frustum: r-l={r-l:.3e}, t-b={t-b:.3e}")

    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * near / (r - l)
    P[1, 1] = 2.0 * near / (t - b)
    P[0, 2] = (r + l) / (r - l)
    P[1, 2] = (t + b) / (t - b)
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0

    return V, P, P[0, 0], P[1, 1], P[0, 2], P[1, 2]


# ---------------------------------------------------------------------------
# DualFrustumCamera — replaces Camera for L-shaped screen rendering
# ---------------------------------------------------------------------------

class DualFrustumCamera(nn.Module):
    """
    Dual asymmetric frustum camera for L-shaped 90-degree outward-fold screens.

    Shares a single viewer position, produces two independent view+projection
    matrices (one per screen panel). Uses OpenGL convention (z_sign=-1) for
    correct off-axis projection.

    The stored matrices follow 3DGS transpose convention:
        world_view_transform = V^T
        full_proj_transform  = V^T @ P^T

    Compatible with the modified CUDA rasterizer that accepts A_x/A_y in place
    of tanfovx/tanfovy.
    """

    def __init__(self, viewer_pos, screen_left_corners, screen_right_corners,
                 znear=0.01, zfar=100.0, image_width=832, image_height=936,
                 data_device="cuda"):
        super(DualFrustumCamera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception:
            print(f"[Warning] Custom device {data_device} failed, fallback to cuda")
            self.data_device = torch.device("cuda")

        self.image_width = image_width
        self.image_height = image_height
        self.znear = znear
        self.zfar = zfar
        self.FoVx = float('nan')
        self.FoVy = float('nan')

        # Build left frustum
        V_L, P_L, ax_L, ay_L, ox_L, oy_L = _build_asymmetric_frustum_np(
            viewer_pos, screen_left_corners, znear, zfar)
        # Build right frustum
        V_R, P_R, ax_R, ay_R, ox_R, oy_R = _build_asymmetric_frustum_np(
            viewer_pos, screen_right_corners, znear, zfar)

        # Store info dicts
        self.info_left = {"A_x": ax_L, "A_y": ay_L, "offset_x": ox_L, "offset_y": oy_L}
        self.info_right = {"A_x": ax_R, "A_y": ay_R, "offset_x": ox_R, "offset_y": oy_R}

        # Shared A_x/A_y (identical for identically-sized panels, same near)
        self.A_x = ax_L
        self.A_y = ay_L

        # Convert to torch tensors in 3DGS transpose convention
        V_L_t = torch.tensor(V_L, dtype=torch.float32, device=self.data_device).T
        P_L_t = torch.tensor(P_L, dtype=torch.float32, device=self.data_device).T
        V_R_t = torch.tensor(V_R, dtype=torch.float32, device=self.data_device).T
        P_R_t = torch.tensor(P_R, dtype=torch.float32, device=self.data_device).T

        # Left
        self.world_view_transform_left = V_L_t
        self.projection_matrix_left = P_L_t
        self.full_proj_transform_left = (
            V_L_t.unsqueeze(0).bmm(P_L_t.unsqueeze(0))
        ).squeeze(0)

        # Right
        self.world_view_transform_right = V_R_t
        self.projection_matrix_right = P_R_t
        self.full_proj_transform_right = (
            V_R_t.unsqueeze(0).bmm(P_R_t.unsqueeze(0))
        ).squeeze(0)

        # Camera centers (for direction computation in renderer)
        V_L_inv = torch.inverse(V_L_t)
        V_R_inv = torch.inverse(V_R_t)
        self.camera_center_left = V_L_inv[3, :3]
        self.camera_center_right = V_R_inv[3, :3]

        # Legacy aliases for MiniCam / render compatibility
        # Default to left frustum for code expecting a single set
        self.world_view_transform = self.world_view_transform_left
        self.full_proj_transform = self.full_proj_transform_left
        self.projection_matrix = self.projection_matrix_left
        self.camera_center = self.camera_center_left

    def get_left_minicam(self):
        """Return a MiniCam wrapping the left frustum (for single-pass rendering)."""
        return MiniCam(self.image_width, self.image_height,
                       self.FoVy, self.FoVx, self.znear, self.zfar,
                       self.world_view_transform_left,
                       self.full_proj_transform_left)

    def get_right_minicam(self):
        """Return a MiniCam wrapping the right frustum (for single-pass rendering)."""
        return MiniCam(self.image_width, self.image_height,
                       self.FoVy, self.FoVx, self.znear, self.zfar,
                       self.world_view_transform_right,
                       self.full_proj_transform_right)

