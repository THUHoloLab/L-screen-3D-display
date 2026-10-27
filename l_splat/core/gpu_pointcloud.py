"""CUDA implementation of the canonical L-Splat pointcloud hot path.

The production helper is deliberately narrow: it mirrors the active
no-segmentation, isotropic, seam-support-off NumPy path and returns the same
seven arrays as CUDA tensors.  Backend selection and fallback live in
``l_splat.core.pipeline`` so unsupported experimental modes keep using the
unchanged CPU implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


COORDINATE_LIMIT_M = 10.0
DEPTH_RANGE_EPSILON = 1.0e-6
WORLD_SIGN_EPSILON_M = 1.0e-6
SUBPIXEL_CLIP_MARGIN_PX = 1.001
SEAM_EXCLUSION_RADIUS_PX = 1
RGB_NORMALIZATION_DENOMINATOR = 255.0


def _validate_inputs(
        rgb_fullres: np.ndarray,
        depth_fullres: np.ndarray,
        depth_norm: np.ndarray | None,
        *,
        super_sample_size: int,
        max_gaussians: int,
        device: torch.device,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    """Validate and normalize the production CUDA pointcloud contract.

    Inputs are RGB uint8 ``(H,W,3)``, raw depth ``(H,W)``, optional externally
    anchored normalized depth, sampling/capacity integers, and a CUDA device.
    Outputs are contiguous input arrays plus integer height/width.  Shape,
    dtype, capacity, or CUDA availability violations raise before allocation.
    """
    rgb = np.asarray(rgb_fullres)
    depth = np.asarray(depth_fullres)
    anchored = None if depth_norm is None else np.asarray(depth_norm)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb_fullres must have shape (H,W,3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb_fullres must be uint8, got {rgb.dtype}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"depth_fullres must have shape {rgb.shape[:2]}, got {depth.shape}")
    if anchored is not None and anchored.shape != rgb.shape[:2]:
        raise ValueError(
            f"depth_norm must have shape {rgb.shape[:2]}, got {anchored.shape}")
    height, width = (int(rgb.shape[0]), int(rgb.shape[1]))
    if height < 2 or width < 2:
        raise ValueError("CUDA pointcloud requires image height and width >= 2")
    if int(super_sample_size) < 1:
        raise ValueError("super_sample_size must be >= 1")
    if int(max_gaussians) < height * width:
        raise ValueError(
            f"max_gaussians ({max_gaussians}) is below the base pixel count "
            f"({height * width})")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA pointcloud requires an available CUDA device")
    return (
        np.ascontiguousarray(rgb),
        np.ascontiguousarray(depth),
        None if anchored is None else np.ascontiguousarray(anchored),
        height,
        width,
    )


def _observer_panel_tangent_coordinate(
        panel_coord: torch.Tensor,
        normal_disp: torch.Tensor,
        eye_tangent: float,
        eye_normal: float,
        ) -> torch.Tensor:
    """Apply the active observer-ray tangent correction on CUDA.

    Inputs are float32 panel coordinates/displacements and scalar sweet-spot
    coordinates in metres.  The output uses
    ``eye_t + (panel-eye_t)*(eye_n-disp)/eye_n``.  A zero eye-normal value is
    rejected because the production GPU path must fail closed on bad geometry.
    """
    if abs(float(eye_normal)) < 1.0e-8:
        raise ValueError("sweet-spot eye-normal coordinate must be non-zero")
    return float(eye_tangent) + (panel_coord - float(eye_tangent)) * (
        (float(eye_normal) - normal_disp) / float(eye_normal)
    )


def _bounded_xyz_mask(
        x_coord: torch.Tensor,
        y_coord: torch.Tensor,
        z_coord: torch.Tensor,
        ) -> torch.Tensor:
    """Return the production finite and ``(-10m,10m)`` coordinate mask.

    Inputs are same-shaped CUDA coordinate tensors.  Output is a bool tensor of
    that shape.  Torch shape/broadcasting errors propagate if inputs disagree.
    """
    limit = float(COORDINATE_LIMIT_M)
    return (
        torch.isfinite(x_coord)
        & torch.isfinite(y_coord)
        & torch.isfinite(z_coord)
        & (x_coord > -limit)
        & (x_coord < limit)
        & (y_coord > -limit)
        & (y_coord < limit)
        & (z_coord > -limit)
        & (z_coord < limit)
    )


def depth_to_unified_pointcloud_cuda(
        rgb_fullres: np.ndarray,
        depth_fullres: np.ndarray,
        depth_near: float,
        depth_far: float,
        content_w: float,
        content_h: float,
        *,
        sweet_spot: tuple[float, float, float],
        super_sample_size: int,
        local_spacing_threshold: float,
        max_gaussians: int,
        depth_norm: np.ndarray | None = None,
        depth_power: float = 1.0,
        device: torch.device | str = "cuda",
        ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Build the canonical seven-field pointcloud directly on CUDA.

    Inputs match the active CPU box projection, including optional anchored
    ``depth_norm`` from ``DepthAnchorStream``.  Outputs are CUDA tensors in
    production order: points, colors, X sign, super-sample flag, panel owner,
    background flag, and edge-aware flag, followed by JSON-safe metadata.

    Geometry uses the seam-exclusive two-panel box mapping and the exact
    offset-major adaptive 4x4 interpolation layout.  Before subpixel tensors
    are allocated, the upper-bound candidate count is compared with
    ``max_gaussians``; overflow keeps base pixels only, matching the CPU
    production policy.  Invalid inputs and CUDA execution/allocation failures
    propagate rather than silently changing backend or output semantics.
    """
    device = torch.device(device)
    (
        rgb_np,
        depth_np,
        anchored_np,
        height,
        width,
    ) = _validate_inputs(
        rgb_fullres,
        depth_fullres,
        depth_norm,
        super_sample_size=int(super_sample_size),
        max_gaussians=int(max_gaussians),
        device=device,
    )
    if len(sweet_spot) != 3:
        raise ValueError("sweet_spot must contain exactly three coordinates")

    rgb = torch.as_tensor(rgb_np, device=device, dtype=torch.uint8)
    depth = torch.as_tensor(depth_np, device=device, dtype=torch.float32)
    if anchored_np is not None:
        normalized = torch.as_tensor(
            anchored_np, device=device, dtype=torch.float32)
        normalization_mode = "provided"
    else:
        depth_min = float(depth.min().item())
        depth_max = float(depth.max().item())
        if depth_max - depth_min < DEPTH_RANGE_EPSILON:
            normalized = torch.zeros_like(depth)
        else:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        normalization_mode = "per_frame_minmax"
    normalized_compressed = torch.pow(normalized, float(depth_power))
    displacement = float(depth_far) + normalized_compressed * (
        float(depth_near) - float(depth_far))

    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    vertical_norm = vv / float(max(height - 1, 1))
    world_z = (0.5 - vertical_norm) * float(content_h)

    split_col = width // 2
    left_mask = uu < float(split_col)
    left_u_norm = torch.clamp(
        uu / float(max(split_col - 1, 1)), min=0.0, max=1.0)
    left_panel_x = -float(content_w) + left_u_norm * float(content_w)
    left_y = displacement
    left_x = _observer_panel_tangent_coordinate(
        left_panel_x,
        left_y,
        float(sweet_spot[0]),
        float(sweet_spot[1]),
    )

    right_u_norm = torch.clamp(
        (uu - float(split_col)) / float(max(width - split_col - 1, 1)),
        min=0.0,
        max=1.0,
    )
    right_x = displacement
    right_panel_y = -right_u_norm * float(content_w)
    right_y = _observer_panel_tangent_coordinate(
        right_panel_y,
        right_x,
        float(sweet_spot[1]),
        float(sweet_spot[0]),
    )

    world_x = torch.where(left_mask, left_x, right_x).to(torch.float32)
    world_y = torch.where(left_mask, left_y, right_y).to(torch.float32)
    owner_full = torch.where(
        left_mask,
        torch.zeros((), device=device, dtype=torch.uint8),
        torch.ones((), device=device, dtype=torch.uint8),
    )
    sign_full = torch.where(
        world_x > WORLD_SIGN_EPSILON_M,
        torch.ones((), device=device, dtype=torch.int8),
        torch.where(
            world_x < -WORLD_SIGN_EPSILON_M,
            -torch.ones((), device=device, dtype=torch.int8),
            torch.zeros((), device=device, dtype=torch.int8),
        ),
    )

    base_valid = _bounded_xyz_mask(world_x, world_y, world_z).reshape(-1)
    points_base = torch.stack([
        world_x.reshape(-1)[base_valid],
        world_y.reshape(-1)[base_valid],
        world_z.reshape(-1)[base_valid],
    ], dim=-1)
    colors_base = (
        rgb.reshape(-1, 3).to(torch.float32)[base_valid]
        / RGB_NORMALIZATION_DENOMINATOR
    )
    sign_base = sign_full.reshape(-1)[base_valid]
    owner_base = owner_full.reshape(-1)[base_valid]

    spacing_u = torch.sqrt(
        torch.diff(world_x, dim=1).abs().square()
        + torch.diff(world_y, dim=1).abs().square()
        + torch.diff(world_z, dim=1).abs().square()
    )
    spacing_v = torch.sqrt(
        torch.diff(world_x, dim=0).abs().square()
        + torch.diff(world_y, dim=0).abs().square()
        + torch.diff(world_z, dim=0).abs().square()
    )
    spacing_u = torch.cat([spacing_u, spacing_u[:, -1:]], dim=1)
    spacing_v = torch.cat([spacing_v, spacing_v[-1:, :]], dim=0)
    spacing_local = torch.maximum(spacing_u, spacing_v)
    spacing_threshold = (
        float(content_w) / float(width) * float(local_spacing_threshold))
    seam_zone = torch.abs(uu - float(split_col)) <= SEAM_EXCLUSION_RADIUS_PX
    adaptive_mask = (spacing_local > spacing_threshold) & ~seam_zone
    mask_rows, mask_cols = torch.where(adaptive_mask)
    masked_count = int(mask_rows.numel())
    offset_count = int(super_sample_size) * int(super_sample_size)
    base_count = int(points_base.shape[0])
    candidate_count = base_count + masked_count * offset_count
    skip_for_capacity = candidate_count > int(max_gaussians)

    if masked_count and not skip_for_capacity:
        offsets = torch.linspace(
            -0.375,
            0.375,
            int(super_sample_size),
            device=device,
            dtype=torch.float32,
        )
        du_grid, dv_grid = torch.meshgrid(offsets, offsets, indexing="xy")
        du = du_grid.reshape(-1, 1)
        dv = dv_grid.reshape(-1, 1)
        mask_cols_f = mask_cols.to(torch.float32).reshape(1, -1)
        mask_rows_f = mask_rows.to(torch.float32).reshape(1, -1)
        sub_u = torch.clamp(
            mask_cols_f + du,
            min=0.0,
            max=float(width) - SUBPIXEL_CLIP_MARGIN_PX,
        )
        sub_v = torch.clamp(
            mask_rows_f + dv,
            min=0.0,
            max=float(height) - SUBPIXEL_CLIP_MARGIN_PX,
        )
        u0 = torch.floor(sub_u).to(torch.int64)
        v0 = torch.floor(sub_v).to(torch.int64)
        u1 = torch.clamp(u0 + 1, min=0, max=width - 1)
        v1 = torch.clamp(v0 + 1, min=0, max=height - 1)
        weight_u = sub_u - u0.to(torch.float32)
        weight_v = sub_v - v0.to(torch.float32)
        weight_00 = (1.0 - weight_u) * (1.0 - weight_v)
        weight_10 = weight_u * (1.0 - weight_v)
        weight_01 = (1.0 - weight_u) * weight_v
        weight_11 = weight_u * weight_v
        source_cols = mask_cols.reshape(1, -1).expand(offset_count, -1)

        def interpolate(source: torch.Tensor) -> torch.Tensor:
            """Gather one coordinate using the CPU path's source-column rule."""
            return (
                weight_00 * source[v0, source_cols]
                + weight_10 * source[v0, u1]
                + weight_01 * source[v1, source_cols]
                + weight_11 * source[v1, u1]
            )

        super_x = interpolate(world_x).reshape(-1)
        super_y = interpolate(world_y).reshape(-1)
        super_z = interpolate(world_z).reshape(-1)
        nearest_u = torch.clamp(
            torch.round(sub_u).to(torch.int64), min=0, max=width - 1)
        nearest_v = torch.clamp(
            torch.round(sub_v).to(torch.int64), min=0, max=height - 1)
        colors_super = (
            rgb[nearest_v, nearest_u].reshape(-1, 3).to(torch.float32)
            / RGB_NORMALIZATION_DENOMINATOR
        )
        sign_super = sign_full[mask_rows, mask_cols].reshape(
            1, -1).expand(offset_count, -1).reshape(-1)
        owner_super = owner_full[mask_rows, mask_cols].reshape(
            1, -1).expand(offset_count, -1).reshape(-1)
        super_valid = _bounded_xyz_mask(super_x, super_y, super_z)
        points_super = torch.stack([
            super_x[super_valid],
            super_y[super_valid],
            super_z[super_valid],
        ], dim=-1)
        colors_super = colors_super[super_valid]
        sign_super = sign_super[super_valid]
        owner_super = owner_super[super_valid]
    else:
        points_super = torch.empty((0, 3), device=device, dtype=torch.float32)
        colors_super = torch.empty((0, 3), device=device, dtype=torch.float32)
        sign_super = torch.empty((0,), device=device, dtype=torch.int8)
        owner_super = torch.empty((0,), device=device, dtype=torch.uint8)

    super_count = int(points_super.shape[0])
    if super_count:
        points = torch.cat([points_base, points_super], dim=0)
        colors = torch.cat([colors_base, colors_super], dim=0)
        x_sign = torch.cat([sign_base, sign_super], dim=0)
        owner = torch.cat([owner_base, owner_super], dim=0)
    else:
        points = points_base
        colors = colors_base
        x_sign = sign_base
        owner = owner_base
    output_count = int(points.shape[0])
    is_super = torch.cat([
        torch.zeros(base_count, device=device, dtype=torch.bool),
        torch.ones(super_count, device=device, dtype=torch.bool),
    ], dim=0)
    is_background = torch.zeros(output_count, device=device, dtype=torch.bool)
    edge_mask = torch.zeros(output_count, device=device, dtype=torch.bool)

    metadata: dict[str, Any] = {
        "schema": "l_splat_gpu_pointcloud_v1",
        "backend": "cuda",
        "device": str(device),
        "input_shape": [height, width, 3],
        "depth_normalization": normalization_mode,
        "depth_power": float(depth_power),
        "base_pixel_count": height * width,
        "base_point_count": base_count,
        "adaptive_masked_pixel_count": masked_count,
        "subpixel_offset_count": offset_count,
        "candidate_point_count": candidate_count,
        "super_sampled_count": super_count,
        "output_point_count": output_count,
        "max_gaussians": int(max_gaussians),
        "supersampling_skipped_for_capacity": bool(skip_for_capacity),
        "spacing_threshold_m": float(spacing_threshold),
        "supported_mode": "no_fg_isotropic_no_optional_repairs",
    }
    return (
        points,
        colors,
        x_sign,
        is_super,
        owner,
        is_background,
        edge_mask,
    ), metadata
