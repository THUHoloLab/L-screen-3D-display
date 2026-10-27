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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out


def render_dual_frustum(dual_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                        scaling_modifier=1.0, separate_sh=False, override_color=None,
                        use_trained_exp=False):
    """
    Two-pass render for L-shaped 90-degree dual-frustum screens.

    Uses asymmetric projection matrices (z_sign=-1, OpenGL convention) from
    a DualFrustumCamera. Each pass renders all Gaussians through one frustum.
    The pseudo-tanfov values are derived from the asymmetric projection matrix:
        tanfovx = 1/A_x  (so that CUDA fx = 1/(2*tanfovx) = A_x/2, correct Jacobian)

    Args:
        dual_camera: DualFrustumCamera with left/right frustums.
        pc:          GaussianModel.
        pipe:        PipelineParams (rendering config).
        bg_color:    Background color tensor (on GPU).
        scaling_modifier, separate_sh, override_color, use_trained_exp:
                     Forwarded to rasterizer (same as render()).

    Returns:
        dict with keys:
            render_left, render_right, depth_left, depth_right,
            viewspace_points_left, viewspace_points_right,
            visibility_filter_left, visibility_filter_right,
            radii_left, radii_right
    """
    # --- Shared Gaussian data (invariant between passes) ---
    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - dual_camera.camera_center_left.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # SH boundary reduction: at the fold seam (world_x ≈ 0), left and right
    # cameras see the same Gaussian from ~90° apart, causing SH view-dependent
    # color disparity that creates visible "wobbly" artifacts at the stitch line.
    # For Gaussians within FOLD_BOUNDARY_M of the X=0 plane, zero all SH bands
    # beyond DC (l=0), making color view-independent and eliminating the seam.
    FOLD_BOUNDARY_M = 0.005  # 5 mm from X=0
    if pc.active_sh_degree > 0 and shs is not None:
        boundary_mask = torch.abs(means3D[:, 0]) < FOLD_BOUNDARY_M
        n_boundary = boundary_mask.sum().item()
        if n_boundary > 0:
            shs = shs.clone()
            # shs shape: (N, (deg+1)^2, 3). Index 0 = DC, indices 1: = higher bands.
            shs[boundary_mask, 1:, :] = 0.0
            if separate_sh and colors_precomp is None:
                pass  # dc remains untouched, shs already zeroed
            print(f"  SH boundary: {n_boundary}/{means3D.shape[0]} Gaussians clamped to DC-only "
                  f"(|X| < {FOLD_BOUNDARY_M*1000:.1f}mm)")

    # ═══════════════════════════════════════════════════════════════
    # [DIAGNOSTIC PROBE A] Active SH Degree + Tensor Shape
    sh_tensor = shs if shs is not None else (dc if separate_sh else None)
    sh_label = "shs" if shs is not None else ("dc+shs" if separate_sh else "colors_precomp")
    print(f"Probe A - Active SH Degree: {pc.active_sh_degree}")
    print(f"Probe A - SHs Tensor Shape: {sh_tensor.shape if sh_tensor is not None else 'N/A'}  (source: {sh_label})")
    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════
    # REVIEW 2 — CUDA Boundary Interception (scales at rasterizer entry)
    # get_scaling is a @property — no parens needed, returns (N,3) tensor.
    review2_scales = pc.get_scaling  # (N, 3) physical meters, exp(_scaling)
    ss_attr_r2 = getattr(pc, '_is_super_sampled', None)
    if ss_attr_r2 is not None and len(ss_attr_r2) == review2_scales.shape[0]:
        r2_mask = ss_attr_r2
        r2_ss_scales = review2_scales[r2_mask]
        print(f"[REVIEW 2] CUDA Boundary — actual scales tensor at rasterizer entry:")
        print(f"  Scales shape: {review2_scales.shape}, dtype: {review2_scales.dtype}")
        print(f"  ALL points:  min={review2_scales.min().item()*1000:.4f}mm  "
              f"mean={review2_scales.mean().item()*1000:.4f}mm  "
              f"max={review2_scales.max().item()*1000:.4f}mm")
        if r2_ss_scales.numel() > 0:
            print(f"  Super-sampled: min={r2_ss_scales.min().item()*1000:.4f}mm  "
                  f"mean={r2_ss_scales.mean().item()*1000:.4f}mm  "
                  f"max={r2_ss_scales.max().item()*1000:.4f}mm  "
                  f"median={r2_ss_scales.median().item()*1000:.4f}mm")
        else:
            print(f"  Super-sampled: 0 points")
    else:
        print(f"[REVIEW 2] CUDA Boundary — NO super-sampled mask on model")
    # ═══════════════════════════════════════════════════════════════

    out = {}

    frustums = [
        ("left",  dual_camera.V_left,  dual_camera.full_proj_transform_left,
         dual_camera.camera_center_left,  dual_camera.A_x_left,  dual_camera.A_y_left),
        ("right", dual_camera.V_right, dual_camera.full_proj_transform_right,
         dual_camera.camera_center_right, dual_camera.A_x_right, dual_camera.A_y_right),
    ]

    for side, wvt, fpt, campos, A_x, A_y in frustums:
        # Per-panel mask filtering: only render Gaussians belonging to this panel.
        # The overlap zone (|X| < 1mm) renders to both panels.
        panel_mask = getattr(pc, f'_panel_{side}_mask', None)
        if panel_mask is not None and len(panel_mask) == means3D.shape[0]:
            n_before = means3D.shape[0]
            masked_means3D = means3D[panel_mask]
            masked_scales = scales[panel_mask] if scales is not None else None
            masked_rotations = rotations[panel_mask] if rotations is not None else None
            masked_opacity = opacity[panel_mask]
            if shs is not None:
                masked_shs = shs[panel_mask]
            else:
                masked_shs = None
            if colors_precomp is not None:
                masked_colors = colors_precomp[panel_mask]
            else:
                masked_colors = None
            if separate_sh and dc is not None:
                masked_dc = dc[panel_mask]
            else:
                masked_dc = None
            masked_cov3D = cov3D_precomp[panel_mask] if cov3D_precomp is not None else None
            # Use original if no mask
            _means3D = masked_means3D
            _scales = masked_scales
            _rotations = masked_rotations
            _opacity = masked_opacity
            _shs = masked_shs
            _colors_precomp = masked_colors
            _dc = masked_dc
            _cov3D_precomp = masked_cov3D
            n_after = _means3D.shape[0]
            print(f'    Panel mask ({side}): {n_after}/{n_before} gaussians rendered')
        else:
            _means3D = means3D
            _scales = scales
            _rotations = rotations
            _opacity = opacity
            _shs = shs
            _colors_precomp = colors_precomp
            _dc = dc if separate_sh else None
            _cov3D_precomp = cov3D_precomp

        # Pseudo-tanfov per frustum: tanfovx = 1/A_x preserves Jacobian correctness
        pseudo_tanfovx = 1.0 / A_x if A_x > 0 else 0.0
        pseudo_tanfovy = 1.0 / A_y if A_y > 0 else 0.0
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype,
                                              requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        raster_settings = GaussianRasterizationSettings(
            image_height=int(dual_camera.image_height),
            image_width=int(dual_camera.image_width),
            tanfovx=pseudo_tanfovx,
            tanfovy=pseudo_tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=wvt,
            projmatrix=fpt,
            sh_degree=pc.active_sh_degree,
            campos=campos,
            prefiltered=False,
            debug=pipe.debug,
            antialiasing=pipe.antialiasing,
            use_asymmetric=True
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if separate_sh:
            rendered_image, radii, depth_image = rasterizer(
                means3D=_means3D,
                means2D=screenspace_points,
                dc=_dc,
                shs=_shs,
                colors_precomp=_colors_precomp,
                opacities=_opacity,
                scales=_scales,
                rotations=_rotations,
                cov3D_precomp=_cov3D_precomp)
        else:
            rendered_image, radii, depth_image = rasterizer(
                means3D=_means3D,
                means2D=screenspace_points,
                shs=_shs,
                colors_precomp=_colors_precomp,
                opacities=_opacity,
                scales=_scales,
                rotations=_rotations,
                cov3D_precomp=_cov3D_precomp)

        if use_trained_exp:
            exposure = pc.get_exposure_from_name(getattr(dual_camera, 'image_name', 'default'))
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0),
                                          exposure[:3, :3]).permute(2, 0, 1) + \
                             exposure[:3, 3, None, None]

        rendered_image = rendered_image.clamp(0, 1)

        # ═══════════════════════════════════════════════════════════════
        # [DIAGNOSTIC PROBE B] Roof-Region 2D Radii
        # [DIAGNOSTIC PROBE C] Roof-Region Camera-Space Z Span
        roof_attr = getattr(pc, '_is_super_sampled', None)
        if roof_attr is not None and len(roof_attr) == radii.shape[0]:
            roof_mask = roof_attr

            # Probe B: 2D radii for roof Gaussians
            roof_radii = radii[roof_mask]
            roof_radii_pos = roof_radii[roof_radii > 0]
            if roof_radii_pos.numel() > 0:
                r_mean = roof_radii_pos.float().mean().item()
                r_med = roof_radii_pos.float().median().item()
            else:
                r_mean = r_med = 0.0
            print(f"Probe B - Roof 2D Radii ({side}) (pixels): "
                  f"Mean={r_mean:.1f}, Median={r_med:.1f}, Max={roof_radii.max().item()}")

            # Probe C: camera-space Z for roof Gaussians
            xyz_roof = means3D[roof_mask]  # (N_roof, 3)
            ones = torch.ones(xyz_roof.shape[0], 1, device=xyz_roof.device, dtype=xyz_roof.dtype)
            xyz_h = torch.cat([xyz_roof, ones], dim=1)  # (N_roof, 4)
            xyz_cam = (wvt @ xyz_h.T).T  # (N_roof, 4)
            z_cam = xyz_cam[:, 2]
            print(f"Probe C - Roof Camera-Space Z ({side}): "
                  f"Min={z_cam.min().item():.4f}, Max={z_cam.max().item():.4f}, "
                  f"Range(Max-Min)={z_cam.max().item() - z_cam.min().item():.4f}")
        else:
            print(f"Probe B - Roof 2D Radii ({side}): NO ROOF MASK (model._is_super_sampled not set)")
            print(f"Probe C - Roof Camera-Space Z ({side}): NO ROOF MASK")
        # ═══════════════════════════════════════════════════════════════

        out[f"render_{side}"] = rendered_image
        out[f"viewspace_points_{side}"] = screenspace_points
        out[f"visibility_filter_{side}"] = (radii > 0).nonzero()
        out[f"radii_{side}"] = radii
        out[f"depth_{side}"] = depth_image

    return out
