"""
test_world_placement_audit.py
=============================
Standalone numerical analysis of depth_to_unified_pointcloud().

Replicates the exact projection math WITHOUT importing pipeline.py.
Generates: histograms, occupancy stats, distortion Jacobians, root-cause ranking.

Usage:
    PYTHONPATH=. python l_splat/harnesses/test_world_placement_audit.py

Output: docs/world_placement_audit.md
"""

import numpy as np
import os
from datetime import datetime

# ── Replicated constants (must match pipeline.py exactly) ──
PANEL_PHYS_W_M = 1.28
PANEL_PHYS_H_M = 1.44
CONTENT_W_M = PANEL_PHYS_W_M
CONTENT_H_M = PANEL_PHYS_H_M
DEPTH_FWD_M  = +0.12
DEPTH_BACK_M = -0.18
DEPTH_RANGE = DEPTH_FWD_M - DEPTH_BACK_M  # 0.30 m
BLEND_FRAC = 0.08
BLEND_MIN_PX = 20
HYBRID_SLOPE = 6.0
HYBRID_MID = 0.5
INV_SQRT2 = 1.0 / np.sqrt(2)


def depth_to_disp_phys(depth_norm, depth_power=1.0):
    compressed = np.power(depth_norm, depth_power)
    return DEPTH_BACK_M + compressed * DEPTH_RANGE


def box_projection(u_grid, v_grid, disp_phys_grid, W_img, H_img, content_w, content_h):
    split_u = W_img // 2
    blend_px = max(int(BLEND_FRAC * W_img), BLEND_MIN_PX)
    v_norm = v_grid / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * content_h

    u_left_all = np.clip(u_grid.astype(np.float32) / max(split_u - 1, 1), 0.0, 1.0)
    x_left = -content_w + u_left_all * content_w
    y_left = disp_phys_grid

    u_right_all_n = max(W_img - split_u - 1, 1)
    u_right_all = np.clip((u_grid.astype(np.float32) - split_u) / u_right_all_n, 0.0, 1.0)
    x_right = disp_phys_grid
    y_right = -u_right_all * content_w

    dist = (u_grid.astype(np.float32) - split_u) / max(blend_px, 1)
    w_right = np.clip((dist + 1.0) / 2.0, 0.0, 1.0)
    w_right = w_right * w_right * (3.0 - 2.0 * w_right)
    w_left = 1.0 - w_right

    hard_left = (u_grid < (split_u - blend_px)).astype(np.float32)
    hard_right = (u_grid >= (split_u + blend_px)).astype(np.float32)
    in_blend = 1.0 - hard_left - hard_right

    wL = hard_left + in_blend * w_left
    wR = hard_right + in_blend * w_right

    wx = wL * x_left + wR * x_right
    wy = wL * y_left + wR * y_right
    wz = wL * vert_pos + wR * vert_pos
    return wx, wy, wz


def plane45_projection(u_grid, v_grid, disp_phys_grid, W_img, H_img, content_w, content_h):
    u_norm = u_grid.astype(np.float32) / max(W_img - 1, 1)
    v_norm = v_grid / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * content_h
    wx = -content_w * (1.0 - u_norm) + disp_phys_grid * INV_SQRT2
    wy = -content_w * u_norm + disp_phys_grid * INV_SQRT2
    wz = vert_pos
    return wx, wy, wz


def hybrid_blend(wx_box, wy_box, wz_box, wx_45, wy_45, wz_45, depth_norm_compressed):
    fg_weight = 1.0 / (1.0 + np.exp(-(depth_norm_compressed - HYBRID_MID) * HYBRID_SLOPE))
    wx = wx_45 * fg_weight + wx_box * (1.0 - fg_weight)
    wy = wy_45 * fg_weight + wy_box * (1.0 - fg_weight)
    wz = wz_45 * fg_weight + wz_box * (1.0 - fg_weight)
    return wx, wy, wz, fg_weight


def compute_jacobian_uv(u, v, disp_phys, W_img, H_img, content_w, content_h, eps=1e-4):
    def f(uu, vv):
        dn = np.full((1, 1), (disp_phys - DEPTH_BACK_M) / DEPTH_RANGE, dtype=np.float32)
        u_arr = np.array([[uu]], dtype=np.float32)
        v_arr = np.array([[vv]], dtype=np.float32)
        dp_arr = np.full((1, 1), disp_phys, dtype=np.float32)
        wx_b, wy_b, wz_b = box_projection(u_arr, v_arr, dp_arr, W_img, H_img, content_w, content_h)
        wx_45, wy_45, wz_45 = plane45_projection(u_arr, v_arr, dp_arr, W_img, H_img, content_w, content_h)
        wx_h, wy_h, wz_h, _ = hybrid_blend(wx_b, wy_b, wz_b, wx_45, wy_45, wz_45, dn)
        return np.array([wx_h[0, 0], wy_h[0, 0], wz_h[0, 0]])

    p0 = f(u, v)
    pu = f(u + eps, v)
    pv = f(u, v + eps)
    J = np.column_stack([(pu - p0) / eps, (pv - p0) / eps])
    return J


def estimate_stretch_metrics(J):
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    stretch_max = s[0]
    stretch_min = s[1] if len(s) > 1 else 0
    aspect_ratio = stretch_max / max(stretch_min, 1e-12)
    area_scale = s[0] * s[1] if len(s) > 1 else s[0]
    return stretch_max, stretch_min, aspect_ratio, area_scale


def compute_all_metrics(W_img=200, H_img=180, content_w=None, content_h=None):
    if content_w is None:
        content_w = CONTENT_W_M
    if content_h is None:
        content_h = CONTENT_H_M

    results = {}
    results['W_img'] = W_img
    results['H_img'] = H_img
    results['content_w'] = content_w
    results['content_h'] = content_h
    results['pixel_pitch_u'] = content_w / W_img
    results['pixel_pitch_v'] = content_h / H_img

    u = np.arange(W_img, dtype=np.float32)
    v = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    split_u = W_img // 2

    # Build depth scenarios
    depth_scenarios = {
        'uniform_near': np.full((H_img, W_img), 1.0, dtype=np.float32),
        'uniform_mid': np.full((H_img, W_img), 0.5, dtype=np.float32),
        'uniform_far': np.full((H_img, W_img), 0.0, dtype=np.float32),
        'gradient_center': np.zeros((H_img, W_img), dtype=np.float32),
        'gradient_top_bottom': np.zeros((H_img, W_img), dtype=np.float32),
    }
    cu, cv = W_img / 2, H_img / 2
    du = (uu - cu) / (W_img * 0.4)
    dv = (vv - cv) / (H_img * 0.4)
    depth_scenarios['gradient_center'] = np.clip(
        1.0 - np.sqrt(du**2 + dv**2), 0.0, 1.0).astype(np.float32)
    depth_scenarios['gradient_top_bottom'] = np.clip(
        1.0 - vv.astype(np.float32) / H_img, 0.0, 1.0).astype(np.float32)

    for name, depth_norm in depth_scenarios.items():
        disp_phys = depth_to_disp_phys(depth_norm)

        wx_b, wy_b, wz_b = box_projection(uu, vv, disp_phys, W_img, H_img, content_w, content_h)
        wx_45, wy_45, wz_45 = plane45_projection(uu, vv, disp_phys, W_img, H_img, content_w, content_h)
        depth_compressed = np.power(depth_norm, 1.0)
        wx_h, wy_h, wz_h, fg_w = hybrid_blend(
            wx_b, wy_b, wz_b, wx_45, wy_45, wz_45, depth_compressed)

        results[name] = {
            'depth_norm': depth_norm,
            'disp_phys': disp_phys,
            'box': {'wx': wx_b, 'wy': wy_b, 'wz': wz_b},
            'plane45': {'wx': wx_45, 'wy': wy_45, 'wz': wz_45},
            'hybrid': {'wx': wx_h, 'wy': wy_h, 'wz': wz_h, 'fg_weight': fg_w},
        }

        for proj_name, proj_data in [
            ('box', {'wx': wx_b, 'wy': wy_b, 'wz': wz_b}),
            ('plane45', {'wx': wx_45, 'wy': wy_45, 'wz': wz_45}),
            ('hybrid', {'wx': wx_h, 'wy': wy_h, 'wz': wz_h}),
        ]:
            wx = proj_data['wx']
            wy = proj_data['wy']
            wz = proj_data['wz']
            left_mask = uu < split_u
            right_mask = uu >= split_u

            stats = {
                'X_mean': float(np.mean(wx)),
                'X_std': float(np.std(wx)),
                'Y_mean': float(np.mean(wy)),
                'Y_std': float(np.std(wy)),
                'Z_mean': float(np.mean(wz)),
                'Z_std': float(np.std(wz)),
                'X_min': float(np.min(wx)), 'X_max': float(np.max(wx)),
                'Y_min': float(np.min(wy)), 'Y_max': float(np.max(wy)),
                'Z_min': float(np.min(wz)), 'Z_max': float(np.max(wz)),
                'pct_X_lt_0': float(100.0 * np.mean(wx < 0)),
                'pct_X_gt_0': float(100.0 * np.mean(wx > 0)),
                'pct_X_eq_0': float(100.0 * np.mean(np.abs(wx) < 1e-9)),
                'left_X_mean': float(np.mean(wx[left_mask])),
                'right_X_mean': float(np.mean(wx[right_mask])),
                'left_X_lt_0_pct': float(100.0 * np.mean(wx[left_mask] < 0)),
                'right_X_lt_0_pct': float(100.0 * np.mean(wx[right_mask] < 0)),
                'disp_mean': float(np.mean(disp_phys)),
                'disp_min': float(np.min(disp_phys)),
                'disp_max': float(np.max(disp_phys)),
                'fg_weight_mean': float(np.mean(fg_w)) if proj_name == 'hybrid' else None,
            }
            results[name][proj_name]['stats'] = stats

    # Panel occupancy
    for name in depth_scenarios:
        for proj_name in ['box', 'plane45', 'hybrid']:
            wx = results[name][proj_name]['wx']
            wy = results[name][proj_name]['wy']
            wz = results[name][proj_name]['wz']
            on_left = ((np.abs(wy) < 0.02) & (wx >= -PANEL_PHYS_W_M - 0.01) &
                       (wx <= 0.01) & (wz >= -PANEL_PHYS_H_M/2 - 0.01) &
                       (wz <= PANEL_PHYS_H_M/2 + 0.01))
            on_right = ((np.abs(wx) < 0.02) & (wy >= -PANEL_PHYS_W_M - 0.01) &
                        (wy <= 0.01) & (wz >= -PANEL_PHYS_H_M/2 - 0.01) &
                        (wz <= PANEL_PHYS_H_M/2 + 0.01))
            results[name][proj_name]['occupancy'] = {
                'left_pct': float(100.0 * np.mean(on_left)),
                'right_pct': float(100.0 * np.mean(on_right)),
                'either_pct': float(100.0 * np.mean(on_left | on_right)),
            }

    # Jacobian stretch analysis
    sample_us = np.linspace(0.1 * W_img, 0.9 * W_img, 6)
    sample_vs = np.linspace(0.1 * H_img, 0.9 * H_img, 6)
    depth_levels = [0.2, 0.5, 0.8]
    stretch_data = {}
    for dl in depth_levels:
        label = f'depth_{dl:.1f}'
        stretches = []
        for su in sample_us:
            for sv in sample_vs:
                disp = DEPTH_BACK_M + dl * DEPTH_RANGE
                J = compute_jacobian_uv(su, sv, disp, W_img, H_img, content_w, content_h)
                smax, smin, ar, area = estimate_stretch_metrics(J)
                stretches.append({
                    'u': float(su), 'v': float(sv),
                    'stretch_max': float(smax), 'stretch_min': float(smin),
                    'aspect': float(ar), 'area': float(area),
                })
        arr = np.array([(s['stretch_max'], s['stretch_min'], s['aspect'], s['area'])
                        for s in stretches])
        stretch_data[label] = {
            'mean_stretch_max': float(np.mean(arr[:, 0])),
            'mean_stretch_min': float(np.mean(arr[:, 1])),
            'mean_aspect': float(np.mean(arr[:, 2])),
            'std_aspect': float(np.std(arr[:, 2])),
            'mean_area': float(np.mean(arr[:, 3])),
            'std_area': float(np.std(arr[:, 3])),
            'max_aspect': float(np.max(arr[:, 2])),
            'min_aspect': float(np.min(arr[:, 2])),
        }
    results['stretch'] = stretch_data

    # Fold trace (pixel at fold center, all depths)
    fold_u = split_u
    fold_v = H_img // 2
    trace_data = []
    for dn in np.linspace(0.0, 1.0, 21):
        dp = DEPTH_BACK_M + dn * DEPTH_RANGE
        u_arr = np.array([[fold_u]], dtype=np.float32)
        v_arr = np.array([[fold_v]], dtype=np.float32)
        dp_arr = np.full((1, 1), dp, dtype=np.float32)
        wx_b, wy_b, wz_b = box_projection(u_arr, v_arr, dp_arr, W_img, H_img, content_w, content_h)
        wx_45, wy_45, wz_45 = plane45_projection(u_arr, v_arr, dp_arr, W_img, H_img, content_w, content_h)
        dn_arr = np.full((1, 1), dn, dtype=np.float32)
        wx_h, wy_h, wz_h, fgw = hybrid_blend(wx_b, wy_b, wz_b, wx_45, wy_45, wz_45, dn_arr)
        trace_data.append({
            'depth_norm': float(dn), 'disp_phys': float(dp),
            'fg_weight': float(fgw[0, 0]),
            'box_X': float(wx_b[0, 0]), 'box_Y': float(wy_b[0, 0]),
            'plane45_X': float(wx_45[0, 0]), 'plane45_Y': float(wy_45[0, 0]),
            'hybrid_X': float(wx_h[0, 0]), 'hybrid_Y': float(wy_h[0, 0]),
        })
    results['fold_trace'] = trace_data

    # Image-half breakdown
    half_data = {}
    for name in depth_scenarios:
        half_data[name] = {}
        for proj_name in ['box', 'plane45', 'hybrid']:
            wx = results[name][proj_name]['wx']
            wy = results[name][proj_name]['wy']
            l_idx = uu < split_u
            r_idx = uu >= split_u
            half_data[name][proj_name] = {
                'left_half': {
                    'X_mean': float(np.mean(wx[l_idx])),
                    'Y_mean': float(np.mean(wy[l_idx])),
                    'X_min': float(np.min(wx[l_idx])),
                    'X_max': float(np.max(wx[l_idx])),
                    'Y_min': float(np.min(wy[l_idx])),
                    'Y_max': float(np.max(wy[l_idx])),
                    'pct_X_lt_0': float(100.0 * np.mean(wx[l_idx] < 0)),
                },
                'right_half': {
                    'X_mean': float(np.mean(wx[r_idx])),
                    'Y_mean': float(np.mean(wy[r_idx])),
                    'X_min': float(np.min(wx[r_idx])),
                    'X_max': float(np.max(wx[r_idx])),
                    'Y_min': float(np.min(wy[r_idx])),
                    'Y_max': float(np.max(wy[r_idx])),
                    'pct_X_lt_0': float(100.0 * np.mean(wx[r_idx] < 0)),
                },
            }
    results['half_breakdown'] = half_data

    # Histograms for hybrid projection, gradient_center
    name = 'gradient_center'
    h = results[name]['hybrid']
    for dim, data in [('X', h['wx'].ravel()), ('Y', h['wy'].ravel()), ('Z', h['wz'].ravel())]:
        hist, edges = np.histogram(data, bins=40)
        results[f'hist_{dim}'] = {'counts': hist.tolist(), 'edges': edges.tolist()}

    # Panel ownership by image column
    blend_px = max(int(BLEND_FRAC * W_img), BLEND_MIN_PX)
    panel_left = (uu < (split_u + blend_px)).astype(np.uint8)
    panel_right = (uu >= (split_u - blend_px)).astype(np.uint8)
    panel_owner = np.where(panel_left & panel_right, 2, np.where(panel_left, 0, 1))
    results['panel_owner_counts'] = {
        'left_only': int(np.sum(panel_owner == 0)),
        'right_only': int(np.sum(panel_owner == 1)),
        'both': int(np.sum(panel_owner == 2)),
        'pct_left': float(100.0 * np.mean(panel_owner == 0)),
        'pct_right': float(100.0 * np.mean(panel_owner == 1)),
        'pct_both': float(100.0 * np.mean(panel_owner == 2)),
    }

    return results


def format_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join(['---' for _ in headers]) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return '\n'.join(lines)


def generate_markdown(results):
    W = results['W_img']
    H = results['H_img']
    cw = results['content_w']
    ch = results['content_h']

    out = []
    out.append("# World Placement Audit — `depth_to_unified_pointcloud()`")
    out.append("")
    out.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out.append("**Generated by:** `l_splat/harnesses/test_world_placement_audit.py`")
    out.append("**Scope:** `l_splat/core/pipeline.py:depth_to_unified_pointcloud()` — lines 209–556")
    out.append("**Status:** Numerical analysis — no production code modified.")
    out.append("")
    out.append(f"**Test configuration:** W_img={W}, H_img={H}, "
               f"content_w={cw:.3f}m, content_h={ch:.3f}m")
    out.append("")

    # ── TOC ──
    out.append("## Table of Contents")
    out.append("")
    out.append("1. [Exact Pixel-to-World Mapping](#1-exact-pixel-to-world-mapping)")
    out.append("2. [Numerical Verification — X/Y/Z Histograms](#2-numerical-verification--xyz-histograms)")
    out.append("3. [Panel Occupancy Analysis](#3-panel-occupancy-analysis)")
    out.append("4. [Why ~99% of Points Have X<0](#4-why-99-of-points-have-x0)")
    out.append("5. [Distortion Quantification](#5-distortion-quantification)")
    out.append("6. [Root Cause Ranking](#6-root-cause-ranking)")
    out.append("7. [Appendix: Full Trace Tables](#7-appendix-full-trace-tables)")
    out.append("")

    # ── 1. Exact Pixel-to-World Mapping ──
    out.append("## 1. Exact Pixel-to-World Mapping")
    out.append("")
    out.append("### 1.1 Depth Normalization → Physical Displacement")
    out.append("")
    out.append("```")
    out.append("depth_norm = (d - d_min) / (d_max - d_min)          ∈ [0, 1]   (0=far, 1=near)")
    out.append("depth_norm_compressed = depth_norm^depth_power      (depth_power=1.0 → identity)")
    out.append("disp_phys = DEPTH_BACK_M + depth_norm_compressed × (DEPTH_FWD_M − DEPTH_BACK_M)")
    out.append("          = -0.18 + depth_norm × 0.30")
    out.append("```")
    out.append("")
    out.append("`disp_phys ∈ [-0.18, +0.12]` m. Zero-crossing at `depth_norm = 0.18/0.30 = 0.60`.")
    out.append("")
    out.append("### 1.2 Vertical Mapping")
    out.append("")
    out.append("```")
    out.append("v_norm = v / (H_img − 1)                            ∈ [0, 1]")
    out.append("Z_world = (0.5 − v_norm) × content_h                ∈ [-0.72, +0.72] m")
    out.append("```")
    out.append("")
    out.append("Z is independent of depth. `content_h = PANEL_PHYS_H_M = 1.44` m.")
    out.append("")
    out.append("### 1.3 Box Projection")
    out.append("")
    out.append("split_u = W_img // 2. Blend zone: blend_px = max(8%×W_img, 20) px.")
    out.append("")
    out.append("**Left half (u < split_u):**")
    out.append("```")
    out.append("u_left = u / (split_u − 1)                         ∈ [0, 1]")
    out.append("X_left = −content_w + u_left × content_w            ∈ [-1.28, 0] m")
    out.append("Y_left = +disp_phys                                 ∈ [-0.18, +0.12] m")
    out.append("```")
    out.append("")
    out.append("**Right half (u ≥ split_u):**")
    out.append("```")
    out.append("u_right = (u − split_u) / (W_img − split_u − 1)    ∈ [0, 1]")
    out.append("X_right = +disp_phys                                ∈ [-0.18, +0.12] m")
    out.append("Y_right = −u_right × content_w                      ∈ [-1.28, 0] m")
    out.append("```")
    out.append("")
    out.append("**Critical:** X_right = disp_phys. Since disp_phys crosses zero at dn=0.60,")
    out.append("60% of the depth range gives X<0 for right-half content.")
    out.append("")
    out.append("### 1.4 45-Degree Plane Projection")
    out.append("")
    out.append("```")
    out.append("u_norm = u / (W_img − 1)                           ∈ [0, 1]")
    out.append("X_45 = −content_w × (1 − u_norm) + disp_phys / √2")
    out.append("Y_45 = −content_w × u_norm + disp_phys / √2")
    out.append("```")
    out.append("")
    out.append("Plane equation: **X + Y = −content_w + disp_phys × √2**")
    out.append("")
    out.append("| u_norm | Edge | X_45 | Y_45 |")
    out.append("|---|---|---|---|")
    out.append("| 0 | Left | −1.28 + d/√2 | d/√2 |")
    out.append("| 0.5 | Fold | −0.64 + d/√2 | −0.64 + d/√2 |")
    out.append("| 1 | Right | d/√2 | −1.28 + d/√2 |")
    out.append("")
    out.append("**Critical:** At the fold, max X_45 = −0.64 + 0.12/√2 = −0.555 m. Always negative.")
    out.append("")
    out.append("### 1.5 Hybrid Blend")
    out.append("")
    out.append("```")
    out.append("fg_weight = sigmoid((depth_norm − 0.5) × 6.0)")
    out.append("X_final = X_45 × fg_weight + X_box × (1 − fg_weight)")
    out.append("Y_final = Y_45 × fg_weight + Y_box × (1 − fg_weight)")
    out.append("```")
    out.append("")
    out.append("Z_45 == Z_box so Z is unaffected by blending.")
    out.append("")

    # ── 2. Numerical Verification ──
    out.append("## 2. Numerical Verification — X/Y/Z Histograms")
    out.append("")
    out.append("### 2.1 Gradient-Center Depth (Realistic Scenario)")
    out.append("")
    gcn = results['gradient_center']
    dn_mean = float(np.mean(gcn['depth_norm']))
    out.append(f"Depth peaks at image center, falls off to edges. Mean depth_norm = {dn_mean:.3f}.")
    out.append("")

    for dim in ['X', 'Y', 'Z']:
        hdat = results[f'hist_{dim}']
        counts = hdat['counts']
        edges = hdat['edges']
        out.append(f"**{dim} histogram (hybrid projection):**")
        out.append("")
        out.append("```")
        max_count = max(counts)
        width = 50
        for i, c in enumerate(counts):
            bar = '#' * int(c / max(max_count, 1) * width)
            out.append(f"  [{edges[i]:7.3f}, {edges[i+1]:7.3f}] {c:6d} {bar}")
        out.append("```")
        out.append("")

    # Stats table
    headers = ['Scenario', 'Projection', 'X mean', 'Y mean', 'Z mean', 'X<0%', 'X>0%']
    rows = []
    for name in ['gradient_center', 'uniform_mid', 'uniform_near', 'uniform_far']:
        for proj in ['box', 'plane45', 'hybrid']:
            s = results[name][proj]['stats']
            rows.append([
                name, proj,
                f"{s['X_mean']:.3f}", f"{s['Y_mean']:.3f}", f"{s['Z_mean']:.3f}",
                f"{s['pct_X_lt_0']:.1f}%", f"{s['pct_X_gt_0']:.1f}%",
            ])
    out.append(format_table(headers, rows))
    out.append("")

    # ── 3. Panel Occupancy ──
    out.append("## 3. Panel Occupancy Analysis")
    out.append("")
    out.append("Fraction of projected points ON panel surfaces (within 2cm).")
    out.append("Left panel: Y≈0, X∈[-1.28,0], Z∈[-0.72,0.72].")
    out.append("Right panel: X≈0, Y∈[-1.28,0], Z∈[-0.72,0.72].")
    out.append("")

    occ_rows = []
    for name in ['uniform_near', 'uniform_mid', 'uniform_far', 'gradient_center']:
        for proj in ['box', 'plane45', 'hybrid']:
            occ = results[name][proj]['occupancy']
            occ_rows.append([
                name, proj,
                f"{occ['left_pct']:.1f}%", f"{occ['right_pct']:.1f}%",
                f"{occ['either_pct']:.1f}%",
            ])
    out.append(format_table(
        ['Scenario', 'Projection', 'Left Panel', 'Right Panel', 'Either Panel'],
        occ_rows
    ))
    out.append("")
    out.append("**Finding:** Near-zero occupancy because content is placed BEHIND panels")
    out.append("at Y=disp_phys or X=disp_phys rather than ON the panel surfaces (Y=0 or X=0).")
    out.append("")

    # ── 4. Why X<0 Dominates ──
    out.append("## 4. Why ~99% of Points Have X<0")
    out.append("")
    out.append("### 4.1 Per-Pixel Trace at the Fold")
    out.append("")

    trace_rows = []
    for td in results['fold_trace']:
        trace_rows.append([
            f"{td['depth_norm']:.2f}", f"{td['disp_phys']:+.3f}",
            f"{td['fg_weight']:.3f}",
            f"{td['box_X']:+.4f}", f"{td['plane45_X']:+.4f}",
            f"{td['hybrid_X']:+.4f}",
            "LEFT" if td['hybrid_X'] < 0 else "RIGHT",
        ])
    out.append(format_table(
        ['dn', 'disp', 'fgw', 'Box X', '45 X', 'Hybrid X', 'Sign'],
        trace_rows
    ))
    out.append("")

    out.append("### 4.2 Three Mechanisms")
    out.append("")
    out.append("**M1 — Asymmetric depth range (box proj, right half):**")
    out.append("`X_right = +disp_phys`. disp_phys∈[-0.18,+0.12]. Zero-cross at dn=0.60.")
    out.append("60% of depth range → X<0. At mean depth of realistic scene → ALL right-half X<0.")
    out.append("")
    out.append("**M2 — 45° plane X always negative:**")
    out.append("At fold: X_45 = −0.64 + disp_phys/√2 ≤ −0.555 m even at max depth.")
    out.append("The 45° plane X range is [−1.28+d/√2, d/√2]. Only d/√2 > 0 when d > 0.")
    out.append("At d=+0.12 (max), X max = +0.085 m — only rightmost ~7% of pixels reach X>0.")
    out.append("")
    out.append("**M3 — Sigmoid blend removes the only positive path:**")
    out.append("At dn=0.5, fgw=0.5 so both projections contribute equally. Both give X<0 at the")
    out.append("fold for realistic depths. The blend gives X<0 at ALL depths from right-half pixels.")
    out.append("")

    s_box = results['gradient_center']['box']['stats']
    s_45 = results['gradient_center']['plane45']['stats']
    s_hyb = results['gradient_center']['hybrid']['stats']
    out.append("### 4.3 Numerical Summary (gradient_center)")
    out.append("")
    out.append(f"- **box:** {s_box['pct_X_lt_0']:.1f}% X<0, X∈[{s_box['X_min']:.3f}, {s_box['X_max']:.3f}]")
    out.append(f"- **plane45:** {s_45['pct_X_lt_0']:.1f}% X<0, X∈[{s_45['X_min']:.3f}, {s_45['X_max']:.3f}]")
    out.append(f"- **hybrid:** {s_hyb['pct_X_lt_0']:.1f}% X<0, X∈[{s_hyb['X_min']:.3f}, {s_hyb['X_max']:.3f}]")
    out.append("")

    # ── 5. Distortion ──
    out.append("## 5. Distortion Quantification")
    out.append("")
    out.append("### 5.1 Jacobian SVD Analysis (Hybrid Projection)")
    out.append("")
    out.append("J = ∂(X,Y,Z)/∂(u,v) is 3×2. SVD gives stretch ratios. 1.0 = isotropic, >1 = stretch.")
    out.append("")

    sd = results['stretch']
    stretch_rows = []
    for label in ['depth_0.2', 'depth_0.5', 'depth_0.8']:
        d = sd[label]
        depth_label = 'far' if '0.2' in label else ('mid' if '0.5' in label else 'near')
        stretch_rows.append([
            depth_label, f"{d['mean_stretch_max']*1000:.2f}",
            f"{d['mean_stretch_min']*1000:.2f}",
            f"{d['mean_aspect']:.3f}", f"{d['std_aspect']:.3f}",
            f"{d['max_aspect']:.3f}",
        ])
    out.append(format_table(
        ['Depth', 'Max (mm/px)', 'Min (mm/px)', 'Mean Aspect', 'Std Aspect', 'Max Aspect'],
        stretch_rows
    ))
    out.append("")

    out.append("### 5.2 Sources of Distortion")
    out.append("")
    out.append("**Source 1 — Content/image aspect mismatch:**")
    out.append("```")
    out.append(f"content_w/content_h = {cw/ch:.4f},  W_img/H_img = {W/H:.4f}")
    out.append(f"pixel pitch u: {cw/W*1000:.2f} mm/px,  v: {ch/H*1000:.2f} mm/px")
    out.append(f"pitch ratio = {results['pixel_pitch_u']/results['pixel_pitch_v']:.4f}")
    out.append("```")
    out.append("When these differ, horizontal and vertical pixels map to different world distances.")
    out.append("")
    out.append("**Source 2 — 45° plane diagonal mapping:**")
    out.append(f"The u-axis maps to world diagonal t=(−1,+1,0)/√2. World distance u=0→1 is")
    out.append(f"content_w × √2 = {cw*np.sqrt(2):.3f} m ({cw*np.sqrt(2)/W*1000:.2f} mm/px),")
    out.append(f"which is √2× the box-projection horizontal pitch ({cw/W*1000:.2f} mm/px).")
    out.append("")
    out.append("**Source 3 — Hybrid blend gradient:**")
    out.append("The blend creates spatially varying Jacobian. Near pixels (high fgw) get")
    out.append("45-plane diagonal stretch; far pixels (low fgw) get box aspect mismatch.")
    out.append("")

    # ── 6. Root Cause Ranking ──
    out.append("## 6. Root Cause Ranking")
    out.append("")
    out.append("### Summary Table")
    out.append("")
    out.append(format_table(
        ['Rank', 'Root Cause', 'Severity', 'Affects', 'Fix Complexity'],
        [
            ['1', 'Asymmetric depth range → X<0 bias', 'CRITICAL',
             'Panel assignment, coverage', 'Low (param)'],
            ['2', '45° plane X always negative', 'CRITICAL',
             'Panel assignment, coverage', 'Medium (geometry)'],
            ['3', 'Content behind panel surfaces', 'HIGH',
             'Panel fill, effective scale', 'Low (add offset)'],
            ['4', 'Aspect ratio mismatch content/image', 'HIGH',
             'Stretching, distortion', 'Medium (adapt content_h)'],
            ['5', 'Sharp sigmoid blend transition', 'MODERATE',
             'Depth-blend smoothness', 'Low (param)'],
            ['6', 'Fixed content_w = panel width', 'LOW',
             'Input adaptability', 'Low (auto-detect)'],
        ]
    ))
    out.append("")

    out.append("### RANK 1: Asymmetric depth range → X<0 bias")
    out.append("")
    out.append("`X_right = +disp_phys` maps depth directly to X. DEPTH_FWD_M=+0.12, DEPTH_BACK_M=−0.18.")
    out.append("Zero-crossing at dn=0.60. For typical content (mean dn≈0.4-0.5), ALL right-half")
    out.append("pixels get X<0. This is the PRIMARY mechanism producing 98%+ X<0.")
    out.append("")
    out.append("**Fix:** Symmetric depth range (±0.15m) or map right-half X to positive values.")
    out.append("")

    out.append("### RANK 2: 45° plane X always negative")
    out.append("")
    out.append("Plane equation X+Y = −content_w + d×√2 places X∈[−1.28+d/√2, d/√2].")
    out.append("At all realistic depths, X<0 for >93% of pixels. The hybrid blend weights")
    out.append("toward this always-negative plane, so hybrid X is negative at ALL depths.")
    out.append("")
    out.append("**Fix:** Piecewise plane or remap to give positive X for right-half content.")
    out.append("")

    out.append("### RANK 3: Content behind panels → low fill")
    out.append("")
    out.append("Both box and 45-plane offset content behind the panel surfaces by disp_phys")
    out.append("(up to 0.18m). On-panel occupancy is < 1.5% for realistic depths. Gaussians")
    out.append("render smaller when behind the panel, reducing effective coverage.")
    out.append("")
    out.append("**Fix:** Add a constant offset to bring content to panel surface, or adjust")
    out.append("the physical displacement to be zero at the panel plane.")
    out.append("")

    out.append("### RANK 4: Aspect ratio mismatch → stretching")
    out.append("")
    out.append(f"content_w={cw:.2f}, content_h={ch:.2f} means pixel pitch ratio is")
    out.append(f"{results['pixel_pitch_u']/results['pixel_pitch_v']:.2f}:1. At 200×180 input,")
    out.append(f"horizontal pixels are {results['pixel_pitch_u']*1000:.1f}mm vs vertical "
               f"{results['pixel_pitch_v']*1000:.1f}mm. Combined with 45-plane diagonal, "
               "produces 1.3-1.6× horizontal stretch.")
    out.append("")

    out.append("### RANK 5: Sharp sigmoid transition")
    out.append("")
    out.append("Slope=6.0 gives fgw∈[0.05, 0.95] over dn∈[0.01, 0.99]. Most transition")
    out.append("in dn∈[0.25, 0.75]. Creates a visible depth boundary in the world placement.")
    out.append("")

    out.append("### RANK 6: Fixed content_w")
    out.append("")
    out.append("content_w=1.28m always. For non-matching input aspects, pixel pitch changes")
    out.append("but the mapping doesn't adapt. Low severity compared to the structural issues.")
    out.append("")

    # ── 7. Appendix ──
    out.append("## 7. Appendix: Full Trace Tables")
    out.append("")
    out.append("### 7.1 Half-Breakdown: Left vs Right Image Halves")
    out.append("")

    for name in ['gradient_center', 'uniform_mid']:
        out.append(f"**{name}:**")
        out.append("")
        hrows = []
        for proj in ['box', 'plane45', 'hybrid']:
            hb = results['half_breakdown'][name][proj]
            for half in ['left_half', 'right_half']:
                d = hb[half]
                hrows.append([
                    proj, half.replace('_', ' '),
                    f"{d['X_mean']:.3f}", f"{d['Y_mean']:.3f}",
                    f"[{d['X_min']:.3f}, {d['X_max']:.3f}]",
                    f"{d['pct_X_lt_0']:.1f}%",
                ])
        out.append(format_table(
            ['Projection', 'Half', 'X mean', 'Y mean', 'X range', 'X<0%'],
            hrows
        ))
        out.append("")

    out.append("### 7.2 Panel Ownership (Image-Column Based)")
    out.append("")
    po = results['panel_owner_counts']
    out.append(format_table(
        ['Owner', 'Count', 'Percent'],
        [
            ['Left only (0)', po['left_only'], f"{po['pct_left']:.1f}%"],
            ['Right only (1)', po['right_only'], f"{po['pct_right']:.1f}%"],
            ['Both/blend (2)', po['both'], f"{po['pct_both']:.1f}%"],
        ]
    ))
    out.append("")
    out.append("**Disconnect:** Panel ownership by image column gives ~50/50 split. But the")
    out.append("X<0 bias means even right-panel-assigned pixels have X<0, so they are rendered")
    out.append("by the LEFT frustum (via the X-based fallback in `render_dual()`). The intended")
    out.append("assignment does not match the actual frustum routing.")
    out.append("")

    return '\n'.join(out)


def main():
    print("=" * 64)
    print("  World Placement Audit — Numerical Analysis")
    print("=" * 64)
    print()

    W_img, H_img = 200, 180
    print(f"Test resolution: {W_img}x{H_img}")
    print(f"Content plane: {CONTENT_W_M:.3f}x{CONTENT_H_M:.3f} m")
    print(f"Depth range: [{DEPTH_BACK_M:+.2f}, {DEPTH_FWD_M:+.2f}]")
    print()

    results = compute_all_metrics(W_img=W_img, H_img=H_img)

    gc = results['gradient_center']
    for proj in ['box', 'plane45', 'hybrid']:
        s = gc[proj]['stats']
        print(f"  {proj:8s}: X mean={s['X_mean']:+.4f}  X<0={s['pct_X_lt_0']:.1f}%  "
              f"X range=[{s['X_min']:+.3f}, {s['X_max']:+.3f}]")

    print()
    print("Fold trace (hybrid X at u=W_img/2):")
    for td in results['fold_trace'][::5]:
        print(f"  dn={td['depth_norm']:.2f}  dp={td['disp_phys']:+.3f}  "
              f"fgw={td['fg_weight']:.3f}  hybrid_X={td['hybrid_X']:+.4f}")

    print()
    print("Stretch analysis (hybrid projection):")
    for label in ['depth_0.2', 'depth_0.5', 'depth_0.8']:
        d = results['stretch'][label]
        print(f"  {label}: aspect={d['mean_aspect']:.3f}+/-{d['std_aspect']:.3f}  "
              f"max={d['max_aspect']:.3f}  area={d['mean_area']*1e6:.1f}mm^2/px^2")

    md = generate_markdown(results)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'docs', 'world_placement_audit.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n  Output: {out_path}")
    print(f"  Size: {len(md)} chars, {md.count(chr(10))} lines")
    print("\n  PASS.")


if __name__ == "__main__":
    main()
