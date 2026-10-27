"""
test_content_placement.py — Content placement audit for depth_to_unified_pointcloud().

Measures box projection, 45-degree plane, hybrid blend, X<0 vs X>0 occupancy,
panel coverage, world-space histograms, and distortion.

Pure numpy — no CUDA/DAV3. Replicates the exact math from pipeline.py.
"""

import numpy as np
import math

# ===========================================================================
# Configuration (mirrors pipeline.py)
# ===========================================================================
PANEL_W = 1.28
PANEL_H = 1.44
HALF_H = PANEL_H / 2.0
CONTENT_W = PANEL_W
CONTENT_H = PANEL_H
DEPTH_NEAR = +0.12
DEPTH_FAR = -0.18
INV_SQRT2 = 1.0 / math.sqrt(2)


def depth_phys_from_norm(depth_norm):
    """Linear mapping: 0=far, 1=near (depth_power=1.0)."""
    return DEPTH_FAR + depth_norm * (DEPTH_NEAR - DEPTH_FAR)


def depth_to_world_boxes(H_img, W_img, depth_norm, content_w, content_h):
    """Replicate box projection math from depth_to_unified_pointcloud()."""
    depth_phys = depth_phys_from_norm(depth_norm)

    uu, vv = np.meshgrid(np.arange(W_img, dtype=np.float32),
                         np.arange(H_img, dtype=np.float32))
    v_norm = vv / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * content_h

    split_u = W_img // 2
    blend_px = max(int(0.08 * W_img), 20)

    # Left plane: Y = depth
    u_left_all = np.clip(uu.astype(np.float32) / max(split_u - 1, 1), 0.0, 1.0)
    x_left = -content_w + u_left_all * content_w
    y_left = depth_phys

    # Right plane: X = depth
    u_right_all_n = max(W_img - split_u - 1, 1)
    u_right_all = np.clip((uu.astype(np.float32) - split_u) / u_right_all_n, 0.0, 1.0)
    x_right = depth_phys
    y_right = -u_right_all * content_w

    # Blend weights
    dist = (uu.astype(np.float32) - split_u) / max(blend_px, 1)
    w_right = np.clip((dist + 1.0) / 2.0, 0.0, 1.0)
    w_right = w_right * w_right * (3.0 - 2.0 * w_right)  # smoothstep
    w_left = 1.0 - w_right

    hard_left = (uu < (split_u - blend_px)).astype(np.float32)
    hard_right = (uu >= (split_u + blend_px)).astype(np.float32)
    in_blend = 1.0 - hard_left - hard_right

    wx = (hard_left + in_blend * w_left) * x_left + (hard_right + in_blend * w_right) * x_right
    wy = (hard_left + in_blend * w_left) * y_left + (hard_right + in_blend * w_right) * y_right
    wz = vert_pos

    return wx, wy, wz, split_u, blend_px


def depth_to_world_45deg(H_img, W_img, depth_norm, content_w, content_h):
    """Replicate 45-degree plane projection math."""
    depth_phys = depth_phys_from_norm(depth_norm)

    uu, vv = np.meshgrid(np.arange(W_img, dtype=np.float32),
                         np.arange(H_img, dtype=np.float32))
    v_norm = vv / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * content_h

    u_norm = uu.astype(np.float32) / max(W_img - 1, 1)
    wx = -content_w * (1.0 - u_norm) + depth_phys * INV_SQRT2
    wy = -content_w * u_norm + depth_phys * INV_SQRT2
    wz = vert_pos

    return wx, wy, wz


def depth_to_world_hybrid(H_img, W_img, depth_norm, content_w, content_h):
    """Replicate hybrid projection (box + 45-degree, depth-based blend)."""
    wxb, wyb, wzb, split_u, blend_px = depth_to_world_boxes(
        H_img, W_img, depth_norm, content_w, content_h)
    wx45, wy45, wz45 = depth_to_world_45deg(
        H_img, W_img, depth_norm, content_w, content_h)

    fg_weight = 1.0 / (1.0 + np.exp(-(depth_norm - 0.5) * 6.0))

    wx = wx45 * fg_weight + wxb * (1.0 - fg_weight)
    wy = wy45 * fg_weight + wyb * (1.0 - fg_weight)
    wz = wz45 * fg_weight + wzb * (1.0 - fg_weight)

    return wx, wy, wz, split_u, blend_px, fg_weight


def analyze_projection(H_img, W_img, depth_norm, label):
    """Run all three projections and measure key metrics."""
    total = H_img * W_img
    dp_phys_mean = depth_phys_from_norm(depth_norm).mean()

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  depth_norm: mean={depth_norm.mean():.3f}  ->  phys={dp_phys_mean:+.4f}m")
    print(f"{'='*70}")

    wxb, wyb, wzb, su, bp = depth_to_world_boxes(H_img, W_img, depth_norm, CONTENT_W, CONTENT_H)
    wx45, wy45, wz45 = depth_to_world_45deg(H_img, W_img, depth_norm, CONTENT_W, CONTENT_H)
    wxh, wyh, wzh, _, _, fgw = depth_to_world_hybrid(H_img, W_img, depth_norm, CONTENT_W, CONTENT_H)

    for name, wx, wy, wz in [("Box", wxb, wyb, wzb), ("45deg", wx45, wy45, wz45), ("Hybrid", wxh, wyh, wzh)]:
        x_neg_pct = (wx < -1e-6).sum() / total * 100
        x_pos_pct = (wx > +1e-6).sum() / total * 100
        x_zero_pct = (np.abs(wx) <= 1e-6).sum() / total * 100

        left_cov = ((wy >= -0.01) & (wy <= 0.01) & (wx >= -PANEL_W-0.01) & (wx <= 0.01)).sum() / total * 100
        right_cov = ((wx >= -0.01) & (wx <= 0.01) & (wy >= -PANEL_W-0.01) & (wy <= 0.01)).sum() / total * 100

        print(f"  {name:6s}: X<0={x_neg_pct:.1f}%  X>0={x_pos_pct:.1f}%  X=0={x_zero_pct:.1f}%  "
              f"Lcov={left_cov:.1f}%  Rcov={right_cov:.1f}%  "
              f"X[{wx.min():+.3f},{wx.max():+.3f}]  Y[{wy.min():+.3f},{wy.max():+.3f}]")

    # Per-column X mean
    for name, wx, wy in [("Box", wxb, wyb), ("Hybrid", wxh, wyh)]:
        col_x = wx.mean(axis=0)
        print(f"  {name} col-X: left_half={col_x[:su].mean():+.4f}  "
              f"right_half={col_x[su:].mean():+.4f}  "
              f"right_half_frac_pos={((col_x[su:] > 0).sum()/len(col_x[su:]))*100:.0f}%")

    print(f"  Hybrid fg_weight: mean={fgw.mean():.3f}  >0.5:{(fgw>0.5).sum()/total*100:.0f}%")
    return {"box": (wxb, wyb), "deg45": (wx45, wy45), "hybrid": (wxh, wyh), "fgw": fgw, "split_u": su}


def measure_grid_distortion(H_img, W_img, depth_norm, label):
    """Place an image-space grid and measure world-space cell distortion."""
    wx, wy, wz, su, bp, fgw = depth_to_world_hybrid(
        H_img, W_img, depth_norm, CONTENT_W, CONTENT_H)

    grid_n = 7
    u_pts = np.linspace(0.1, 0.9, grid_n)
    v_pts = np.linspace(0.1, 0.9, grid_n)

    cell_aspects = []
    cell_areas = []

    for i in range(grid_n - 1):
        for j in range(grid_n - 1):
            ui, uj = int(u_pts[i] * (W_img - 1)), int(u_pts[i + 1] * (W_img - 1))
            vi, vj = int(v_pts[j] * (H_img - 1)), int(v_pts[j + 1] * (H_img - 1))
            c00 = np.array([wx[vi, ui], wy[vi, ui], wz[vi, ui]])
            c10 = np.array([wx[vi, uj], wy[vi, uj], wz[vi, uj]])
            c01 = np.array([wx[vj, ui], wy[vj, ui], wz[vj, ui]])
            c11 = np.array([wx[vj, uj], wy[vj, uj], wz[vj, uj]])
            h_w = (np.linalg.norm(c10 - c00) + np.linalg.norm(c11 - c01)) / 2.0
            v_w = (np.linalg.norm(c01 - c00) + np.linalg.norm(c11 - c10)) / 2.0
            aspect = h_w / max(v_w, 0.0001)
            cell_aspects.append(aspect)
            area = np.linalg.norm(np.cross(c10 - c00, c01 - c00))
            cell_areas.append(area)

    cell_aspects = np.array(cell_aspects)
    cell_areas = np.array(cell_areas)
    stretch = max(cell_aspects.max(), 1/cell_aspects.min()) - 1

    print(f"  [{label}] Grid {grid_n-1}x{grid_n-1}: aspect μ={cell_aspects.mean():.3f} "
          f"σ={cell_aspects.std():.3f} [{cell_aspects.min():.3f},{cell_aspects.max():.3f}]  "
          f"stretch={stretch*100:.1f}%  area_CV={cell_areas.std()/cell_areas.mean():.3f}  "
          f"area_ratio={cell_areas.max()/cell_areas.min():.2f}x")

    return cell_aspects, cell_areas


def harness():
    print("=" * 70)
    print("  CONTENT PLACEMENT AUDIT — depth_to_unified_pointcloud()")
    print("=" * 70)
    print(f"  Panel: {PANEL_W}x{PANEL_H}m  Content: {CONTENT_W}x{CONTENT_H}m")
    print(f"  Depth: [{DEPTH_FAR:+.2f}, {DEPTH_NEAR:+.2f}]m  ({DEPTH_NEAR-DEPTH_FAR:.2f}m range)")
    H, W = 180, 200  # ~real video res

    # ── 1. PIXEL-TO-WORLD TRACE ──
    print(f"\n{'='*70}")
    print("  1. PIXEL → DEPTH → WORLD TRACE (5 depth values)")
    print("=" * 70)

    for dp, name in [(0.0,"FAR"), (0.3,"FAR-MID"), (0.5,"MID"), (0.7,"NEAR-MID"), (1.0,"NEAR")]:
        analyze_projection(H, W, np.full((H, W), dp, dtype=np.float32), f"{name} (dp={dp})")

    # Gradient
    uu, vv = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    dist = np.sqrt((uu - 0.5)**2 + (vv - 0.5)**2)
    depth_grad = np.clip(0.7 - 0.5 * (dist / 0.7), 0.0, 1.0).astype(np.float32)
    analyze_projection(H, W, depth_grad, "GRADIENT (center-near, edge-far)")

    # ── 2. ROOT CAUSE OF X<0 DOMINATION ──
    print(f"\n{'='*70}")
    print("  2. ROOT CAUSE: Why X<0 dominates")
    print("=" * 70)

    dp_phys_zero = -DEPTH_FAR / (DEPTH_NEAR - DEPTH_FAR)
    dp_phys_mid = depth_phys_from_norm(np.float32(0.5))
    print(f"  Box projection right-half: x_right = +disp_phys")
    print(f"  At depth_norm=0.5: disp_phys = {dp_phys_mid:+.4f}m (negative → X<0!)")
    print(f"  X crosses zero at depth_norm = {dp_phys_zero:.4f} ({dp_phys_zero*100:.0f}%)")
    print(f"  Depth range: [{DEPTH_FAR:+.2f}, 0.00]={abs(DEPTH_FAR):.2f}m → X<0 ({abs(DEPTH_FAR)/(DEPTH_NEAR-DEPTH_FAR)*100:.0f}% of range)")
    print(f"  Depth range: [0.00, {DEPTH_NEAR:+.2f}]={DEPTH_NEAR:.2f}m → X>0 ({DEPTH_NEAR/(DEPTH_NEAR-DEPTH_FAR)*100:.0f}% of range)")
    print(f"  → 60% of depth values map to X<0 even from right-half pixels")

    x45_mid = -CONTENT_W * 0.5 + dp_phys_mid * INV_SQRT2
    print(f"  45-plane at fold: X = -{CONTENT_W/2:.3f} + disp_phys/sqrt2 = {x45_mid:+.4f}m")
    print(f"  → Always negative for realistic depths")

    # ── 3. HYBRID BLEND EFFECT ──
    print(f"\n{'='*70}")
    print("  3. HYBRID BLEND: right-half pixel X at each depth")
    print("=" * 70)
    print(f"  {'depth_norm':>10s}  {'phys':>8s}  {'fgw':>6s}  {'box_X':>9s}  {'45_X':>9s}  {'hybrid_X':>10s}  sign")
    for dp in np.linspace(0, 1, 11):
        dp_phys = depth_phys_from_norm(dp)
        fgw = 1.0 / (1.0 + math.exp(-(dp - 0.5) * 6.0))
        x_box = dp_phys
        x_45 = -CONTENT_W * 0.5 + dp_phys * INV_SQRT2
        x_hyb = x_45 * fgw + x_box * (1.0 - fgw)
        print(f"  {dp:10.1f}  {dp_phys:+7.4f}  {fgw:6.3f}  {x_box:+8.4f}  {x_45:+8.4f}  {x_hyb:+9.4f}  "
              f"{'RIGHT' if x_hyb > 0 else 'LEFT'}")

    # ── 4. DISTORTION ──
    print(f"\n{'='*70}")
    print("  4. DISTORTION MEASUREMENTS (hybrid projection)")
    print("=" * 70)
    for dp, name in [(0.2,"FAR"), (0.5,"MID"), (0.8,"NEAR")]:
        measure_grid_distortion(H, W, np.full((H, W), dp, dtype=np.float32), name)

    # ── 5. SUMMARY ──
    print(f"\n{'='*70}")
    print("  AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    harness()
