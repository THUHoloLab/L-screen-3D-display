"""
test_corrected_placement.py
===========================
RED-phase harness: validates that the OLD placement model has X<0 bias,
and proves the NEW corrected box projection model fixes it.

Independent replication — does NOT import pipeline.py.

Usage:
    PYTHONPATH=. python l_splat/harnesses/test_corrected_placement.py

Exit 0 = NEW model passes, OLD model correctly identified as broken.
Exit 1 = unexpected failure.
"""

import numpy as np
import sys

# ── Constants ──
PANEL_PHYS_W_M = 1.28
PANEL_PHYS_H_M = 1.44
CONTENT_W_M = PANEL_PHYS_W_M
CONTENT_H_M = PANEL_PHYS_H_M
BLEND_FRAC = 0.08
BLEND_MIN_PX = 20

# OLD depth constants (asymmetric)
OLD_DEPTH_FWD  = +0.12
OLD_DEPTH_BACK = -0.18
OLD_DEPTH_RANGE = OLD_DEPTH_FWD - OLD_DEPTH_BACK  # 0.30

# NEW depth constants (symmetric)
NEW_DEPTH_FWD  = +0.15
NEW_DEPTH_BACK = -0.15
NEW_DEPTH_RANGE = NEW_DEPTH_FWD - NEW_DEPTH_BACK  # 0.30

INV_SQRT2 = 1.0 / np.sqrt(2)
HYBRID_SLOPE = 6.0
HYBRID_MID = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# OLD model — exact replication of current depth_to_unified_pointcloud()
# ══════════════════════════════════════════════════════════════════════════════

def old_depth_to_disp(depth_norm):
    compressed = np.power(depth_norm, 1.0)
    return OLD_DEPTH_BACK + compressed * OLD_DEPTH_RANGE


def old_box_projection(u_grid, v_grid, disp_phys_grid, W_img, H_img):
    split_u = W_img // 2
    blend_px = max(int(BLEND_FRAC * W_img), BLEND_MIN_PX)
    v_norm = v_grid / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * CONTENT_H_M

    u_left_all = np.clip(u_grid.astype(np.float32) / max(split_u - 1, 1), 0.0, 1.0)
    x_left = -CONTENT_W_M + u_left_all * CONTENT_W_M
    y_left = disp_phys_grid

    u_right_all_n = max(W_img - split_u - 1, 1)
    u_right_all = np.clip((u_grid.astype(np.float32) - split_u) / u_right_all_n, 0.0, 1.0)
    x_right = disp_phys_grid
    y_right = -u_right_all * CONTENT_W_M

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


def old_plane45_projection(u_grid, v_grid, disp_phys_grid, W_img, H_img):
    u_norm = u_grid.astype(np.float32) / max(W_img - 1, 1)
    v_norm = v_grid / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * CONTENT_H_M
    wx = -CONTENT_W_M * (1.0 - u_norm) + disp_phys_grid * INV_SQRT2
    wy = -CONTENT_W_M * u_norm + disp_phys_grid * INV_SQRT2
    wz = vert_pos
    return wx, wy, wz


def old_hybrid_blend(wx_box, wy_box, wz_box, wx_45, wy_45, wz_45, depth_compressed):
    fg_weight = 1.0 / (1.0 + np.exp(-(depth_compressed - HYBRID_MID) * HYBRID_SLOPE))
    wx = wx_45 * fg_weight + wx_box * (1.0 - fg_weight)
    wy = wy_45 * fg_weight + wy_box * (1.0 - fg_weight)
    wz = wz_45 * fg_weight + wz_box * (1.0 - fg_weight)
    return wx, wy, wz, fg_weight


def old_full_projection(depth_norm, u_grid, v_grid, W_img, H_img):
    disp_phys = old_depth_to_disp(depth_norm)
    wx_b, wy_b, wz_b = old_box_projection(u_grid, v_grid, disp_phys, W_img, H_img)
    wx_45, wy_45, wz_45 = old_plane45_projection(u_grid, v_grid, disp_phys, W_img, H_img)
    depth_compressed = np.power(depth_norm, 1.0)
    wx, wy, wz, fgw = old_hybrid_blend(wx_b, wy_b, wz_b, wx_45, wy_45, wz_45,
                                        depth_compressed)
    return wx, wy, wz, disp_phys, fgw


# ══════════════════════════════════════════════════════════════════════════════
# NEW model — corrected box projection
# ══════════════════════════════════════════════════════════════════════════════

def new_depth_to_disp(depth_norm):
    compressed = np.power(depth_norm, 1.0)
    return NEW_DEPTH_BACK + compressed * NEW_DEPTH_RANGE


def new_box_projection(u_grid, v_grid, disp_phys_grid, W_img, H_img):
    """
    Pure box projection — no 45 degree plane, no hybrid blend.

    Left half:  X on left panel  [-content_w, 0],  Y = disp_phys (depth)
    Right half: X = disp_phys (depth),  Y on right panel [-content_w, 0]
    Blend zone with smoothstep at fold.
    """
    split_u = W_img // 2
    blend_px = max(int(BLEND_FRAC * W_img), BLEND_MIN_PX)
    v_norm = v_grid / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * CONTENT_H_M

    u_left_all = np.clip(u_grid.astype(np.float32) / max(split_u - 1, 1), 0.0, 1.0)
    x_left = -CONTENT_W_M + u_left_all * CONTENT_W_M
    y_left = disp_phys_grid

    u_right_all_n = max(W_img - split_u - 1, 1)
    u_right_all = np.clip((u_grid.astype(np.float32) - split_u) / u_right_all_n, 0.0, 1.0)
    x_right = disp_phys_grid
    y_right = -u_right_all * CONTENT_W_M

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


def new_full_projection(depth_norm, u_grid, v_grid, W_img, H_img):
    disp_phys = new_depth_to_disp(depth_norm)
    wx, wy, wz = new_box_projection(u_grid, v_grid, disp_phys, W_img, H_img)
    return wx, wy, wz, disp_phys


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(wx, wy, wz, panel_owner, W_img, H_img):
    n_total = wx.size

    pct_X_lt_0 = 100.0 * np.mean(wx < 0)
    pct_X_gt_0 = 100.0 * np.mean(wx > 0)

    on_left = ((np.abs(wy) < 0.02) & (wx >= -PANEL_PHYS_W_M - 0.01) &
               (wx <= 0.01) & (wz >= -PANEL_PHYS_H_M/2 - 0.01) &
               (wz <= PANEL_PHYS_H_M/2 + 0.01))
    on_right = ((np.abs(wx) < 0.02) & (wy >= -PANEL_PHYS_W_M - 0.01) &
                (wy <= 0.01) & (wz >= -PANEL_PHYS_H_M/2 - 0.01) &
                (wz <= PANEL_PHYS_H_M/2 + 0.01))

    pct_left_surface = 100.0 * np.mean(on_left)
    pct_right_surface = 100.0 * np.mean(on_right)

    in_left_bounds = ((wx >= -PANEL_PHYS_W_M) & (wx <= 0) &
                      (wz >= -PANEL_PHYS_H_M/2) & (wz <= PANEL_PHYS_H_M/2))
    in_right_bounds = ((wy >= -PANEL_PHYS_W_M) & (wy <= 0) &
                       (wz >= -PANEL_PHYS_H_M/2) & (wz <= PANEL_PHYS_H_M/2))

    pct_left_bounds = 100.0 * np.mean(in_left_bounds)
    pct_right_bounds = 100.0 * np.mean(in_right_bounds)

    return {
        'n_total': n_total,
        'pct_X_lt_0': pct_X_lt_0, 'pct_X_gt_0': pct_X_gt_0,
        'pct_left_surface': pct_left_surface, 'pct_right_surface': pct_right_surface,
        'pct_left_bounds': pct_left_bounds, 'pct_right_bounds': pct_right_bounds,
        'X_min': float(np.min(wx)), 'X_max': float(np.max(wx)),
        'Y_min': float(np.min(wy)), 'Y_max': float(np.max(wy)),
        'X_mean': float(np.mean(wx)), 'Y_mean': float(np.mean(wy)),
    }


def run_all_tests(W_img=200, H_img=180):
    u = np.arange(W_img, dtype=np.float32)
    v = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    split_u = W_img // 2
    blend_px = max(int(BLEND_FRAC * W_img), BLEND_MIN_PX)
    panel_left = (uu < (split_u + blend_px)).astype(np.uint8)
    panel_right = (uu >= (split_u - blend_px)).astype(np.uint8)
    panel_owner = np.where(panel_left & panel_right, 2, np.where(panel_left, 0, 1))

    depth_scenarios = {
        'uniform_near': np.full((H_img, W_img), 1.0, dtype=np.float32),
        'uniform_mid': np.full((H_img, W_img), 0.5, dtype=np.float32),
        'uniform_far': np.full((H_img, W_img), 0.0, dtype=np.float32),
        'gradient_center': np.zeros((H_img, W_img), dtype=np.float32),
    }
    cu, cv = W_img / 2, H_img / 2
    du = (uu - cu) / (W_img * 0.4)
    dv = (vv - cv) / (H_img * 0.4)
    depth_scenarios['gradient_center'] = np.clip(
        1.0 - np.sqrt(du**2 + dv**2), 0.0, 1.0).astype(np.float32)

    all_pass = True
    print("=" * 72)
    print("  Corrected Placement Model — RED/GREEN Validation")
    print("=" * 72)
    print()

    for name, depth_norm in depth_scenarios.items():
        owx, owy, owz, _, _ = old_full_projection(
            depth_norm, uu, vv, W_img, H_img)
        old_m = compute_metrics(owx, owy, owz, panel_owner, W_img, H_img)

        nwx, nwy, nwz, _ = new_full_projection(
            depth_norm, uu, vv, W_img, H_img)
        new_m = compute_metrics(nwx, nwy, nwz, panel_owner, W_img, H_img)

        print(f"--- {name} ---")
        print(f"  depth mean={float(np.mean(depth_norm)):.3f}")
        print(f"  OLD: X<0={old_m['pct_X_lt_0']:.1f}%  X>0={old_m['pct_X_gt_0']:.1f}%  "
              f"surf_L={old_m['pct_left_surface']:.1f}%  surf_R={old_m['pct_right_surface']:.1f}%")
        print(f"  NEW: X<0={new_m['pct_X_lt_0']:.1f}%  X>0={new_m['pct_X_gt_0']:.1f}%  "
              f"surf_L={new_m['pct_left_surface']:.1f}%  surf_R={new_m['pct_right_surface']:.1f}%")
        print(f"       X range=[{new_m['X_min']:.3f}, {new_m['X_max']:.3f}]  "
              f"mean X={new_m['X_mean']:.3f}  Y={new_m['Y_mean']:.3f}")

        if 'uniform_near' not in name:
            if old_m['pct_X_lt_0'] > 90.0:
                print(f"  [PASS] OLD X<0 bias confirmed ({old_m['pct_X_lt_0']:.1f}%)")
            else:
                print(f"  [FAIL] OLD expected >90% X<0, got {old_m['pct_X_lt_0']:.1f}%")
                all_pass = False

        if name == 'uniform_mid':
            # At dn=0.5 with symmetric depth, disp_phys=0.
            # Left half: X<0. Right half: X=0. So ~50% X<0, ~50% X=0.
            pct_X_ge_0 = 100.0 - new_m['pct_X_lt_0']
            if 40.0 < new_m['pct_X_lt_0'] < 60.0 and 40.0 < pct_X_ge_0 < 60.0:
                print(f"  [PASS] NEW X balanced: {new_m['pct_X_lt_0']:.1f}% X<0, "
                      f"{pct_X_ge_0:.1f}% X>=0 (right half at panel surface)")
            else:
                print(f"  [FAIL] NEW expected ~50% X<0 / ~50% X>=0, got "
                      f"{new_m['pct_X_lt_0']:.1f}% / {pct_X_ge_0:.1f}%")
                all_pass = False

        if name == 'uniform_near':
            if new_m['pct_X_gt_0'] > 40.0:
                print(f"  [PASS] NEW near depth X>0={new_m['pct_X_gt_0']:.1f}%")
            else:
                print(f"  [FAIL] NEW near expected X>0>40%, got {new_m['pct_X_gt_0']:.1f}%")
                all_pass = False

        if name == 'uniform_far':
            if new_m['pct_X_lt_0'] > 90.0:
                print(f"  [PASS] NEW far depth X<0={new_m['pct_X_lt_0']:.1f}% (physically correct)")
            else:
                print(f"  [FAIL] Far content should be behind panels")
                all_pass = False

        if new_m['pct_left_bounds'] > 40.0 and new_m['pct_right_bounds'] > 40.0:
            print(f"  [PASS] Panel bounds: L={new_m['pct_left_bounds']:.1f}% R={new_m['pct_right_bounds']:.1f}%")
        else:
            print(f"  [INFO] Panel bounds: L={new_m['pct_left_bounds']:.1f}% R={new_m['pct_right_bounds']:.1f}%")
        print()

    # ── Fold trace ──
    print("--- Fold Trace (u=W/2, v=H/2) ---")
    fold_u = split_u
    fold_v = H_img // 2
    old_neg = new_neg = new_pos = 0
    for dn in np.linspace(0.0, 1.0, 21):
        dp_old = OLD_DEPTH_BACK + dn * OLD_DEPTH_RANGE
        u_arr = np.array([[fold_u]], dtype=np.float32)
        v_arr = np.array([[fold_v]], dtype=np.float32)
        dp_arr_o = np.full((1, 1), dp_old, dtype=np.float32)
        owx_b, owy_b, _ = old_box_projection(u_arr, v_arr, dp_arr_o, W_img, H_img)
        owx_45, owy_45, _ = old_plane45_projection(u_arr, v_arr, dp_arr_o, W_img, H_img)
        dn_arr = np.full((1, 1), dn, dtype=np.float32)
        owx, _, _, _ = old_hybrid_blend(owx_b, owy_b, 0, owx_45, owy_45, 0, dn_arr)

        dp_new = NEW_DEPTH_BACK + dn * NEW_DEPTH_RANGE
        dp_arr_n = np.full((1, 1), dp_new, dtype=np.float32)
        nwx, _, _ = new_box_projection(u_arr, v_arr, dp_arr_n, W_img, H_img)

        if owx[0, 0] < 0:
            old_neg += 1
        if nwx[0, 0] < 0:
            new_neg += 1
        else:
            new_pos += 1

        if dn in (0.0, 0.25, 0.5, 0.75, 1.0):
            print(f"  dn={dn:.2f}  OLD X={owx[0,0]:+.4f}  NEW X={nwx[0,0]:+.4f}")

    print(f"\n  OLD: {old_neg}/21 negative  NEW: {new_neg}/21 negative, {new_pos}/21 positive")

    if new_pos > 0:
        print(f"  [PASS] NEW has {new_pos} positive fold traces (near-depth pop-out)")
    else:
        print(f"  [FAIL] NEW has zero positive fold traces")
        all_pass = False

    if old_neg == 21:
        print(f"  [PASS] OLD confirmed: all 21 fold traces negative")
    else:
        print(f"  [INFO] OLD: {old_neg}/21 negative")

    # ── Distortion ──
    print()
    print("--- Distortion ---")
    pp_u = CONTENT_W_M / W_img
    pp_v = CONTENT_H_M / H_img
    old_h = CONTENT_W_M * np.sqrt(2) / W_img
    print(f"  OLD 45-plane h-pitch: {old_h*1000:.2f}mm/px")
    print(f"  NEW box h-pitch:      {pp_u*1000:.2f}mm/px (no sqrt2 stretch)")
    print(f"  Vertical pitch:       {pp_v*1000:.2f}mm/px")

    if pp_u < old_h:
        print(f"  [PASS] NEW removes sqrt2 diagonal stretch")
    else:
        print(f"  [FAIL] Unexpected stretch")
        all_pass = False

    print()
    if all_pass:
        print("=" * 72)
        print("  ALL CHECKS PASSED")
        print("=" * 72)
        return 0
    else:
        print("=" * 72)
        print("  SOME CHECKS FAILED")
        print("=" * 72)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
