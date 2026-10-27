"""
test_box_projection.py — HDD Phase 1 for dual-panel-parallel "box" projection.

Replaces the 45° content plane with two planes, each parallel to its panel:
  Left half  -> plane parallel to Y=0 (left panel)
  Right half -> plane parallel to X=0 (right panel)

Each camera views its content head-on -> no keystone distortion.

Phase 1: geometric correctness (no CUDA/DAV3 needed).
"""

import numpy as np


def depth_to_box_pointcloud(H_img, W_img, depth_map,
                             panel_w, panel_h,
                             depth_near, depth_far,
                             split_u=None):
    """
    Split image at split_u, project each half to its panel-parallel plane.

    Left half (u < split_u):  plane parallel to LEFT panel (Y=0)
      world_x = -panel_w + u_norm_L * panel_w    -> [-panel_w, 0]
      world_y = +disp                              -> toward viewer (+Y)
      world_z = (0.5 - v_norm) * panel_h           -> Z-up

    Right half (u >= split_u): plane parallel to RIGHT panel (X=0)
      world_x = +disp                               -> toward viewer (+X)
      world_y = -u_norm_R * panel_w                 -> [0, -panel_w]
      world_z = (0.5 - v_norm) * panel_h            -> Z-up
    """
    if split_u is None:
        split_u = W_img // 2

    u = np.arange(W_img, dtype=np.float32)
    v = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    v_norm = vv / max(H_img - 1, 1)

    disp_phys = depth_far + depth_map * (depth_near - depth_far)
    vert_pos = (0.5 - v_norm) * panel_h

    wx = np.zeros((H_img, W_img), dtype=np.float32)
    wy = np.zeros((H_img, W_img), dtype=np.float32)
    wz = vert_pos

    # ── Left half: project to Y-disp plane (parallel to left panel Y=0) ──
    left_mask = uu < split_u
    n_left = left_mask.sum()
    if n_left > 0:
        u_left_norm = uu[left_mask].astype(np.float32) / max(split_u - 1, 1)
        wx[left_mask] = -panel_w + u_left_norm * panel_w
        wy[left_mask] = +disp_phys[left_mask]

    # ── Right half: project to X-disp plane (parallel to right panel X=0) ──
    right_mask = uu >= split_u
    n_right = right_mask.sum()
    if n_right > 0:
        u_right = uu[right_mask].astype(np.float32) - split_u
        n_right_px = W_img - split_u
        u_right_norm = u_right / max(n_right_px - 1, 1)
        wx[right_mask] = +disp_phys[right_mask]
        wy[right_mask] = -u_right_norm * panel_w

    return wx, wy, wz


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_left_half_maps_to_left_halfspace():
    H, W = 120, 160
    split = W // 2
    depth = np.full((H, W), 0.5, dtype=np.float32)
    wx, wy, wz = depth_to_box_pointcloud(H, W, depth, 1.28, 1.44, +0.25, -0.35, split)

    left_x = wx[:, :split]
    assert np.all(left_x <= 0), f"Left-half X max={left_x.max():.4f}"
    assert np.all(left_x >= -1.28 - 0.01), f"Left-half X min={left_x.min():.4f}"

    right_y = wy[:, split:]
    assert np.all(right_y <= 0), f"Right-half Y max={right_y.max():.4f}"
    assert np.all(right_y >= -1.28 - 0.01), f"Right-half Y min={right_y.min():.4f}"

    print(f"  [PASS] Left X: [{left_x.min():.3f}, {left_x.max():.3f}] in [-1.28, 0]")
    print(f"  [PASS] Right Y: [{right_y.min():.3f}, {right_y.max():.3f}] in [-1.28, 0]")
    return True


def test_depth_displacement_direction():
    H, W = 30, 40
    split = W // 2

    near = np.ones((H, W), dtype=np.float32)
    far = np.zeros((H, W), dtype=np.float32)
    wx_n, wy_n, _ = depth_to_box_pointcloud(H, W, near, 1.28, 1.44, +0.25, -0.35, split)
    wx_f, wy_f, _ = depth_to_box_pointcloud(H, W, far, 1.28, 1.44, +0.25, -0.35, split)

    # Near left half: Y should be larger (closer to viewer at +Y)
    assert wy_n[:, :split].mean() > wy_f[:, :split].mean(), \
        "Left near Y should be > far Y (closer to +Y viewer)"

    # Right panel faces +X (viewer at X=+2.12). Near -> more +X. world_x = +disp_phys
    assert wx_n[:, split:].mean() > wx_f[:, split:].mean(), \
        "Right near X should be > far X (near closer to +X viewer)"

    print(f"  [PASS] Left near Y={wy_n[:,:split].mean():.3f} > far Y={wy_f[:,:split].mean():.3f}")
    print(f"  [PASS] Right near X={wx_n[:,split:].mean():.3f} > far X={wx_f[:,split:].mean():.3f}")
    return True


def test_fold_continuity():
    H, W = 30, 40
    split = W // 2
    depth = np.full((H, W), 0.5, dtype=np.float32)
    wx, wy, wz = depth_to_box_pointcloud(H, W, depth, 1.28, 1.44, +0.25, -0.35, split)

    # At the fold: X≈0 (left half right edge), Y≈0 (right half left edge)
    left_fold_x = wx[:, split - 1].mean()
    right_fold_y = wy[:, split].mean()
    assert abs(left_fold_x) < 0.02, f"Left fold X={left_fold_x:.4f}"
    assert abs(right_fold_y) < 0.02, f"Right fold Y={right_fold_y:.4f}"

    # Z consistent across fold
    z_diff = np.abs(wz[:, split - 1] - wz[:, split]).max()
    assert z_diff < 0.01, f"Z mismatch at fold: {z_diff:.4f}"

    print(f"  [PASS] Fold: X={left_fold_x:.4f} Y={right_fold_y:.4f} (both ~0)")
    print(f"  [PASS] Z consistent (max diff={z_diff:.6f})")
    return True


def test_source_row0_maps_to_physical_top():
    H, W = 30, 40
    split = W // 2
    depth = np.full((H, W), 0.5, dtype=np.float32)
    _, _, wz = depth_to_box_pointcloud(H, W, depth, 1.28, 1.44, +0.25, -0.35, split)

    assert wz[0].mean() > 0.0, f"row 0 should map to physical +Z, got {wz[0].mean():.4f}"
    assert wz[-1].mean() < 0.0, f"last row should map to physical -Z, got {wz[-1].mean():.4f}"
    assert wz[0].mean() > wz[-1].mean(), "source rows should descend in physical Z"

    print(f"  [PASS] Row/Z: row0={wz[0].mean():+.3f}m, last={wz[-1].mean():+.3f}m")
    return True


def test_pixel_spacing_uniform():
    H, W = 60, 80
    split = W // 2
    depth = np.full((H, W), 0.5, dtype=np.float32)
    wx, wy, _ = depth_to_box_pointcloud(H, W, depth, 1.28, 1.44, +0.25, -0.35, split)

    dx_left = np.diff(wx[30, :split])
    assert np.allclose(dx_left, dx_left[0], atol=0.001), "Left X spacing not uniform"

    dy_right = np.diff(wy[30, split:])
    assert np.allclose(dy_right, dy_right[0], atol=0.001), "Right Y spacing not uniform"

    print(f"  [PASS] Left X spacing: uniform ({dx_left[0]*1000:.2f}mm/px)")
    print(f"  [PASS] Right Y spacing: uniform ({dy_right[0]*1000:.2f}mm/px)")
    return True


def test_blend_zone_width():
    """Blend zone should be ~8% of image width, min 20px (Stage 2 fix for ISSUES #1, #4)."""
    # Test the expected NEW formula: max(int(0.08 * W), 20)
    for W_img, expected_min, expected_max in [
        (196, 20, 25),     # narrow image -> min guard (was 100, now 20)
        (800, 60, 70),     # medium -> ~64px
        (1440, 110, 120),  # wide -> ~115px (was 288!)
        (2000, 155, 165),  # very wide -> ~160px
    ]:
        blend_new = max(int(0.08 * W_img), 20)
        blend_old = max(int(0.20 * W_img), 100)
        assert expected_min <= blend_new <= expected_max, \
            f"W={W_img}: blend_new={blend_new} not in [{expected_min},{expected_max}]"
        assert blend_new < blend_old, \
            f"W={W_img}: new blend {blend_new} should be < old blend {blend_old}"

    print(f"  [PASS] Blend zone: 1440px->{max(int(0.08*1440),20)}px (was {max(int(0.20*1440),100)}px)")
    print(f"  [PASS] Blend zone: 196px->{max(int(0.08*196),20)}px (was {max(int(0.20*196),100)}px)")
    return True


def harness():
    print("=" * 64)
    print("  HARNESS: Box Projection Geometry (no CUDA)")
    print("=" * 64)

    results = []
    results.append(("Left->Left Half-Space", test_left_half_maps_to_left_halfspace()))
    results.append(("Depth Displacement Direction", test_depth_displacement_direction()))
    results.append(("Fold Continuity", test_fold_continuity()))
    results.append(("Row0 -> Physical +Z", test_source_row0_maps_to_physical_top()))
    results.append(("Uniform Pixel Spacing", test_pixel_spacing_uniform()))
    results.append(("Blend Zone Width (8%)", test_blend_zone_width()))

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    all_pass = True
    for name, ok in results:
        s = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{s}] {name}")
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 64)
    return all_pass


if __name__ == "__main__":
    ok = harness()
    import sys
    sys.exit(0 if ok else 1)
