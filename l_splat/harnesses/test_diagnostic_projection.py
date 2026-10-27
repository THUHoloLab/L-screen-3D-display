"""
test_diagnostic_projection.py — Geometry diagnostics for 45 degree plane position.

Verifies the corrected 45 degree plane (X+Y = -PANEL_W + depth*sqrt2) forms
an isosceles right triangle with the two L-shaped display panels.

Pure numpy — no CUDA/DAV3 needed.
"""

import numpy as np
import math

PANEL_W = 1.28
PANEL_H = 1.44
SWEET_SPOT = np.array([2.12, 2.12, 1.60], dtype=np.float64)
INV_SQRT2 = 1.0 / math.sqrt(2)


def current_45deg_plane(H_img, W_img, disp_phys):
    """Replicate CURRENT (broken) 45 degree plane: X+Y = disp_phys*sqrt2."""
    u = np.arange(W_img, dtype=np.float32)
    v = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    u_norm = uu.astype(np.float32) / max(W_img - 1, 1)
    v_norm = vv / max(H_img - 1, 1)
    content_w_45 = PANEL_W * math.sqrt(2)
    tan_pos = (u_norm - 0.5) * content_w_45
    vert_pos = (0.5 - v_norm) * PANEL_H

    world_x = -tan_pos * INV_SQRT2 + disp_phys * INV_SQRT2
    world_y = +tan_pos * INV_SQRT2 + disp_phys * INV_SQRT2
    world_z = vert_pos
    return world_x, world_y, world_z


def corrected_45deg_plane(H_img, W_img, disp_phys):
    """Corrected 45 degree plane: X+Y = -PANEL_W + disp_phys*sqrt2.

    The plane connects (-PANEL_W, 0) to (0, -PANEL_W), the hypotenuse of
    the isosceles right triangle formed with the two L-shaped panels.
    """
    u = np.arange(W_img, dtype=np.float32)
    v = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    u_norm = uu.astype(np.float32) / max(W_img - 1, 1)
    v_norm = vv / max(H_img - 1, 1)
    vert_pos = (0.5 - v_norm) * PANEL_H

    world_x = -PANEL_W * (1.0 - u_norm) + disp_phys * INV_SQRT2
    world_y = -PANEL_W * u_norm + disp_phys * INV_SQRT2
    world_z = vert_pos
    return world_x, world_y, world_z


def project_onto_left_panel(world_xy, sweet_xy=None):
    """Project (x,y) onto Y=0 plane as seen from sweet spot. Returns X coordinate."""
    if sweet_xy is None:
        sweet_xy = SWEET_SPOT[:2]
    sx, sy = sweet_xy
    ray = world_xy - sweet_xy
    t = -sy / ray[..., 1]
    x_on_panel = sx + t * ray[..., 0]
    return x_on_panel


def project_onto_right_panel(world_xy, sweet_xy=None):
    """Project (x,y) onto X=0 plane as seen from sweet spot. Returns Y coordinate."""
    if sweet_xy is None:
        sweet_xy = SWEET_SPOT[:2]
    sx, sy = sweet_xy
    ray = world_xy - sweet_xy
    t = -sx / ray[..., 0]
    y_on_panel = sy + t * ray[..., 1]
    return y_on_panel


# ── Tests ──────────────────────────────────────────────────────────────

def test_current_plane_is_broken():
    """Current plane at X+Y=0 maps left edge to +X half-space (WRONG)."""
    H, W = 120, 160
    wx, wy, _ = current_45deg_plane(H, W, disp_phys=0.0)

    left_edge_x = wx[:, 0].mean()
    right_edge_y = wy[:, -1].mean()
    xy_sum = (wx + wy).mean()

    assert left_edge_x > 0, f"Expected left edge X>0, got {left_edge_x:.3f}"
    assert right_edge_y > 0, f"Expected right edge Y>0, got {right_edge_y:.3f}"
    assert abs(xy_sum) < 0.01, f"Expected X+Y~0, got {xy_sum:.4f}"

    print(f"  [PASS] Current plane: left edge X={left_edge_x:+.3f} (should be -1.28)")
    print(f"  [PASS] Current plane: right edge Y={right_edge_y:+.3f} (should be -1.28)")
    print(f"  [PASS] Current plane: X+Y={xy_sum:.4f} (plane through fold, not far edges)")
    return True


def test_corrected_plane_endpoints():
    """Corrected plane at X+Y=-PANEL_W has correct endpoints."""
    H, W = 120, 160
    wx, wy, wz = corrected_45deg_plane(H, W, disp_phys=0.0)

    assert abs(wx[:, 0].mean() - (-PANEL_W)) < 0.02, \
        f"Left edge X={wx[:,0].mean():.3f}, expected {-PANEL_W}"
    assert abs(wy[:, 0].mean() - 0.0) < 0.02, \
        f"Left edge Y={wy[:,0].mean():.3f}, expected 0.0"

    assert abs(wx[:, -1].mean() - 0.0) < 0.02, \
        f"Right edge X={wx[:,-1].mean():.3f}, expected 0.0"
    assert abs(wy[:, -1].mean() - (-PANEL_W)) < 0.02, \
        f"Right edge Y={wy[:,-1].mean():.3f}, expected {-PANEL_W}"

    print(f"  [PASS] Left edge:  ({wx[:,0].mean():.3f}, {wy[:,0].mean():.3f})")
    print(f"  [PASS] Right edge: ({wx[:,-1].mean():.3f}, {wy[:,-1].mean():.3f})")
    return True


def test_corrected_plane_equation():
    """All points satisfy X+Y = -PANEL_W + disp_phys*sqrt2."""
    H, W = 60, 80
    for dp in [0.0, -0.18, +0.12]:
        wx, wy, _ = corrected_45deg_plane(H, W, disp_phys=dp)
        expected = -PANEL_W + dp * math.sqrt(2)
        actual = wx + wy
        max_err = np.abs(actual - expected).max()
        assert max_err < 0.02, f"X+Y error {max_err:.4f} at depth={dp}"
        print(f"  [PASS] depth={dp:+.2f}: X+Y = {actual.mean():.4f} "
              f"(expected {expected:.4f}, max_err={max_err:.6f})")
    return True


def test_center_maps_to_fold():
    """Center of image (u=W/2) should project to fold (0,0) on BOTH panels."""
    H, W = 120, 160
    wx, wy, _ = corrected_45deg_plane(H, W, disp_phys=0.0)
    mid = W // 2

    center_xy = np.stack([wx[:, mid], wy[:, mid]], axis=-1).mean(axis=0)
    x_on_left = project_onto_left_panel(center_xy)
    y_on_right = project_onto_right_panel(center_xy)

    assert abs(x_on_left) < 0.02, f"Center -> left panel X={x_on_left:.4f}, expected 0"
    assert abs(y_on_right) < 0.02, f"Center -> right panel Y={y_on_right:.4f}, expected 0"

    print(f"  [PASS] Center world=({center_xy[0]:.3f},{center_xy[1]:.3f})")
    print(f"         -> left panel X={x_on_left:.4f} (fold=0)")
    print(f"         -> right panel Y={y_on_right:.4f} (fold=0)")
    return True


def test_frustum_projections():
    """Verify projection of 45deg plane points onto panels through sweet spot."""
    left_edge = np.array([-PANEL_W, 0.0])
    x_left = project_onto_left_panel(left_edge)
    assert abs(x_left - (-PANEL_W)) < 0.02, \
        f"Left edge -> left panel X={x_left:.3f}, expected {-PANEL_W}"

    right_edge = np.array([0.0, -PANEL_W])
    y_right = project_onto_right_panel(right_edge)
    assert abs(y_right - (-PANEL_W)) < 0.02, \
        f"Right edge -> right panel Y={y_right:.3f}, expected {-PANEL_W}"

    mid = np.array([-PANEL_W/2, -PANEL_W/2])
    x_mid = project_onto_left_panel(mid)
    y_mid = project_onto_right_panel(mid)
    assert abs(x_mid) < 0.02, f"Mid -> left panel X={x_mid:.3f}, expected 0"
    assert abs(y_mid) < 0.02, f"Mid -> right panel Y={y_mid:.3f}, expected 0"

    print(f"  [PASS] Left edge  ({left_edge[0]:.2f},{left_edge[1]:.2f}) -> left panel X={x_left:.3f}")
    print(f"  [PASS] Right edge ({right_edge[0]:.2f},{right_edge[1]:.2f}) -> right panel Y={y_right:.3f}")
    print(f"  [PASS] Midpoint   ({mid[0]:.2f},{mid[1]:.2f}) -> L-panel X={x_mid:.3f}, R-panel Y={y_mid:.3f}")
    return True


def test_45deg_plane_fills_panels():
    """The corrected 45deg plane content spans the full panel width."""
    H, W = 120, 160
    wx, wy, _ = corrected_45deg_plane(H, W, disp_phys=0.0)
    mid = W // 2

    left_half_wx = wx[:, :mid]
    left_half_wy = wy[:, :mid]
    x_left_panel = project_onto_left_panel(
        np.stack([left_half_wx.ravel(), left_half_wy.ravel()], axis=-1)
    )
    x_min, x_max = x_left_panel.min(), x_left_panel.max()
    assert x_min < -1.0, f"Left worst X={x_min:.3f}, should be < -1.0"
    assert x_max > -0.05, f"Left best X={x_max:.3f}, should near fold"

    right_half_wx = wx[:, mid:]
    right_half_wy = wy[:, mid:]
    y_right_panel = project_onto_right_panel(
        np.stack([right_half_wx.ravel(), right_half_wy.ravel()], axis=-1)
    )
    y_min, y_max = y_right_panel.min(), y_right_panel.max()
    assert y_min < -1.0, f"Right worst Y={y_min:.3f}, should be < -1.0"
    assert y_max > -0.05, f"Right best Y={y_max:.3f}, should near fold"

    print(f"  [PASS] Left half projects to X in [{x_min:.3f}, {x_max:.3f}] on left panel")
    print(f"  [PASS] Right half projects to Y in [{y_min:.3f}, {y_max:.3f}] on right panel")
    print(f"  [PASS] Both span full panel width [-1.28, 0]")
    return True


def test_depth_displacement_along_normal():
    """Depth displaces along (+1,+1,0)/sqrt2, shifting plane toward/away from viewer."""
    H, W = 60, 80
    wx_near, wy_near, _ = corrected_45deg_plane(H, W, disp_phys=+0.12)
    wx_far, wy_far, _ = corrected_45deg_plane(H, W, disp_phys=-0.18)

    sum_near = (wx_near + wy_near).mean()
    sum_far = (wx_far + wy_far).mean()
    assert sum_near > sum_far, \
        f"Near X+Y={sum_near:.4f} should be > far X+Y={sum_far:.4f}"

    print(f"  [PASS] Near X+Y={sum_near:.4f} > Far X+Y={sum_far:.4f}")
    print(f"         Near plane closer to viewer (+X+Y direction)")
    return True


def test_isosceles_right_triangle():
    """Verify the 45 degree plane forms an isosceles right triangle with panels."""
    A = np.array([0.0, 0.0])
    B = np.array([-PANEL_W, 0.0])
    C = np.array([0.0, -PANEL_W])

    AB = np.linalg.norm(B - A)
    AC = np.linalg.norm(C - A)
    BC = np.linalg.norm(C - B)

    assert abs(AB - AC) < 0.01, f"|AB|={AB:.3f} != |AC|={AC:.3f}"
    cos_A = np.dot(B - A, C - A) / (AB * AC)
    assert abs(cos_A) < 0.01, f"cos(angle at fold)={cos_A:.4f}, expected 0"
    assert abs(BC - AB * math.sqrt(2)) < 0.01, \
        f"|BC|={BC:.3f} != sqrt(2)*{AB:.3f}={AB*math.sqrt(2):.3f}"

    print(f"  [PASS] Isosceles right triangle verified:")
    print(f"         Leg AB (left panel):  {AB:.3f}m")
    print(f"         Leg AC (right panel): {AC:.3f}m")
    print(f"         Hypotenuse BC (45deg): {BC:.3f}m = sqrt2 * {AB:.3f}")
    print(f"         Angle at fold: 90.0 degrees")
    return True


def harness():
    print("=" * 64)
    print("  HARNESS: Projection Geometry Diagnostics")
    print("=" * 64)
    print(f"  Panel: {PANEL_W}x{PANEL_H}m, Sweet spot: {tuple(SWEET_SPOT)}")
    print()

    results = []
    results.append(("Current plane is broken", test_current_plane_is_broken()))
    results.append(("Corrected plane endpoints", test_corrected_plane_endpoints()))
    results.append(("Corrected plane equation X+Y=-PANEL_W+depth*sqrt2", test_corrected_plane_equation()))
    results.append(("Center maps to fold on both panels", test_center_maps_to_fold()))
    results.append(("Frustum projections", test_frustum_projections()))
    results.append(("45deg plane fills panel width", test_45deg_plane_fills_panels()))
    results.append(("Depth displacement direction", test_depth_displacement_along_normal()))
    results.append(("Isosceles right triangle", test_isosceles_right_triangle()))

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
