"""
harness_physical_frustum.py — Phase 1 & 2 HDD test harness.

Verifies that the dual asymmetric frustum projection is perfectly continuous
at the fold line (X=0, Y=0) for the L-shaped 90° outward-fold LED screen.

Physical parameters:
  - Screen width  W = 1.28 m
  - Screen height H = 1.60 m
  - Left screen:  Y=0 plane, X in [0, -1.28] (extends toward -X from fold)
  - Right screen: X=0 plane, Y in [0, -1.28] (extends toward -Y from fold)
  - Screen bottom is 0.50 m above floor; viewer eye height is 1.70 m.
  - Viewer horizontal distance to fold is 3.00 m.
  - Viewer (Sweet Spot): V = (3/sqrt(2), 3/sqrt(2), 0.40) meters
  - Z-up right-handed world coordinate system.
  - Fold line = Z axis (X=0, Y=0).

Key assertion: the projection of any 3D point at the fold (X=0, Y=0, Z)
onto the near plane MUST be:
  - Left camera:  x_near = r_left   (right edge, NDC x = +1)
  - Right camera: x_near = l_right  (left edge,  NDC x = -1)

Furthermore, for content on the content plane X+Y=0 intersecting the fold,
the depth scaling and parallax must be consistent across both views.
"""

import numpy as np

from l_splat.geometry.frustum import (
    look_at_offscreen,
    apply_flip_y,
    apply_flip_x,
    FLIP_Y,
    FLIP_X,
    compute_frustum_bounds,
    build_asymmetric_proj_matrix,
    build_l_screen_corners,
)

# ===========================================================================
# Physical configuration
# ===========================================================================
PANEL_W = 1.28   # screen width (m)
PANEL_H = 1.60   # screen height (m)
HALF_H = PANEL_H / 2.0   # 0.80 m

VIEWER_HORIZONTAL_DISTANCE_M = 3.00
VIEWER_EYE_HEIGHT_M = 1.70
SCREEN_BOTTOM_HEIGHT_M = 0.50
VIEWER_XY_M = VIEWER_HORIZONTAL_DISTANCE_M / np.sqrt(2.0)
VIEWER_WORLD_Z_M = (
    VIEWER_EYE_HEIGHT_M - (SCREEN_BOTTOM_HEIGHT_M + PANEL_H / 2.0)
)
VIEWER_POS = np.array(
    [VIEWER_XY_M, VIEWER_XY_M, VIEWER_WORLD_Z_M], dtype=np.float64)
NEAR_CLIP = 0.01

# Build screen corners using the same convention as build_l_screen_corners.
# Order: [top-left, top-right, bottom-left, bottom-right]
# "top" = +Z, "bottom" = -Z.


# ===========================================================================
# View matrix: look_at_offscreen + flip_y (Candidate H from previous harness)
# ===========================================================================



def build_view_matrix(eye, forward_world, up_guess):
    """Build Candidate-H view matrix for one camera."""
    target = eye + forward_world
    V_raw = look_at_offscreen(eye, target, up_guess)
    return apply_flip_y(V_raw)


# ===========================================================================
# Frustum bounds computation
# ===========================================================================



def project_to_ndc(V, P, point_world):
    """Project a world point to NDC coordinates [x_ndc, y_ndc, z_ndc]."""
    p_h = np.append(np.asarray(point_world, dtype=np.float64), 1.0)
    p_view = V @ p_h
    p_clip = P @ p_view
    w = p_clip[3]
    if abs(w) < 1e-15:
        return None
    return np.array([p_clip[0]/w, p_clip[1]/w, p_clip[2]/w])


# ===========================================================================
# Build both cameras
# ===========================================================================

def build_dual_frustum(viewer, W, H, near, far):
    """Build complete dual-frustum camera system and return all parameters."""
    left_corners, right_corners = build_l_screen_corners(W, H)

    # Left camera: looks perpendicular to Y=0 plane, forward = (0, -1, 0)
    V_L = build_view_matrix(viewer, np.array([0.0, -1.0, 0.0]),
                            np.array([0.0, 0.0, 1.0]))
    (l_L, r_L, b_L, t_L), _ = compute_frustum_bounds(viewer, left_corners, near, V_L)
    P_L, _ = build_asymmetric_proj_matrix(near, far, l_L, r_L, b_L, t_L)

    # Right camera: looks perpendicular to X=0 plane, forward = (-1, 0, 0)
    V_R = build_view_matrix(viewer, np.array([-1.0, 0.0, 0.0]),
                            np.array([0.0, 0.0, 1.0]))
    (l_R, r_R, b_R, t_R), _ = compute_frustum_bounds(viewer, right_corners, near, V_R)
    P_R, _ = build_asymmetric_proj_matrix(near, far, l_R, r_R, b_R, t_R)

    return {
        "V_L": V_L, "P_L": P_L, "l_L": l_L, "r_L": r_L, "b_L": b_L, "t_L": t_L,
        "V_R": V_R, "P_R": P_R, "l_R": l_R, "r_R": r_R, "b_R": b_R, "t_R": t_R,
        "left_corners": left_corners, "right_corners": right_corners,
    }


# ===========================================================================
# Tests
# ===========================================================================


def test_patent_outward_fold_contract(cam, viewer):
    """Verify the patent's exact outward-fold geometry and sweet-spot side."""
    left_corners = cam["left_corners"]
    right_corners = cam["right_corners"]

    left_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    left_tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right_tangent = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    fold_tangent = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    expected_left = np.array([
        [-PANEL_W, 0.0, HALF_H],
        [0.0, 0.0, HALF_H],
        [-PANEL_W, 0.0, -HALF_H],
        [0.0, 0.0, -HALF_H],
    ])
    expected_right = np.array([
        [0.0, 0.0, HALF_H],
        [0.0, -PANEL_W, HALF_H],
        [0.0, 0.0, -HALF_H],
        [0.0, -PANEL_W, -HALF_H],
    ])
    np.testing.assert_allclose(left_corners, expected_left, atol=1e-12)
    np.testing.assert_allclose(right_corners, expected_right, atol=1e-12)

    expected_viewer = np.array([
        3.0 / np.sqrt(2.0),
        3.0 / np.sqrt(2.0),
        0.40,
    ])
    np.testing.assert_allclose(viewer, expected_viewer, atol=1e-12)

    left_center = left_corners.mean(axis=0)
    right_center = right_corners.mean(axis=0)
    assert np.dot(viewer - left_center, left_normal) > 0.0
    assert np.dot(viewer - right_center, right_normal) > 0.0

    np.testing.assert_allclose(-cam["V_L"][2, :3], -left_normal, atol=1e-12)
    np.testing.assert_allclose(-cam["V_R"][2, :3], -right_normal, atol=1e-12)
    assert abs(np.dot(left_tangent, right_tangent)) < 1e-12
    assert abs(np.dot(left_normal, fold_tangent)) < 1e-12
    assert abs(np.dot(right_normal, fold_tangent)) < 1e-12

    displacement_m = 0.15
    left_surface_distance = np.linalg.norm(viewer - left_center)
    right_surface_distance = np.linalg.norm(viewer - right_center)
    assert np.linalg.norm(viewer - (left_center + displacement_m * left_normal)) < left_surface_distance
    assert np.linalg.norm(viewer - (right_center + displacement_m * right_normal)) < right_surface_distance

    print("\n  ASSERT: patent outward-fold geometry contract — PASSED")
    return True

def test_fold_continuity(cam, viewer, near):
    """
    THE CRITICAL TEST: For points on the fold line (X=0, Y=0, varying Z),
    verify that:
      1. Left camera maps them to its RIGHT edge (x_ndc ≈ +1)
      2. Right camera maps them to its LEFT edge (x_ndc ≈ -1)
      3. The Y-NDC values match (same vertical position in both images)
      4. The Z-NDC (depth) values are consistent
    """
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]
    l_L, r_L = cam["l_L"], cam["r_L"]
    l_R, r_R = cam["l_R"], cam["r_R"]

    print("=" * 70)
    print("  TEST 1: Fold Continuity (X=0, Y=0)")
    print("=" * 70)

    z_samples = np.linspace(-HALF_H, HALF_H, 7)
    all_pass = True

    for z in z_samples:
        pt = np.array([0.0, 0.0, z], dtype=np.float64)

        ndc_L = project_to_ndc(V_L, P_L, pt)
        ndc_R = project_to_ndc(V_R, P_R, pt)

        if ndc_L is None or ndc_R is None:
            print(f"  Z={z:+.2f}: ERROR — point behind camera")
            all_pass = False
            continue

        # Check fold at correct edges
        xL_ok = abs(ndc_L[0] - 1.0) < 0.02   # left cam: right edge
        xR_ok = abs(ndc_R[0] - (-1.0)) < 0.02  # right cam: left edge
        # Vertical consistency: same Y-NDC for same Z (up to numerical error)
        y_ok = abs(ndc_L[1] - ndc_R[1]) < 0.02
        # Depth consistency
        z_ok = abs(ndc_L[2] - ndc_R[2]) < 0.05

        status = "PASS" if (xL_ok and xR_ok and y_ok and z_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  Z={z:+.2f}:  "
              f"L_NDC=({ndc_L[0]:+.4f}, {ndc_L[1]:+.4f}, {ndc_L[2]:+.4f})  "
              f"R_NDC=({ndc_R[0]:+.4f}, {ndc_R[1]:+.4f}, {ndc_R[2]:+.4f})  "
              f"xL={xL_ok} xR={xR_ok} y_match={y_ok} z_match={z_ok}  [{status}]")

    # --- Assertion ---
    assert all_pass, "FOLD CONTINUITY FAILED: fold points don't map to correct edges!"
    print("\n  ASSERT: fold continuity — PASSED")
    return all_pass


def test_frustum_bound_match(cam):
    """
    Verify that the frustum bounds at the fold are mathematically consistent.

    The left camera's 'r' boundary (right edge) corresponds to the fold (X=0).
    The right camera's 'l' boundary (left edge) corresponds to the fold (Y=0).

    These are computed from screen corner projections. The key insight:
    both the left screen's right corners and the right screen's left corners
    are at the same world-space location (X=0, Y=0, Z=±H/2).

    Therefore, after projection through their respective view matrices,
    the x_near values at the fold corners should have the SAME MAGNITUDE
    (differing only in sign, since they're on opposite sides of the optical axis).
    """
    V_L, V_R = cam["V_L"], cam["V_R"]
    r_L = cam["r_L"]   # left camera's right boundary (fold)
    l_R = cam["l_R"]   # right camera's left boundary (fold)

    print("\n" + "=" * 70)
    print("  TEST 2: Frustum Bound Magnitude Match at Fold")
    print("=" * 70)

    print(f"  Left  camera: l={cam['l_L']:.6f}  r={cam['r_L']:.6f} "
          f" b={cam['b_L']:.6f}  t={cam['t_L']:.6f}")
    print(f"  Right camera: l={cam['l_R']:.6f}  r={cam['r_R']:.6f} "
          f" b={cam['b_R']:.6f}  t={cam['t_R']:.6f}")

    # The fold bound magnitudes should match (|r_L| ≈ |l_R|)
    # since both represent the distance from optical axis to fold, scaled by near/z.
    ratio = abs(r_L) / (abs(l_R) + 1e-15)
    bound_ok = abs(ratio - 1.0) < 0.02

    print(f"\n  |r_L| / |l_R| = {abs(r_L):.6f} / {abs(l_R):.6f} = {ratio:.6f}")
    print(f"  Bound magnitude match: {'PASS' if bound_ok else 'FAIL'} "
          f"(expected ratio ≈ 1.0)")

    # Verify both fold-corners produce the same x_near magnitude
    # Left screen top-right corner (at fold): (0, 0, +H/2)
    # Right screen top-left corner (at fold): (0, 0, +H/2)
    # These are the SAME world point!
    fold_top = np.array([0.0, 0.0, HALF_H, 1.0], dtype=np.float64)
    fold_bot = np.array([0.0, 0.0, -HALF_H, 1.0], dtype=np.float64)

    for label, pt_h in [("Z=+H/2 (top)", fold_top), ("Z=-H/2 (bot)", fold_bot)]:
        vL = V_L @ pt_h
        vR = V_R @ pt_h
        tL = -NEAR_CLIP / vL[2]
        tR = -NEAR_CLIP / vR[2]
        xnL = vL[0] * tL
        xnR = vR[0] * tR
        print(f"  {label}:  x_near_L={xnL:.6f}  x_near_R={xnR:.6f}  "
              f"mag_ratio={abs(xnL)/(abs(xnR)+1e-15):.6f}")

    assert bound_ok, \
        f"FRUSTUM BOUND MISMATCH: |r_L|/|l_R| = {ratio:.4f} != 1.0 (diff={abs(ratio-1)*100:.2f}%)"
    print("\n  ASSERT: frustum bound magnitude match — PASSED")
    return bound_ok


def test_content_plane_parallax(cam, viewer, near, far):
    """
    Test points on the content plane X+Y=0 at varying distances behind
    the screens. Verify that the parallax is consistent between views.

    The content plane is the 45° bisector. Points on it should project
    consistently: a point with coordinates (-d, +d, z) on the content plane
    is behind the left screen (Y=d>0, in front of the Y=0 screen plane)
    and visible to the left camera. Similarly, (+d, -d, z) is behind the
    right screen.

    For a point EXACTLY at the fold (0, 0, z), both cameras see it at
    their respective fold edges. For a point slightly offset from the fold,
    the displacement from the fold edge should be consistent.
    """
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST 3: Content-Plane Parallax Consistency")
    print("=" * 70)

    # Sample points on the content plane X+Y=0, at the fold
    # Point: (eps, -eps, z) — slightly on the RIGHT side of the fold
    #   Left camera should see it near its right edge
    #   Right camera should see it just to the right of its left edge
    eps_values = [0.0, 0.01, 0.05, 0.10]
    z_test = 0.0  # mid-height

    all_ok = True

    for eps in eps_values:
        # Point on content plane: X+Y=0, on the right side (X>0, Y<0)
        pt = np.array([eps, -eps, z_test], dtype=np.float64)

        ndc_L = project_to_ndc(V_L, P_L, pt)
        ndc_R = project_to_ndc(V_R, P_R, pt)

        if ndc_L is None or ndc_R is None:
            print(f"  eps={eps:.2f}: behind camera, skipping")
            continue

        # For eps=0: both should be at fold edges
        # For eps>0: point is to the RIGHT of the fold
        #   → Right camera should see it more clearly (left edge + some offset)
        #   → Left camera may or may not see it (behind Y=0 screen plane)

        # The key check: for non-zero eps, the displacement from the fold edge
        # should be directionally consistent
        disp_L = ndc_L[0] - 1.0    # displacement from left-camera's right edge
        disp_R = ndc_R[0] - (-1.0) # displacement from right-camera's left edge

        print(f"  eps={eps:+.2f} (X={eps:+.2f}, Y={-eps:+.2f}):  "
              f"L_xNDC={ndc_L[0]:+.4f} (disp={disp_L:+.4f})  "
              f"R_xNDC={ndc_R[0]:+.4f} (disp={disp_R:+.4f})  "
              f"y_match={abs(ndc_L[1]-ndc_R[1])<0.02}")

        # For a point on the content plane, the right-camera parallax
        # should make the point shift RIGHT from the fold edge (positive disp_R)
        # as the point moves away from the fold into the right screen's volume.
        if eps > 0:
            # Right camera: point behind right screen → should be visible
            ok = ndc_R[0] > -1.0   # should be to the right of left edge
            if not ok:
                print(f"    WARNING: eps={eps:.2f} not visible in right camera")
                all_ok = False

    print(f"\n  Content-plane parallax: {'CONSISTENT' if all_ok else 'INCONSISTENT'}")

    # --- Assertion ---
    assert all_ok, "CONTENT-PLANE PARALLAX INCONSISTENT!"
    print("  ASSERT: content-plane parallax consistency — PASSED")
    return all_ok


def test_stereoscopic_depth(cam, viewer):
    """
    For a point at the fold (0, 0, z), compute depth in view space.
    Both cameras should report the same view-space Z (distance from camera).
    """
    V_L, V_R = cam["V_L"], cam["V_R"]

    print("\n" + "=" * 70)
    print("  TEST 4: View-Space Depth Consistency at Fold")
    print("=" * 70)

    z_samples = np.linspace(-HALF_H, HALF_H, 5)
    all_ok = True

    for z in z_samples:
        pt = np.array([0.0, 0.0, z, 1.0], dtype=np.float64)
        vL = V_L @ pt
        vR = V_R @ pt

        # View-space depth: negative Z in OpenGL convention (camera looks -Z)
        # Both cameras are at the same viewer position, so distance to a
        # fold point should be identical
        depth_L = -vL[2]
        depth_R = -vR[2]
        depth_ok = abs(depth_L - depth_R) < 0.02

        print(f"  Z={z:+.2f}:  depth_L={depth_L:.4f}m  depth_R={depth_R:.4f}m  "
              f"match={'PASS' if depth_ok else 'FAIL'}")

        if not depth_ok:
            all_ok = False

    assert all_ok, "STEREO DEPTH INCONSISTENT AT FOLD!"
    print("\n  ASSERT: view-space depth consistency — PASSED")
    return all_ok


def test_projection_matrix_derived_quantities(cam, near):
    """
    Verify that A_x, A_y (CUDA Jacobian parameters) and offsets are
    correctly derived from l,r,b,t.
    """
    print("\n" + "=" * 70)
    print("  TEST 5: Projection Matrix Derived Quantities")
    print("=" * 70)

    for side, l, r, b, t in [
        ("Left",  cam["l_L"], cam["r_L"], cam["b_L"], cam["t_L"]),
        ("Right", cam["l_R"], cam["r_R"], cam["b_R"], cam["t_R"]),
    ]:
        A_x = 2.0 * near / (r - l)
        A_y = 2.0 * near / (t - b)
        off_x = (r + l) / (r - l)
        off_y = (t + b) / (t - b)

        # Verify reconstruction: given A_x, off_x, near, can we recover l, r?
        denom = 2.0 * near / A_x   # = r - l
        recovered_l = (off_x * denom - denom) / 2.0
        recovered_r = (off_x * denom + denom) / 2.0

        l_ok = abs(recovered_l - l) < 1e-10
        r_ok = abs(recovered_r - r) < 1e-10

        print(f"  {side}: A_x={A_x:.4f} A_y={A_y:.4f} "
              f"off_x={off_x:+.4f} off_y={off_y:+.4f}")
        print(f"         recovered (l, r) = ({recovered_l:.6f}, {recovered_r:.6f}) "
              f"→ l_ok={l_ok} r_ok={r_ok}")

        assert l_ok and r_ok, f"{side}: l,r recovery failed!"

    print("\n  ASSERT: derived quantity recovery — PASSED")
    return True


def test_mirror_projection(cam, viewer, near):
    """
    TEST 6: FLIP_X Mirror Projection

    Verify that FLIP_X (horizontal mirror) correctly mirrors the view:
      1. FLIP_X is self-inverse: FLIP_X @ FLIP_X = I
      2. FLIP_X negates the X component of projected points
      3. FLIP_X + FLIP_Y combined have det=+1 (proper rotation)
      4. Mirroring left camera swaps the sign of frustum l,r bounds
    """
    V_L, V_R = cam["V_L"], cam["V_R"]

    print("\n" + "=" * 70)
    print("  TEST 6: FLIP_X Mirror Projection")
    print("=" * 70)

    all_ok = True

    # 1. Self-inverse: FLIP_X @ FLIP_X = I
    identity = FLIP_X @ FLIP_X
    is_self_inverse = np.allclose(identity, np.eye(4))
    print(f"  FLIP_X self-inverse: {'PASS' if is_self_inverse else 'FAIL'}")
    if not is_self_inverse:
        all_ok = False

    # 2. FLIP_X negates X component of projected points
    pt = np.array([0.5, -0.5, 0.0], dtype=np.float64)
    pt_h = np.append(pt, 1.0)

    v_L_normal = V_L @ pt_h
    V_L_mirrored = FLIP_X @ V_L
    v_L_mirrored = V_L_mirrored @ pt_h

    x_flipped = abs(v_L_normal[0] - (-v_L_mirrored[0])) < 1e-10
    y_preserved = abs(v_L_normal[1] - v_L_mirrored[1]) < 1e-10
    z_preserved = abs(v_L_normal[2] - v_L_mirrored[2]) < 1e-10

    print(f"  X negated: {x_flipped}  Y preserved: {y_preserved}  Z preserved: {z_preserved}")
    if not (x_flipped and y_preserved and z_preserved):
        print(f"    v_normal=({v_L_normal[0]:+.4f}, {v_L_normal[1]:+.4f}, {v_L_normal[2]:+.4f})")
        print(f"    v_mirror=({v_L_mirrored[0]:+.4f}, {v_L_mirrored[1]:+.4f}, {v_L_mirrored[2]:+.4f})")
        all_ok = False

    # 3. FLIP_X + FLIP_Y combined (before any V applied): det = +1
    V_flipxy = FLIP_X @ FLIP_Y
    det_flipxy = np.linalg.det(V_flipxy[:3, :3])
    det_ok = abs(det_flipxy - 1.0) < 0.01
    print(f"  FLIP_X @ FLIP_Y det={det_flipxy:+.1f}: {'PASS (≈+1)' if det_ok else 'FAIL'}")

    # 3b. apply_flip_x to V_L (which already has FLIP_Y): each flip contributes -1
    #     FLIP_X @ V_L = FLIP_X @ FLIP_Y @ V_raw → det = (-1)*(-1)*(+1) = +1
    V_L_mirrored_det = np.linalg.det(apply_flip_x(V_L)[:3, :3])
    det_mir_ok = abs(V_L_mirrored_det - 1.0) < 0.01
    print(f"  apply_flip_x(V_L) det={V_L_mirrored_det:+.1f}: {'PASS (≈+1)' if det_mir_ok else 'FAIL'}")
    if not (det_ok and det_mir_ok):
        all_ok = False

    # 4. Mirroring swaps the sign of frustum bounds
    # Build a mirrored left-camera frustum and check l,r sign flip
    V_L_mir = apply_flip_x(V_L)
    left_corners = cam["left_corners"]
    (l_norm, r_norm, b_norm, t_norm), _ = compute_frustum_bounds(
        viewer, left_corners, near, V_L)
    (l_mir, r_mir, b_mir, t_mir), _ = compute_frustum_bounds(
        viewer, left_corners, near, V_L_mir)

    # Mirroring should swap l,r: l_mir ≈ -r_norm, r_mir ≈ -l_norm
    lr_swap = abs(l_mir + r_norm) < 0.02 and abs(r_mir + l_norm) < 0.02
    tb_preserved = abs(b_mir - b_norm) < 0.01 and abs(t_mir - t_norm) < 0.01
    print(f"  Frustum l,r swap: {lr_swap}  t,b preserved: {tb_preserved}")
    print(f"    Normal:  l={l_norm:.6f} r={r_norm:.6f} b={b_norm:.6f} t={t_norm:.6f}")
    print(f"    Mirrored: l={l_mir:.6f} r={r_mir:.6f} b={b_mir:.6f} t={t_mir:.6f}")
    if not (lr_swap and tb_preserved):
        all_ok = False

    assert all_ok, "FLIP_X MIRROR PROJECTION FAILED!"
    print("\n  ASSERT: mirror projection — PASSED")
    return all_ok


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  PHYSICAL FRUSTUM HARNESS")
    print(f"  Panel: {PANEL_W:.2f} x {PANEL_H:.2f} m")
    print(f"  Viewer: {VIEWER_POS}")
    print(f"  Near clip: {NEAR_CLIP} m")
    print("=" * 70)

    cam = build_dual_frustum(VIEWER_POS, PANEL_W, PANEL_H, NEAR_CLIP, 100.0)

    # Print camera matrices
    for side, V, P in [("LEFT", cam["V_L"], cam["P_L"]),
                        ("RIGHT", cam["V_R"], cam["P_R"])]:
        print(f"\n  {side} View Matrix V (world→camera):")
        for i in range(4):
            print(f"    [{V[i,0]:10.6f} {V[i,1]:10.6f} {V[i,2]:10.6f} {V[i,3]:10.6f}]")
        det = np.linalg.det(V[:3, :3])
        print(f"    det(V_rot) = {det:.1f}")
        right = V[0, :3]
        up = V[1, :3]
        fwd = -V[2, :3]
        print(f"    right (world): [{right[0]:.4f}, {right[1]:.4f}, {right[2]:.4f}]")
        print(f"    up    (world): [{up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f}]")
        print(f"    fwd   (world): [{fwd[0]:.4f}, {fwd[1]:.4f}, {fwd[2]:.4f}]")

    # Run tests
    results = []
    results.append(("Patent Outward-Fold Contract", test_patent_outward_fold_contract(cam, VIEWER_POS)))
    results.append(("Fold Continuity", test_fold_continuity(cam, VIEWER_POS, NEAR_CLIP)))
    results.append(("Frustum Bound Match", test_frustum_bound_match(cam)))
    results.append(("Content-Plane Parallax", test_content_plane_parallax(cam, VIEWER_POS, NEAR_CLIP, 100.0)))
    results.append(("View-Space Depth", test_stereoscopic_depth(cam, VIEWER_POS)))
    results.append(("Derived Quantities", test_projection_matrix_derived_quantities(cam, NEAR_CLIP)))
    results.append(("Mirror Projection", test_mirror_projection(cam, VIEWER_POS, NEAR_CLIP)))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_pass = all(r[1] for r in results)
    for name, ok in results:
        print(f"  [{('PASS' if ok else 'FAIL')}] {name}")
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)

    if not all_pass:
        exit(1)
