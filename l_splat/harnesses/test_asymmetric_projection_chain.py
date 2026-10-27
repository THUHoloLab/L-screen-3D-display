"""
test_asymmetric_projection_chain.py — Full projection chain verification.

Verifies the mathematical correctness of the dual asymmetric off-axis projection
pipeline by tracing the complete projection chain:

  world point → view matrix → asymmetric proj → clip → NDC → pixel → Jacobian

Measures:
  - Reprojection error (round-trip: world→NDC→pixel→world via inverse)
  - Square aspect ratio error (does a world-space square stay square?)
  - Seam continuity error (do fold points map to the same pixel from both cameras?)

Pure numpy — no CUDA/DAV3 needed.
"""

import numpy as np
import math

from l_splat.geometry.frustum import (
    look_at_offscreen,
    FLIP_Y,
    compute_frustum_bounds,
    build_asymmetric_proj_matrix,
    build_l_screen_corners,
)

# ===========================================================================
# Configuration (mirrors pipeline.py and dual_frustum.py)
# ===========================================================================
PANEL_W = 1.28
PANEL_H = 1.44
HALF_H = PANEL_H / 2.0
VIEWER_POS = np.array([2.12, 2.12, 1.60], dtype=np.float64)
NEAR = 0.01
FAR = 100.0
PANEL_PX_W = 832
PANEL_PX_H = 936


def ndc2pix(x_ndc, S):
    """Replicate CUDA ndc2Pix: ((v + 1.0) * S - 1.0) * 0.5"""
    return ((x_ndc + 1.0) * S - 1.0) * 0.5


def pix2ndc(x_pix, S):
    """Inverse of ndc2Pix: v = (2*x_pix + 1) / S - 1"""
    return (2.0 * x_pix + 1.0) / S - 1.0


def build_dual_frustum():
    """Build both frustums using the same math as the main pipeline."""
    left_corners, right_corners = build_l_screen_corners(PANEL_W, PANEL_H)

    left_target = VIEWER_POS + np.array([0.0, -1.0, 0.0])
    V_L = look_at_offscreen(VIEWER_POS, left_target, np.array([0.0, 0.0, 1.0]))
    V_L = FLIP_Y @ V_L
    (l_L, r_L, b_L, t_L), _ = compute_frustum_bounds(VIEWER_POS, left_corners, NEAR, V_L)
    P_L, info_L = build_asymmetric_proj_matrix(NEAR, FAR, l_L, r_L, b_L, t_L)

    right_target = VIEWER_POS + np.array([-1.0, 0.0, 0.0])
    V_R = look_at_offscreen(VIEWER_POS, right_target, np.array([0.0, 0.0, 1.0]))
    V_R = FLIP_Y @ V_R
    (l_R, r_R, b_R, t_R), _ = compute_frustum_bounds(VIEWER_POS, right_corners, NEAR, V_R)
    P_R, info_R = build_asymmetric_proj_matrix(NEAR, FAR, l_R, r_R, b_R, t_R)

    return {
        "V_L": V_L, "P_L": P_L, "info_L": info_L,
        "V_R": V_R, "P_R": P_R, "info_R": info_R,
    }


# ===========================================================================
# Projection chain: world → view → clip → NDC → pixel
# ===========================================================================

def world_to_view(V, points_world):
    """Transform world points to view space. points_world: (N, 3)."""
    N = points_world.shape[0]
    pts_h = np.hstack([points_world, np.ones((N, 1), dtype=np.float64)])
    view = (V @ pts_h.T).T
    return view[:, :3]


def view_to_clip(P, points_view):
    """Transform view points to clip space."""
    N = points_view.shape[0]
    pts_h = np.hstack([points_view, np.ones((N, 1), dtype=np.float64)])
    return (P @ pts_h.T).T


def clip_to_ndc(points_clip):
    """Perspective division: clip → NDC."""
    w = points_clip[:, 3:4]
    return points_clip[:, :3] / np.where(np.abs(w) < 1e-15, 1e-15, w)


def ndc_to_pixel(points_ndc, W_img, H_img):
    """NDC → pixel using CUDA ndc2Pix formula."""
    px = ndc2pix(points_ndc[:, 0], W_img)
    py = ndc2pix(points_ndc[:, 1], H_img)
    return np.stack([px, py, points_ndc[:, 2]], axis=-1)


def project_points(V, P, points_world, W_img, H_img):
    """Full projection: world → pixel. Returns pixel coords, NDC, view_pos."""
    v = world_to_view(V, points_world)
    c = view_to_clip(P, v)
    ndc = clip_to_ndc(c)
    pix = ndc_to_pixel(ndc, W_img, H_img)
    return pix, ndc, v


# ===========================================================================
# Jacobian computation (replicating CUDA computeCov2D logic)
# ===========================================================================

def compute_jacobian_cuda(t, focal_x, focal_y):
    """
    Replicate the CUDA Jacobian from computeCov2D().

    J = [[focal_x/t.z,  0,             -focal_x*t.x/t.z^2],
         [0,             focal_y/t.z,   -focal_y*t.y/t.z^2],
         [0,             0,              0                ]]
    """
    N = t.shape[0]
    J = np.zeros((N, 3, 3), dtype=np.float64)
    tz = t[:, 2]
    tz2 = tz * tz

    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -focal_x * t[:, 0] / tz2
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -focal_y * t[:, 1] / tz2
    return J


def compute_focal_from_asymmetric_proj(P, W, H):
    """focal_x = W*P[0,0]/2, focal_y = H*P[1,1]/2 (pseudo-tanfov convention)."""
    return W * P[0, 0] / 2.0, H * P[1, 1] / 2.0


def compute_jacobian_analytic(V, P, points_world, W_img, H_img):
    """
    Analytic Jacobian of full projection (world→pixel) via chain rule.

    x_pixel = ndc2Pix(-(A_x * tx/tz + offset_x), W)
    dx_pixel/dtx = -W * A_x / (2 * tz)
    dx_pixel/dtz =  W * A_x * tx / (2 * tz^2)

    Returns (N, 2, 3) Jacobian: d(pixel_x, pixel_y) / d(view_x, view_y, view_z).
    """
    t = world_to_view(V, points_world)
    N = t.shape[0]
    J = np.zeros((N, 2, 3), dtype=np.float64)

    A_x = P[0, 0]
    A_y = P[1, 1]
    tz = t[:, 2]
    tz2 = tz * tz

    J[:, 0, 0] = -W_img * A_x / (2.0 * tz)
    J[:, 0, 2] = W_img * A_x * t[:, 0] / (2.0 * tz2)
    J[:, 1, 1] = -H_img * A_y / (2.0 * tz)
    J[:, 1, 2] = H_img * A_y * t[:, 1] / (2.0 * tz2)

    return J


# ===========================================================================
# 1. World-space square test
# ===========================================================================

def test_world_square(cam):
    """Project squares on (a) panel surface and (b) 45-degree plane.

    Panel-surface square should be ~isotropic (aspect ~ 1.0).
    45-degree plane square measures inherent anamorphic distortion.
    """
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("=" * 70)
    print("  TEST A: World-Space Square Projection")
    print("=" * 70)

    results = {}

    # --- A1: Square on panel surface (should be ~isotropic) ---
    # Left panel: constant Y = 0, X in [-1.28, 0], Z in [-0.72, 0.72]
    # Center at X = -0.64, Z = 0.0, Y = 0
    half_panel = 0.15
    center_L = np.array([-0.64, 0.0, 0.0])
    # World X is "horizontal" for left panel, Z is vertical
    corners_L = np.array([
        center_L + [-half_panel, 0, -half_panel],
        center_L + [+half_panel, 0, -half_panel],
        center_L + [+half_panel, 0, +half_panel],
        center_L + [-half_panel, 0, +half_panel],
    ], dtype=np.float64)

    # Right panel: constant X = 0, Y in [-1.28, 0], Z in [-0.72, 0.72]
    center_R = np.array([0.0, -0.64, 0.0])
    corners_R = np.array([
        center_R + [0, -half_panel, -half_panel],
        center_R + [0, +half_panel, -half_panel],
        center_R + [0, +half_panel, +half_panel],
        center_R + [0, -half_panel, +half_panel],
    ], dtype=np.float64)

    panel_aspects = []
    for cam_name, V, P, corners in [
        ("Left (panel surf)",  V_L, P_L, corners_L),
        ("Right (panel surf)", V_R, P_R, corners_R),
    ]:
        pix, ndc, _ = project_points(V, P, corners, PANEL_PX_W, PANEL_PX_H)
        edges = [np.linalg.norm(pix[i, :2] - pix[(i+1)%4, :2]) for i in range(4)]
        h = (edges[0] + edges[2]) / 2.0
        v = (edges[1] + edges[3]) / 2.0
        aspect = h / max(v, 0.01)
        err = abs(aspect - 1.0)
        panel_aspects.append(err)
        in_f = np.all((ndc[:,0]>=-1)&(ndc[:,0]<=1)&(ndc[:,1]>=-1)&(ndc[:,1]<=1))
        print(f"  {cam_name}: H={h:.1f}px V={v:.1f}px aspect={aspect:.4f} "
              f"err={err:.4f} in_f={in_f} [{'PASS' if err<0.05 else 'MEAS'}]]")

    # --- A2: Square on 45-degree plane (measures anamorphic distortion) ---
    half_45 = 0.12
    inv_sqrt2 = 1.0 / math.sqrt(2)
    # Center on left half: u=0.3 (away from fold to stay in left frustum)
    u_c = 0.30
    c45 = np.array([-PANEL_W*(1-u_c), -PANEL_W*u_c, 0.0])
    tangent = np.array([-1.0, 1.0, 0.0]) * inv_sqrt2
    corners_45 = np.array([
        c45 - half_45 * tangent - half_45 * np.array([0,0,1]),
        c45 + half_45 * tangent - half_45 * np.array([0,0,1]),
        c45 + half_45 * tangent + half_45 * np.array([0,0,1]),
        c45 - half_45 * tangent + half_45 * np.array([0,0,1]),
    ], dtype=np.float64)

    for cam_name, V, P in [("Left (45deg plane)", V_L, P_L), ("Right (45deg plane)", V_R, P_R)]:
        pix, ndc, _ = project_points(V, P, corners_45, PANEL_PX_W, PANEL_PX_H)
        edges = [np.linalg.norm(pix[i, :2] - pix[(i+1)%4, :2]) for i in range(4)]
        h = (edges[0] + edges[2]) / 2.0
        v = (edges[1] + edges[3]) / 2.0
        aspect = h / max(v, 0.01)
        err = abs(aspect - 1.0)
        in_f = np.all((ndc[:,0]>=-1)&(ndc[:,0]<=1)&(ndc[:,1]>=-1)&(ndc[:,1]<=1))
        print(f"  {cam_name}: H={h:.1f}px V={v:.1f}px aspect={aspect:.4f} "
              f"err={err:.4f} in_f={in_f} [MEAS]")

    # Panel-surface squares should be nearly isotropic (test passes if so)
    panel_aspect_ok = all(e < 0.05 for e in panel_aspects)
    avg_aspect_err = np.mean(panel_aspects)
    print(f"\n  Panel-surface aspect error: {avg_aspect_err:.6f} "
          f"[{'PASS' if panel_aspect_ok else 'FAIL'}]")

    results = {"panel_aspect_err": avg_aspect_err, "panel_ok": panel_aspect_ok}
    return panel_aspect_ok, results


# ===========================================================================
# 2. World-space grid test
# ===========================================================================

def test_world_grid(cam):
    """Project an 11x11 grid. Measure cell size uniformity and max distortion."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST B: World-Space Grid Projection")
    print("=" * 70)

    grid_n = 11
    u_vals = np.linspace(0.0, 1.0, grid_n)
    z_vals = np.linspace(-HALF_H, HALF_H, grid_n)

    all_pass = True
    all_results = {}

    for cam_name, V, P, side in [
        ("Left", V_L, P_L, "left"),
        ("Right", V_R, P_R, "right"),
    ]:
        cells_w = []
        cells_h = []

        for i in range(grid_n - 1):
            for j in range(grid_n - 1):
                u0, u1 = u_vals[i], u_vals[i + 1]
                z0, z1 = z_vals[j], z_vals[j + 1]

                corners = np.array([
                    [-PANEL_W * (1.0 - u0), -PANEL_W * u0, z0],
                    [-PANEL_W * (1.0 - u1), -PANEL_W * u1, z0],
                    [-PANEL_W * (1.0 - u1), -PANEL_W * u1, z1],
                    [-PANEL_W * (1.0 - u0), -PANEL_W * u0, z1],
                ], dtype=np.float64)

                pix, ndc, _ = project_points(V, P, corners, PANEL_PX_W, PANEL_PX_H)

                if not np.all((ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
                              (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)):
                    continue

                w = (np.linalg.norm(pix[1, :2] - pix[0, :2]) +
                     np.linalg.norm(pix[2, :2] - pix[3, :2])) / 2.0
                h = (np.linalg.norm(pix[3, :2] - pix[0, :2]) +
                     np.linalg.norm(pix[2, :2] - pix[1, :2])) / 2.0
                cells_w.append(w)
                cells_h.append(h)

        if len(cells_w) == 0:
            print(f"  {cam_name}: No cells in frustum!")
            all_pass = False
            continue

        cells_w = np.array(cells_w)
        cells_h = np.array(cells_h)

        w_cv = cells_w.std() / max(cells_w.mean(), 0.01)
        h_cv = cells_h.std() / max(cells_h.mean(), 0.01)
        max_distort = max(cells_w.max() / max(cells_w.min(), 0.01),
                          cells_h.max() / max(cells_h.min(), 0.01))

        ok = w_cv < 0.30 and h_cv < 0.30
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False

        print(f"  {cam_name} ({len(cells_w)} cells): "
              f"W CV={w_cv:.4f} H CV={h_cv:.4f} "
              f"max_distort={max_distort:.3f}x [{status}]")

        all_results[side] = {"w_cv": w_cv, "h_cv": h_cv, "max_distort": max_distort}

    if all_results:
        avg_distort = np.mean([r["max_distort"] for r in all_results.values()])
        print(f"\n  Avg max distortion: {avg_distort:.3f}x (target: < 3.0x)")
    return all_pass, all_results


# ===========================================================================
# 3. World-space cube test
# ===========================================================================

def test_world_cube(cam):
    """Project a small cube on the panel surface. Check visibility and winding."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST C: World-Space Cube Projection (panel surface)")
    print("=" * 70)

    r = 0.12
    # Left-camera cube: centered on left panel surface
    cx, cy, cz = -0.64, 0.0, 0.0
    corners = np.array([
        [cx-r, cy-r, cz-r], [cx+r, cy-r, cz-r],
        [cx+r, cy+r, cz-r], [cx-r, cy+r, cz-r],
        [cx-r, cy-r, cz+r], [cx+r, cy-r, cz+r],
        [cx+r, cy+r, cz+r], [cx-r, cy+r, cz+r],
    ], dtype=np.float64)

    # Right-camera cube: centered on right panel surface
    cor_R = np.array([
        [-r, -0.64-r, cz-r], [+r, -0.64-r, cz-r],
        [+r, -0.64+r, cz-r], [-r, -0.64+r, cz-r],
        [-r, -0.64-r, cz+r], [+r, -0.64-r, cz+r],
        [+r, -0.64+r, cz+r], [-r, -0.64+r, cz+r],
    ], dtype=np.float64)

    faces = [
        [0,1,2,3], [4,5,6,7], [0,1,5,4],
        [2,3,7,6], [0,3,7,4], [1,2,6,5],
    ]

    all_pass = True

    for cam_name, V, P, c in [
        ("Left",  V_L, P_L, corners),
        ("Right", V_R, P_R, cor_R),
    ]:
        pix, ndc, v = project_points(V, P, c, PANEL_PX_W, PANEL_PX_H)

        in_frustum = np.all((ndc[:,0]>=-1)&(ndc[:,0]<=1)&
                            (ndc[:,1]>=-1)&(ndc[:,1]<=1))

        n_front = 0
        for fi in faces:
            p0, p1, p2 = pix[fi[0],:2], pix[fi[1],:2], pix[fi[2],:2]
            area = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
            if area > 0:
                n_front += 1

        ok = in_frustum and n_front >= 1  # off-axis view: most faces away
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        zr = v[:,2]
        print(f"  {cam_name}: in_frustum={in_frustum} front={n_front}/6 "
              f"z=[{zr.min():.3f},{zr.max():.3f}] "
              f"ndc_x=[{ndc[:,0].min():.3f},{ndc[:,0].max():.3f}] [{status}]")

    print(f"\n  Cube visibility: {'PASS' if all_pass else 'FAIL'}")
    return all_pass, {}


# ===========================================================================
# 4. Seam-crossing geometry test
# ===========================================================================

def test_seam_continuity(cam):
    """Verify fold points (X=0,Y=0) map to correct NDC edges from both cameras."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST D: Seam Continuity at Fold (X=0, Y=0)")
    print("=" * 70)

    z_samples = np.linspace(-HALF_H, HALF_H, 9)
    seam_pts = np.column_stack([np.zeros(9), np.zeros(9), z_samples])

    _, ndc_L, _ = project_points(V_L, P_L, seam_pts, PANEL_PX_W, PANEL_PX_H)
    _, ndc_R, _ = project_points(V_R, P_R, seam_pts, PANEL_PX_W, PANEL_PX_H)

    # Fold at: L x_ndc≈+1, R x_ndc≈-1
    xL_err = np.abs(ndc_L[:, 0] - 1.0).max()
    xR_err = np.abs(ndc_R[:, 0] - (-1.0)).max()
    y_mm = np.abs(ndc_L[:, 1] - ndc_R[:, 1]).max()
    z_mm = np.abs(ndc_L[:, 2] - ndc_R[:, 2]).max()

    xL_ok = xL_err < 0.02
    xR_ok = xR_err < 0.02
    y_ok = y_mm < 0.02
    z_ok = z_mm < 0.05

    print(f"  Fold at Z=0: L_NDC=({ndc_L[4,0]:.4f},{ndc_L[4,1]:.4f}) "
          f"R_NDC=({ndc_R[4,0]:.4f},{ndc_R[4,1]:.4f})")
    print(f"  Max errors: xL={xL_err:.6f} xR={xR_err:.6f} "
          f"y_mm={y_mm:.6f} z_mm={z_mm:.6f}")
    print(f"  xL={xL_ok} xR={xR_ok} y={y_ok} z={z_ok} "
          f"[{'PASS' if (xL_ok and xR_ok and y_ok and z_ok) else 'FAIL'}]")

    # Extended: 21-point scan
    z_dense = np.linspace(-HALF_H, HALF_H, 21)
    dense_pts = np.column_stack([np.zeros(21), np.zeros(21), z_dense])
    _, ndc_Ld, _ = project_points(V_L, P_L, dense_pts, PANEL_PX_W, PANEL_PX_H)
    _, ndc_Rd, _ = project_points(V_R, P_R, dense_pts, PANEL_PX_W, PANEL_PX_H)

    pxL = ndc2pix(ndc_Ld[:, 0], PANEL_PX_W)
    pxR = ndc2pix(ndc_Rd[:, 0], PANEL_PX_W)
    pyL = ndc2pix(ndc_Ld[:, 1], PANEL_PX_H)
    pyR = ndc2pix(ndc_Rd[:, 1], PANEL_PX_H)

    print(f"\n  Extended scan (21 pts):")
    print(f"    Left fold pixel X: {pxL.mean():.2f} (target: {PANEL_PX_W-1:.0f})")
    print(f"    Right fold pixel X: {pxR.mean():.2f} (target: 0.0)")
    print(f"    Max Y pixel diff: {np.abs(pyL - pyR).max():.2f}px")

    results = {
        "xL_ndc_err": xL_err,
        "xR_ndc_err": xR_err,
        "y_mm_ndc": y_mm,
        "z_mm_ndc": z_mm,
        "fold_px_L": pxL.mean(),
        "fold_px_R": pxR.mean(),
        "y_px_diff": np.abs(pyL - pyR).max(),
    }

    seam_pass = xL_ok and xR_ok and y_ok and z_ok
    return seam_pass, results


# ===========================================================================
# 5. Jacobian verification
# ===========================================================================

def test_jacobian_consistency(cam):
    """Verify CUDA Jacobian matches analytic asymmetric projection Jacobian."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST E: Jacobian Consistency (CUDA vs Analytic)")
    print("=" * 70)

    inv_sqrt2 = 1.0 / math.sqrt(2)
    n_pts = 10
    u_vals = np.linspace(0.0, 1.0, n_pts)
    z_vals = np.linspace(-HALF_H, HALF_H, n_pts)
    depth_vals = [0.0, -0.18, 0.12]

    points = []
    for dp in depth_vals:
        for i in range(n_pts):
            for j in range(n_pts):
                wx = -PANEL_W * (1.0 - u_vals[i]) + dp * inv_sqrt2
                wy = -PANEL_W * u_vals[i] + dp * inv_sqrt2
                points.append([wx, wy, z_vals[j]])

    points = np.array(points[:200], dtype=np.float64)

    all_pass = True
    max_rel_err = 0.0

    for cam_name, V, P in [("Left", V_L, P_L), ("Right", V_R, P_R)]:
        fx, fy = compute_focal_from_asymmetric_proj(P, PANEL_PX_W, PANEL_PX_H)
        t = world_to_view(V, points)
        J_cuda = compute_jacobian_cuda(t, fx, fy)
        J_analytic = compute_jacobian_analytic(
            V, P, points, PANEL_PX_W, PANEL_PX_H)

        # CUDA and analytic Jacobians differ by a sign convention:
        #   J_cuda[0,0] = +focal_x/tz
        #   J_analytic[0,0] = -W*A_x/(2*tz) = -focal_x/tz
        # The sign cancels in cov2D = J^T * cov3D * J, so we compare
        # absolute values.
        comparisons = [
            ("dx/dtx", J_cuda[:, 0, 0], J_analytic[:, 0, 0]),
            ("dx/dtz", J_cuda[:, 0, 2], J_analytic[:, 0, 2]),
            ("dy/dty", J_cuda[:, 1, 1], J_analytic[:, 1, 1]),
            ("dy/dtz", J_cuda[:, 1, 2], J_analytic[:, 1, 2]),
        ]

        for label, vc, va in comparisons:
            denom = np.mean(np.abs(va)) + 0.001
            rel_err = np.max(np.abs(np.abs(vc) - np.abs(va))) / denom
            max_rel_err = max(max_rel_err, rel_err)
            if rel_err > 0.01:
                all_pass = False

        print(f"  {cam_name}: A_x={P[0,0]:.4f} A_y={P[1,1]:.4f} "
              f"fx={fx:.1f} fy={fy:.1f} ratio={fx/fy:.4f}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  Jacobian consistency: {status} (max_rel_err={max_rel_err:.6f})")
    return all_pass, {"max_rel_err": max_rel_err}


# ===========================================================================
# 6. Near-plane culling analysis
# ===========================================================================

def test_culling_analysis(cam):
    """Check if the 1.0m near-plane (fabsf(p_view.z) < 1.0f) culls valid content."""
    V_L = cam["V_L"]
    V_R = cam["V_R"]

    print("\n" + "=" * 70)
    print("  TEST F: Near-Plane Culling Analysis")
    print("=" * 70)

    inv_sqrt2 = 1.0 / math.sqrt(2)
    n_s = 20
    u_vals = np.linspace(0.0, 1.0, n_s)
    z_vals = np.linspace(-HALF_H, HALF_H, n_s)
    depth_vals = np.linspace(-0.18, 0.12, 5)

    for cam_name, V in [("Left", V_L), ("Right", V_R)]:
        n_culled = 0
        n_total = 0
        z_min = float('inf')
        z_max = float('-inf')

        for dp in depth_vals:
            for i in range(0, n_s, 2):
                for j in range(0, n_s, 2):
                    wx = -PANEL_W * (1.0 - u_vals[i]) + dp * inv_sqrt2
                    wy = -PANEL_W * u_vals[i] + dp * inv_sqrt2
                    t = world_to_view(V, np.array([[wx, wy, z_vals[j]]]))
                    tz = abs(t[0, 2])
                    z_min = min(z_min, tz)
                    z_max = max(z_max, tz)
                    n_total += 1
                    if tz < 1.0:
                        n_culled += 1

        pct = 100.0 * n_culled / n_total
        print(f"  {cam_name}: view_z=[{z_min:.3f},{z_max:.3f}]m "
              f"culled={n_culled}/{n_total} ({pct:.1f}%) "
              f"[{'WARN' if n_culled > 0 else 'OK'}]")

    return True, {}


# ===========================================================================
# 7. Large world-space square (1m x 1m)
# ===========================================================================

def test_large_square(cam):
    """Project a 1m x 1m square on the panel surface for each camera."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST G: Large World-Space Square (1.0m x 1.0m)")
    print("=" * 70)

    half = 0.5
    all_pass = True
    results = {}

    for cam_name, V, P, center_xyz, orient in [
        ("Left panel surface",  V_L, P_L, np.array([-0.64, 0.0, 0.0]), "XZ"),
        ("Right panel surface", V_R, P_R, np.array([0.0, -0.64, 0.0]), "YZ"),
    ]:
        cx, cy, cz = center_xyz
        # Left panel (Y=0): square in XZ plane (vary X and Z)
        # Right panel (X=0): square in YZ plane (vary Y and Z)
        if orient == "XZ":
            corners = np.array([
                [cx - half, cy, cz - half],
                [cx + half, cy, cz - half],
                [cx + half, cy, cz + half],
                [cx - half, cy, cz + half],
            ], dtype=np.float64)
        else:  # YZ
            corners = np.array([
                [cx, cy - half, cz - half],
                [cx, cy + half, cz - half],
                [cx, cy + half, cz + half],
                [cx, cy - half, cz + half],
            ], dtype=np.float64)

        pix, ndc, _ = project_points(V, P, corners, PANEL_PX_W, PANEL_PX_H)
        edges = [np.linalg.norm(pix[i, :2] - pix[(i+1) % 4, :2]) for i in range(4)]
        h = (edges[0] + edges[2]) / 2.0
        v = (edges[1] + edges[3]) / 2.0
        aspect = h / max(v, 0.01)
        in_f = np.all((ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1))
        ok = in_f and abs(aspect - 1.0) < 0.05
        if not ok:
            all_pass = False
        print(f"  {cam_name}: H={h:.1f}px V={v:.1f}px aspect={aspect:.4f} "
              f"in_frustum={in_f} [{'PASS' if ok else 'MEAS'}]")
        results[cam_name] = {"width_px": h, "height_px": v, "aspect": aspect, "in_frustum": in_f}

    print(f"\n  Large square: {'PASS' if all_pass else 'MEAS'}")
    return all_pass, results


# ===========================================================================
# 8. Cube crossing fold seam
# ===========================================================================

def test_cube_crossing_fold(cam):
    """Place a cube centered at the fold (X=0, Y=0) and verify both cameras see it."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST H: Cube Crossing Fold Seam")
    print("=" * 70)

    r = 0.15
    corners = np.array([
        [-r, -r, -r], [+r, -r, -r], [+r, +r, -r], [-r, +r, -r],
        [-r, -r, +r], [+r, -r, +r], [+r, +r, +r], [-r, +r, +r],
    ], dtype=np.float64)

    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
    ]

    all_pass = True

    for cam_name, V, P in [("Left", V_L, P_L), ("Right", V_R, P_R)]:
        pix, ndc, v = project_points(V, P, corners, PANEL_PX_W, PANEL_PX_H)

        in_f = np.all((ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1))
        n_in = int(np.sum((ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)))

        n_front = 0
        for fi in faces:
            p0, p1, p2 = pix[fi[0], :2], pix[fi[1], :2], pix[fi[2], :2]
            area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if area > 0:
                n_front += 1

        # Fold-crossing: at least some corners should be in frustum
        ok = n_in >= 2
        if not ok:
            all_pass = False
        print(f"  {cam_name}: {n_in}/8 corners in frustum, {n_front} front faces "
              f"z_view=[{v[:,2].min():.2f},{v[:,2].max():.2f}] "
              f"[{'PASS' if ok else 'FAIL'}]")

    # Check both cameras see some part of the cube
    print(f"\n  Cube fold crossing: {'PASS' if all_pass else 'FAIL'}")
    return all_pass, {}


# ===========================================================================
# 9. Points on panel surfaces
# ===========================================================================

def test_panel_surface_points(cam):
    """Verify points on exact panel surfaces project correctly."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST I: Panel Surface Points (left, right, fold edge)")
    print("=" * 70)

    all_pass = True
    results = {}

    # --- Left panel surface (Y=0) ---
    n_pts = 7
    x_vals = np.linspace(-PANEL_W, 0.0, n_pts)
    z_vals = np.linspace(-HALF_H, HALF_H, n_pts)
    left_pts = np.array([[-PANEL_W/2, 0.0, 0.0],
                         [-0.01, 0.0, 0.0],     # near fold
                         [-PANEL_W, 0.0, 0.0],   # far edge
                         [-0.64, 0.0, +0.72],    # top
                         [-0.64, 0.0, -0.72],    # bottom
                         ], dtype=np.float64)

    pix_L, ndc_L, _ = project_points(V_L, P_L, left_pts, PANEL_PX_W, PANEL_PX_H)
    in_f_L = (ndc_L[:, 0] >= -1) & (ndc_L[:, 0] <= 1) & (ndc_L[:, 1] >= -1) & (ndc_L[:, 1] <= 1)
    n_in_L = int(in_f_L.sum())

    print(f"  Left panel (Y=0) points: {n_in_L}/{len(left_pts)} in left frustum")
    for i, pt in enumerate(left_pts):
        print(f"    ({pt[0]:+.3f},{pt[1]:+.3f},{pt[2]:+.3f}) "
              f"→ NDC=({ndc_L[i,0]:+.4f},{ndc_L[i,1]:+.4f}) pix=({pix_L[i,0]:.1f},{pix_L[i,1]:.1f}) "
              f"{'IN' if in_f_L[i] else 'OUT'}")

    # Near-fold point should be near right edge of left panel
    fold_pt_ndc_x = ndc_L[1, 0]  # point at X=-0.01
    fold_ok = fold_pt_ndc_x > 0.9  # near NDC right edge
    if not fold_ok:
        all_pass = False

    # --- Right panel surface (X=0) ---
    right_pts = np.array([[0.0, -PANEL_W/2, 0.0],
                          [0.0, -0.01, 0.0],     # near fold
                          [0.0, -PANEL_W, 0.0],   # far edge
                          ], dtype=np.float64)

    pix_R, ndc_R, _ = project_points(V_R, P_R, right_pts, PANEL_PX_W, PANEL_PX_H)
    in_f_R = (ndc_R[:, 0] >= -1) & (ndc_R[:, 0] <= 1) & (ndc_R[:, 1] >= -1) & (ndc_R[:, 1] <= 1)
    n_in_R = int(in_f_R.sum())

    print(f"  Right panel (X=0) points: {n_in_R}/{len(right_pts)} in right frustum")
    for i, pt in enumerate(right_pts):
        print(f"    ({pt[0]:+.3f},{pt[1]:+.3f},{pt[2]:+.3f}) "
              f"→ NDC=({ndc_R[i,0]:+.4f},{ndc_R[i,1]:+.4f}) pix=({pix_R[i,0]:.1f},{pix_R[i,1]:.1f}) "
              f"{'IN' if in_f_R[i] else 'OUT'}")

    fold_pt_ndc_x_R = ndc_R[1, 0]  # point at Y=-0.01
    fold_ok_R = fold_pt_ndc_x_R < -0.9  # near NDC left edge (fold is at x=-1 for right cam)
    if not fold_ok_R:
        all_pass = False

    # --- Fold edge (X=0, Y=0) ---
    fold_pts = np.array([[0.0, 0.0, -HALF_H],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, +HALF_H]], dtype=np.float64)
    _, ndc_fL, _ = project_points(V_L, P_L, fold_pts, PANEL_PX_W, PANEL_PX_H)
    _, ndc_fR, _ = project_points(V_R, P_R, fold_pts, PANEL_PX_W, PANEL_PX_H)

    # Fold should be at x_ndc=+1 for left, x_ndc=-1 for right
    xL_ok = np.all(np.abs(ndc_fL[:, 0] - 1.0) < 0.02)
    xR_ok = np.all(np.abs(ndc_fR[:, 0] - (-1.0)) < 0.02)

    print(f"  Fold edge: L_x_ndc={ndc_fL[:,0]} ({'OK' if xL_ok else 'FAIL'})")
    print(f"  Fold edge: R_x_ndc={ndc_fR[:,0]} ({'OK' if xR_ok else 'FAIL'})")

    if not (xL_ok and xR_ok):
        all_pass = False

    print(f"\n  Panel surface points: {'PASS' if all_pass else 'FAIL'}")
    return all_pass, {"fold_L_ok": xL_ok, "fold_R_ok": xR_ok, "n_left_in": n_in_L, "n_right_in": n_in_R}


# ===========================================================================
# 10. Reprojection error (round-trip)
# ===========================================================================

def test_reprojection_error(cam):
    """Measure round-trip error: world → NDC → pixel → world via inverse projection."""
    V_L, P_L = cam["V_L"], cam["P_L"]
    V_R, P_R = cam["V_R"], cam["P_R"]

    print("\n" + "=" * 70)
    print("  TEST J: Reprojection Error (Round-Trip)")
    print("=" * 70)

    all_pass = True
    results = {}

    n_pts = 15
    u_vals = np.linspace(0.05, 0.95, n_pts)
    z_vals = np.linspace(-HALF_H * 0.8, HALF_H * 0.8, n_pts)
    depths = [0.0, -0.18, 0.12]
    inv_sqrt2 = 1.0 / math.sqrt(2)

    world_pts = []
    for dp in depths:
        for u in u_vals[::3]:
            for z in z_vals[::3]:
                wx = -PANEL_W * (1.0 - u) + dp * inv_sqrt2
                wy = -PANEL_W * u + dp * inv_sqrt2
                world_pts.append([wx, wy, z])
    world_pts = np.array(world_pts[:100], dtype=np.float64)

    for cam_name, V, P in [("Left", V_L, P_L), ("Right", V_R, P_R)]:
        pix, ndc, view_pos = project_points(V, P, world_pts, PANEL_PX_W, PANEL_PX_H)

        # Forward projection outputs: pix, ndc
        # Reverse: ndc → view direction → intersect with known-depth plane
        # Given tz = view_pos[:, 2], recover the view direction

        # Inverse NDC → view space (recover tx, ty up to scale)
        A_x, A_y = P[0, 0], P[1, 1]
        offset_x, offset_y = P[0, 2], P[1, 2]

        # From ndc: ndc.x = -(A_x * tx/tz + offset_x)
        # → tx = -tz * (ndc.x + offset_x) / A_x
        tx_recovered = -view_pos[:, 2] * (ndc[:, 0] + offset_x) / A_x
        ty_recovered = -view_pos[:, 2] * (ndc[:, 1] + offset_y) / A_y

        view_recovered = np.stack([tx_recovered, ty_recovered, view_pos[:, 2]], axis=-1)

        # Inverse view → world via V^{-1}
        V_inv = np.linalg.inv(V)
        N = world_pts.shape[0]
        view_h = np.hstack([view_recovered, np.ones((N, 1))])
        world_recovered = (V_inv @ view_h.T).T[:, :3]

        errors = np.linalg.norm(world_pts - world_recovered, axis=1)
        max_err = errors.max()
        mean_err = errors.mean()

        ok = max_err < 0.01
        if not ok:
            all_pass = False

        print(f"  {cam_name}: {N} points, mean_err={mean_err:.6f}m, "
              f"max_err={max_err:.6f}m [{'PASS' if ok else 'WARN'}]")

        results[cam_name] = {"mean_err": mean_err, "max_err": max_err}

    # Also verify using pixel → NDC → view ray → intersect panel surface
    # For left panel: point projects to pixel, then ray from viewer through pixel
    # should intersect the panel surface (Y=0) at the correct world position
    px_test = np.array([PANEL_PX_W // 4, PANEL_PX_W // 2, 3 * PANEL_PX_W // 4])
    py_test = np.array([PANEL_PX_H // 2, PANEL_PX_H // 2, PANEL_PX_H // 2])
    pxg, pyg = np.meshgrid(px_test, py_test)
    pix_test = np.stack([pxg.ravel(), pyg.ravel()], axis=-1).astype(np.float64)

    for cam_name, V, P, side in [
        ("Left", V_L, P_L, "left"),
        ("Right", V_R, P_R, "right"),
    ]:
        ndc_x = pix2ndc(pix_test[:, 0], PANEL_PX_W)
        ndc_y = pix2ndc(pix_test[:, 1], PANEL_PX_H)

        A_x, A_y = P[0, 0], P[1, 1]
        offset_x, offset_y = P[0, 2], P[1, 2]

        # Ray direction in view space (from pixel + ndc)
        dir_x = (ndc_x + offset_x) / A_x
        dir_y = (ndc_y + offset_y) / A_y

        # Intersect with panel surface in world space
        # Ray from viewer in world: viewer + t * world_dir
        V_inv = np.linalg.inv(V)
        max_ray_err = 0.0

        for k in range(len(pix_test)):
            dir_view = np.array([dir_x[k], dir_y[k], -1.0])
            dir_view = dir_view / np.linalg.norm(dir_view)
            dir_h = np.array([dir_view[0], dir_view[1], dir_view[2], 0.0])
            dir_world = (V_inv @ dir_h)[:3]

            eye = VIEWER_POS.copy()
            if side == "left":
                t = -eye[1] / dir_world[1]
            else:
                t = -eye[0] / dir_world[0]

            hit = eye + t * dir_world

            # For left panel: Y should be ~0, X in [-1.28, 0]
            # For right panel: X should be ~0, Y in [-1.28, 0]
            if side == "left":
                ray_err = abs(hit[1])
                valid = (hit[0] >= -PANEL_W - 0.01) and (hit[0] <= 0.01)
            else:
                ray_err = abs(hit[0])
                valid = (hit[1] >= -PANEL_W - 0.01) and (hit[1] <= 0.01)
            max_ray_err = max(max_ray_err, ray_err)

        ok = max_ray_err < 0.01
        if not ok:
            all_pass = False
        print(f"  {cam_name} ray intersection ({side} panel): "
              f"max_plane_error={max_ray_err:.6f}m [{'PASS' if ok else 'WARN'}]")

    print(f"\n  Reprojection: {'PASS' if all_pass else 'WARN'}")
    return all_pass, results


# ===========================================================================
# Main harness
# ===========================================================================

def harness():
    print("=" * 70)
    print("  HARNESS: Asymmetric Projection Chain Verification")
    print("=" * 70)
    print(f"  Panel: {PANEL_W:.2f}x{PANEL_H:.2f}m, px: {PANEL_PX_W}x{PANEL_PX_H}")
    print(f"  Viewer: {VIEWER_POS}")
    print(f"  Near: {NEAR}m")
    print()

    cam = build_dual_frustum()

    for side, info in [("Left", cam["info_L"]), ("Right", cam["info_R"])]:
        print(f"  {side}: l={info['l']:.6f} r={info['r']:.6f} "
              f"b={info['b']:.6f} t={info['t']:.6f}")
        print(f"         A_x={info['A_x']:.4f} A_y={info['A_y']:.4f} "
              f"off_x={info['offset_x']:+.4f} off_y={info['offset_y']:+.4f}")
        rl = info['r'] - info['l']
        tb = info['t'] - info['b']
        print(f"         r-l={rl:.6f} t-b={tb:.6f} "
              f"2n/(r-l)={2*NEAR/rl:.4f} 2n/(t-b)={2*NEAR/tb:.4f}")

    results = []
    metrics = {}

    tests = [
        ("A. World-Space Square", test_world_square),
        ("B. World-Space Grid", test_world_grid),
        ("C. World-Space Cube", test_world_cube),
        ("D. Seam Continuity", test_seam_continuity),
        ("E. Jacobian Consistency", test_jacobian_consistency),
        ("F. Near-Plane Culling", test_culling_analysis),
        ("G. Large Square (1m x 1m)", test_large_square),
        ("H. Cube Crossing Fold", test_cube_crossing_fold),
        ("I. Panel Surface Points", test_panel_surface_points),
        ("J. Reprojection Error", test_reprojection_error),
    ]

    for name, test_fn in tests:
        try:
            ok, m = test_fn(cam)
            results.append((name, ok))
            metrics[name] = m
        except Exception as e:
            print(f"\n  [{name}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            metrics[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        s = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{s}] {name}")

    if "D. Seam Continuity" in metrics:
        m = metrics["D. Seam Continuity"]
        print(f"\n  SEAM METRICS:")
        print(f"    X_NDC error: L={m.get('xL_ndc_err','?'):.6f} "
              f"R={m.get('xR_ndc_err','?'):.6f}")
        print(f"    Y_NDC mismatch: {m.get('y_mm_ndc','?'):.6f}")
        print(f"    Fold pixel: L={m.get('fold_px_L','?'):.2f} "
              f"R={m.get('fold_px_R','?'):.2f}")
        print(f"    Y pixel diff: {m.get('y_px_diff','?'):.2f}")

    if "E. Jacobian Consistency" in metrics:
        m = metrics["E. Jacobian Consistency"]
        print(f"\n  JACOBIAN METRICS:")
        print(f"    Max relative error: {m.get('max_rel_err','?'):.6f}")

    if "A. World-Space Square" in metrics:
        m = metrics["A. World-Space Square"]
        ae = m.get('panel_aspect_err', None)
        print(f"\n  SQUARE METRICS:")
        if ae is not None:
            print(f"    Panel-surface aspect error: {ae:.6f}")
        print(f"    Panel-surface passed: {m.get('panel_ok', '?')}")

    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return all_pass, metrics


if __name__ == "__main__":
    ok, _ = harness()
    import sys
    sys.exit(0 if ok else 1)
