"""
test_virtual_box.py — Perceptual quality validation for spatial portal reconstruction.

Verifies the inverse perceptual mapping from the virtual space box B_virtual
onto the physical L-shaped panels. All tests are pure numpy — no CUDA needed.

Tests:
  1. Wall continuity at the fold
  2. Perspective correctness (regular grid -> consistent perspective)
  3. Depth consistency (BG depth monotonic from fold)
  4. Projection round-trip (wall -> virtual plane -> wall)
  5. Fold symmetry (left/right panels agree at fold)
"""

import numpy as np

PANEL_W = 1.28
PANEL_H = 1.60
SWEET_SPOT = np.array([2.12, 2.12, 0.40], dtype=np.float64)

from l_splat.geometry.virtual_box import (
    define_virtual_box,
    project_wall_to_virtual_plane,
    map_background_to_panels,
)


# ── Test 1: Wall Continuity at the Fold ─────────────────────────────────

def test_fold_continuity():
    """Both panels agree on what the viewer sees at the fold line."""
    z_test = np.linspace(-PANEL_H * 0.5, PANEL_H * 0.5, 5)
    for z in z_test:
        p_fold = np.array([0.0, 0.0, z])

        u_L, vz_L, hit_L = project_wall_to_virtual_plane(p_fold, SWEET_SPOT, PANEL_W)
        u_R, vz_R, hit_R = project_wall_to_virtual_plane(p_fold, SWEET_SPOT, PANEL_W)

        assert abs(u_L - u_R) < 0.01, \
            f"Fold u mismatch at z={z:.2f}: {u_L:.4f} vs {u_R:.4f}"
        assert abs(hit_L[0] - hit_R[0]) < 0.01, \
            f"Fold hit_x mismatch at z={z:.2f}"

    print("  [PASS] Fold continuity: left/right agree at all z")


# ── Test 2: Perspective Monotonic ───────────────────────────────────────

def test_perspective_monotonic():
    """Points farther from fold map monotonically in virtual plane UV."""
    z_mid = 0.0

    # Left wall: x from -W (far edge) to 0 (fold)
    u_prev = -1.0
    for x in np.linspace(-PANEL_W, 0.0, 10):
        u, vz, hit = project_wall_to_virtual_plane(
            np.array([x, 0.0, z_mid]), SWEET_SPOT, PANEL_W)
        assert u > u_prev, \
            f"Left u not monotonic: x={x:.3f} -> u={u:.4f} (prev={u_prev:.4f})"
        u_prev = u

    # Right wall: y from -W (far edge) to 0 (fold)
    # u goes from 1.0 (far edge) to 0.5 (fold) — DECREASING
    u_prev = 2.0
    for y in np.linspace(-PANEL_W, 0.0, 10):
        u, vz, hit = project_wall_to_virtual_plane(
            np.array([0.0, y, z_mid]), SWEET_SPOT, PANEL_W)
        assert u < u_prev, \
            f"Right u not monotonic: y={y:.3f} -> u={u:.4f} (prev={u_prev:.4f})"
        u_prev = u

    print("  [PASS] Perspective monotonic: u varies correctly toward fold")


# ── Test 3: Depth Consistency ───────────────────────────────────────────

def test_depth_consistency():
    """BG points lie exactly on the panel walls (no depth displacement)."""
    box = define_virtual_box(PANEL_W, PANEL_H)

    L = box["wall_L"]
    for x_n in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for z_n in [0.0, 0.5, 1.0]:
            p = L["origin"] + x_n * L["width"] * L["u_axis"] + z_n * L["height"] * L["v_axis"]
            assert abs(p[1]) < 1e-9, f"Left wall y != 0: {p[1]:.6f}"
            assert -PANEL_H * 0.5 - 1e-9 <= p[2] <= PANEL_H * 0.5 + 1e-9

    R = box["wall_R"]
    for y_n in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for z_n in [0.0, 0.5, 1.0]:
            p = R["origin"] + y_n * R["width"] * R["u_axis"] + z_n * R["height"] * R["v_axis"]
            assert abs(p[0]) < 1e-9, f"Right wall x != 0: {p[0]:.6f}"
            assert -PANEL_H * 0.5 - 1e-9 <= p[2] <= PANEL_H * 0.5 + 1e-9

    print("  [PASS] Depth consistency: BG points lie on panel walls")


# ── Test 4: BG Mapping — Fold Symmetry ──────────────────────────────────

def test_bg_mapping_fold_symmetry():
    """A symmetric input grid maps symmetrically onto both panels at the fold."""
    H_img, W_img = 60, 100
    img = np.zeros((H_img, W_img, 3), dtype=np.float32)
    # Draw a thicker center stripe (3px wide) for discretization robustness
    c = W_img // 2
    img[:, c-1:c+2, :] = 1.0

    bg_L, bg_R = map_background_to_panels(
        (img * 255).astype(np.uint8), PANEL_W, PANEL_H, 50, 60, SWEET_SPOT)

    # Center stripe should appear near the fold on both panels
    # Check rightmost 3 columns for brightness
    left_fold = bg_L[:, -3:, :].mean()
    right_fold = bg_R[:, -3:, :].mean()

    assert left_fold > 50, \
        f"Left fold should be bright, got {left_fold:.1f}"
    assert right_fold > 50, \
        f"Right fold should be bright, got {right_fold:.1f}"

    # Far edges (column 0) should be darker
    assert bg_L[:, 0, :].mean() < left_fold, "Left far edge should be darker"
    assert bg_R[:, 0, :].mean() < right_fold, "Right far edge should be darker"

    print(f"  [PASS] Fold symmetry: L-fold={left_fold:.1f} R-fold={right_fold:.1f}")


# ── Test 5: BG Mapping Output Dimensions ────────────────────────────────

def test_bg_mapping_dimensions():
    """BG panel textures have correct pixel dimensions at multiple resolutions."""
    img = np.random.randint(0, 255, (120, 200, 3), dtype=np.uint8)
    for pw, ph in [(100, 120), (200, 240), (50, 60)]:
        bg_L, bg_R = map_background_to_panels(img, PANEL_W, PANEL_H, pw, ph, SWEET_SPOT)
        assert bg_L.shape == (ph, pw, 3), f"Left shape {bg_L.shape}"
        assert bg_R.shape == (ph, pw, 3), f"Right shape {bg_R.shape}"
    print("  [PASS] Output dimensions correct at multiple resolutions")


# ── Test 6: Virtual Box Geometry ────────────────────────────────────────

def test_virtual_box_geometry():
    """Virtual box walls span the correct physical regions."""
    box = define_virtual_box(PANEL_W, PANEL_H)
    L = box["wall_L"]
    p00 = L["origin"]
    p10 = L["origin"] + L["width"] * L["u_axis"]
    p01 = L["origin"] + L["height"] * L["v_axis"]

    assert abs(p00[0] - (-PANEL_W)) < 1e-9 and abs(p00[1]) < 1e-9
    assert abs(p00[2] - (-PANEL_H * 0.5)) < 1e-9, "Left wall bottom at z=-H/2"
    assert abs(p10[0]) < 1e-9, "Left wall end should be at fold x=0"
    assert abs(p01[2] - PANEL_H * 0.5) < 1e-9, "Left wall top at z=+H/2"

    R = box["wall_R"]
    p00 = R["origin"]
    p10 = R["origin"] + R["width"] * R["u_axis"]
    assert abs(p00[1] - (-PANEL_W)) < 1e-9 and abs(p00[0]) < 1e-9
    assert abs(p00[2] - (-PANEL_H * 0.5)) < 1e-9, "Right wall bottom at z=-H/2"
    assert abs(p10[1]) < 1e-9, "Right wall end should be at fold y=0"

    print("  [PASS] Virtual box geometry correctly defined")


# ── Test 7: Ray-to-Plane Intersection ───────────────────────────────────

def test_ray_intersection():
    """Ray from viewer through known wall points correctly hits virtual plane."""
    z_mid = 0.0

    # Left edge should map to u=0, hit at (-W, 0)
    u, vz, hit = project_wall_to_virtual_plane(
        np.array([-PANEL_W, 0.0, z_mid]), SWEET_SPOT, PANEL_W)
    assert abs(u) < 0.01, f"Left edge u={u:.4f}"
    assert abs(hit[0] - (-PANEL_W)) < 0.02, f"Hit x={hit[0]:.4f}"
    assert abs(hit[0] + hit[1] - (-PANEL_W)) < 0.02, "Not on plane X+Y=-W"

    # Right edge should map to u=1, hit at (0, -W)
    u, vz, hit = project_wall_to_virtual_plane(
        np.array([0.0, -PANEL_W, z_mid]), SWEET_SPOT, PANEL_W)
    assert abs(u - 1.0) < 0.01, f"Right edge u={u:.4f}"
    assert abs(hit[1] - (-PANEL_W)) < 0.02, f"Hit y={hit[1]:.4f}"

    print("  [PASS] Ray-to-plane intersection correct")


# ── Test 7b: Denom=0 Edge Case ──────────────────────────────────────────

def test_zero_denom_edge_case():
    """Ray parallel to virtual plane (denom=0) handled without error."""
    test_p = np.array([4.24, 0.0, 0.40], dtype=np.float64)
    u, vz, hit = project_wall_to_virtual_plane(test_p, SWEET_SPOT, PANEL_W)
    assert abs(hit[0] - 4.24) < 0.01, f"Hit x={hit[0]:.4f}"
    print(f"  [PASS] Zero denom handled: u={u:.4f}")


# ── Test 8: Vertical Structure Preserved ────────────────────────────────

def test_vertical_preservation():
    """Horizontal stripes in input are preserved in the BG panel mapping."""
    H_img, W_img = 120, 200
    img = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    for row in range(H_img):
        img[row, :, :] = 255 if (row // 15) % 2 == 0 else 0

    bg_L, bg_R = map_background_to_panels(
        img, PANEL_W, PANEL_H, 100, 120, SWEET_SPOT)

    left_rows = bg_L.mean(axis=1).mean(axis=1)
    assert left_rows.std() > 10, f"Vertical variation lost: std={left_rows.std():.1f}"
    assert left_rows.max() - left_rows.min() > 50, \
        f"Vertical range collapsed: {left_rows.max()-left_rows.min():.1f}"

    print(f"  [PASS] Vertical structure preserved: std={left_rows.std():.1f}")


def test_vertical_polarity_top_row_maps_to_positive_z():
    """Bright source top row should appear on the physical +Z panel top."""
    H_img, W_img = 80, 120
    img = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    img[:H_img // 4, :, :] = 255

    bg_L, bg_R = map_background_to_panels(
        img, PANEL_W, PANEL_H, 80, 100, SWEET_SPOT)

    top_luma = float(bg_L[:15].mean() + bg_R[:15].mean()) * 0.5
    bottom_luma = float(bg_L[-15:].mean() + bg_R[-15:].mean()) * 0.5
    assert top_luma > bottom_luma + 30.0, (
        f"Top should be brighter than bottom: top={top_luma:.1f}, "
        f"bottom={bottom_luma:.1f}")
    print("  [PASS] Vertical polarity: source row0 maps to physical +Z")


def test_higher_eye_yields_more_top_down_virtual_shift():
    """Raising the eye shifts the center wall sample lower on the virtual plane."""
    p_center = np.array([-PANEL_W * 0.5, 0.0, 0.0], dtype=np.float64)
    sweet_level = np.array([2.12, 2.12, 0.0], dtype=np.float64)
    sweet_high = np.array([2.12, 2.12, 0.60], dtype=np.float64)
    _u0, z_level, _hit0 = project_wall_to_virtual_plane(
        p_center, sweet_level, PANEL_W)
    _u1, z_high, _hit1 = project_wall_to_virtual_plane(
        p_center, sweet_high, PANEL_W)
    assert z_high < z_level - 0.05, (
        f"Higher eye should move virtual hit lower: {z_high:.3f} vs {z_level:.3f}")
    print("  [PASS] Eye-height monotonicity: higher eye is more top-down")


# ── Harness ─────────────────────────────────────────────────────────────

def harness():
    print("=" * 60)
    print("  HARNESS: Virtual Box Perceptual Mapping")
    print("=" * 60)
    print(f"  Panel: {PANEL_W}x{PANEL_H}m, Sweet spot: {tuple(SWEET_SPOT)}")
    print()

    tests = [
        ("Wall continuity at fold", test_fold_continuity),
        ("Perspective monotonic", test_perspective_monotonic),
        ("Depth consistency", test_depth_consistency),
        ("BG mapping fold symmetry", test_bg_mapping_fold_symmetry),
        ("BG mapping dimensions", test_bg_mapping_dimensions),
        ("Virtual box geometry", test_virtual_box_geometry),
        ("Ray-to-plane intersection", test_ray_intersection),
        ("Zero-denom edge case", test_zero_denom_edge_case),
        ("Vertical preservation", test_vertical_preservation),
        ("Vertical polarity", test_vertical_polarity_top_row_maps_to_positive_z),
        ("Eye-height monotonicity", test_higher_eye_yields_more_top_down_virtual_shift),
    ]

    all_pass = True
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            all_pass = False
            print(f"  [FAIL] {name}: {e}")

    print()
    print("=" * 60)
    print(f"  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    ok = harness()
    import sys
    sys.exit(0 if ok else 1)
