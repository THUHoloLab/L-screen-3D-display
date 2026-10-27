"""
virtual_box.py — Virtual space box B_virtual and inverse perceptual mapping.

Defines B_virtual = {(x,y,z) | x in [-W,0], y in [-W,0], z in [0,H]}
and computes the anamorphic background texture mapping from the input image
onto the physical panel walls, such that the background appears as a
continuous 3D room when viewed from the sweet spot.

Pure numpy — no CUDA/torch dependency, harness-importable.

Core functions:
  define_virtual_box(W, H) -> dict of wall definitions
  project_wall_to_virtual_plane(point_3d, sweet_spot, panel_W)
    -> (u_norm, v_z, hit_point)
  map_background_to_panels(bg_image, panel_W, panel_H, pixel_W, pixel_H, sweet_spot)
    -> (bg_left, bg_right) — uint8 panel images
"""

import numpy as np


def define_virtual_box(W, H):
    """
    Define the virtual space box B_virtual.

    Returns dict with wall definitions. Each wall is a dict with:
      - 'origin': (3,) world coordinate of one corner
      - 'u_axis': (3,) horizontal direction along the wall
      - 'v_axis': (3,) vertical direction along the wall
      - 'width': physical width (m)
      - 'height': physical height (m)
    """
    return {
        "wall_L": {  # Left panel: y=0, x in [-W, 0], z in [-H/2, +H/2]
            "origin": np.array([-W, 0.0, -H * 0.5], dtype=np.float64),
            "u_axis": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "v_axis": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "width": W,
            "height": H,
        },
        "wall_R": {  # Right panel: x=0, y in [-W, 0], z in [-H/2, +H/2]
            "origin": np.array([0.0, -W, -H * 0.5], dtype=np.float64),
            "u_axis": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "v_axis": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "width": W,
            "height": H,
        },
    }


# ---------------------------------------------------------------------------
# Inverse Perceptual Mapping
# ---------------------------------------------------------------------------
# The virtual perceptual plane Pi_virtual is the 45-degree plane:
#   X + Y = -W
# This plane forms the hypotenuse of the isosceles right triangle with the
# two L-shaped panels, connecting (-W, 0) to (0, -W).
#
# For a point p on a physical panel wall, cast a ray from the sweet spot V
# through p, intersect with Pi_virtual, and map the intersection to UV
# coordinates on the input image.
#
# This is the INVERSE of the current depth-based projection:
#   Current: image(u,v) -> world(x,y,z) via DAV3 depth
#   New (BG): wall_point(x,y,z) -> Pi_virtual intersection -> image(u,v)


def project_wall_to_virtual_plane(point_3d, sweet_spot, panel_W):
    """
    Project a 3D point on a panel wall onto the virtual perceptual plane.

    The virtual plane is X + Y = -panel_W (the 45-degree plane at zero depth).

    Args:
        point_3d: (3,) world coordinate of a point on a panel wall.
        sweet_spot: (3,) viewer position V = (vx, vy, vz).
        panel_W: physical panel width in meters.

    Returns:
        u_norm: float in [0,1], 0=left edge, 1=right edge of virtual plane.
        v_z: float, Z coordinate of intersection (caller normalizes to height).
        hit_point: (3,) intersection point on the virtual plane.
    """
    p = np.asarray(point_3d, dtype=np.float64)
    V = np.asarray(sweet_spot, dtype=np.float64)

    direction = p - V
    denom = direction[0] + direction[1]

    if abs(denom) < 1e-12:
        # Ray parallel to plane — point is at infinity in plane direction
        hit = p.copy()
        u_norm = float((hit[0] + panel_W) / panel_W)
        return max(0.0, min(1.0, u_norm)), float(hit[2]), hit

    # t = (-W - V_x - V_y) / (p_x + p_y - V_x - V_y)
    t = (-panel_W - V[0] - V[1]) / denom
    hit = V + t * direction

    # Horizontal UV: u_norm = (hit_x + W) / W, where hit_x in [-W, 0].
    u_norm = float((hit[0] + panel_W) / panel_W)

    return max(0.0, min(1.0, u_norm)), float(hit[2]), hit


def inverse_perceptual_map_for_panel(panel_wall, panel_W, panel_H,
                                      pixel_W, pixel_H,
                                      bg_image, sweet_spot):
    """
    Compute the anamorphic background texture for one physical panel.

    For each pixel on the panel:
      1. Compute its world position on the wall.
      2. Project through the sweet spot onto Pi_virtual (X+Y=-W).
      3. Bilinear-sample the input background image at the projected UV.

    Args:
        panel_wall: dict from define_virtual_box().
        panel_W, panel_H: physical panel dimensions (m).
        pixel_W, pixel_H: panel render resolution (px).
        bg_image: (H_img, W_img, 3) uint8 background image.
        sweet_spot: (3,) viewer position.

    Returns:
        panel_texture: (pixel_H, pixel_W, 3) uint8.
    """
    H_img, W_img = bg_image.shape[:2]
    bg_f = bg_image.astype(np.float32)

    u_px = (np.arange(pixel_W, dtype=np.float32) + 0.5) / float(pixel_W)
    v_px = (np.arange(pixel_H, dtype=np.float32) + 0.5) / float(pixel_H)
    uu, vv = np.meshgrid(u_px, v_px)

    origin = panel_wall["origin"]
    u_axis = panel_wall["u_axis"]
    v_axis = panel_wall["v_axis"]
    w = panel_wall["width"]
    h = panel_wall["height"]
    vv_world = 1.0 - vv

    wx = origin[0] + uu * w * u_axis[0] + vv_world * h * v_axis[0]
    wy = origin[1] + uu * w * u_axis[1] + vv_world * h * v_axis[1]
    wz = origin[2] + uu * w * u_axis[2] + vv_world * h * v_axis[2]

    Vx, Vy, Vz = sweet_spot
    dx = wx - Vx
    dy = wy - Vy
    dz = wz - Vz

    denom = dx + dy
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t = (-panel_W - Vx - Vy) / safe_denom

    hit_x = Vx + t * dx
    hit_z = Vz + t * dz

    u_virtual = np.clip((hit_x + panel_W) / panel_W, 0.0, 1.0)
    v_virtual = np.clip((hit_z + panel_H * 0.5) / panel_H, 0.0, 1.0)

    u_img = u_virtual * (W_img - 1)
    v_img = (1.0 - v_virtual) * (H_img - 1)

    u0 = np.clip(np.floor(u_img).astype(np.int32), 0, W_img - 1)
    v0 = np.clip(np.floor(v_img).astype(np.int32), 0, H_img - 1)
    u1 = np.clip(u0 + 1, 0, W_img - 1)
    v1 = np.clip(v0 + 1, 0, H_img - 1)

    wu = u_img - u0.astype(np.float32)
    wv = v_img - v0.astype(np.float32)

    panel_tex = np.zeros((pixel_H, pixel_W, 3), dtype=np.uint8)
    for ch in range(3):
        panel_tex[:, :, ch] = (
            (1 - wu) * (1 - wv) * bg_f[v0, u0, ch] +
            wu * (1 - wv) * bg_f[v0, u1, ch] +
            (1 - wu) * wv * bg_f[v1, u0, ch] +
            wu * wv * bg_f[v1, u1, ch]
        ).astype(np.uint8)

    return panel_tex


def map_background_to_panels(bg_image, panel_W, panel_H,
                              pixel_W, pixel_H, sweet_spot):
    """
    Compute anamorphic background textures for both panels.

    Args:
        bg_image: (H_img, W_img, 3) uint8 background portion of input image.
        panel_W, panel_H: physical panel dimensions (m).
        pixel_W, pixel_H: panel render resolution (px).
        sweet_spot: (3,) viewer position.

    Returns:
        (bg_left, bg_right): tuple of (pixel_H, pixel_W, 3) uint8 arrays.
    """
    box = define_virtual_box(panel_W, panel_H)

    print(f"[virtual_box] Mapping BG {bg_image.shape[1]}x{bg_image.shape[0]} "
          f"-> {pixel_W}x{pixel_H} per panel")

    bg_left = inverse_perceptual_map_for_panel(
        box["wall_L"], panel_W, panel_H, pixel_W, pixel_H, bg_image, sweet_spot)

    bg_right = inverse_perceptual_map_for_panel(
        box["wall_R"], panel_W, panel_H, pixel_W, pixel_H, bg_image, sweet_spot)

    return bg_left, bg_right


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _harness():
    """Verify virtual box geometry and inverse perceptual mapping."""
    W = 1.28
    H = 1.44
    sweet = np.array([2.12, 2.12, 1.60], dtype=np.float64)

    print("=" * 60)
    print("  virtual_box self-test")
    print("=" * 60)

    # Test 1: define_virtual_box
    box = define_virtual_box(W, H)
    assert "wall_L" in box
    assert "wall_R" in box
    assert abs(box["wall_L"]["width"] - W) < 1e-6
    assert abs(box["wall_R"]["height"] - H) < 1e-6
    print("  [PASS] define_virtual_box")

    # Test 2: fold point (0,0,H/2) projects to center of virtual plane
    fold_center = np.array([0.0, 0.0, H / 2.0], dtype=np.float64)
    u, vz, hit = project_wall_to_virtual_plane(fold_center, sweet, W)
    assert abs(u - 0.5) < 0.05, \
        f"Fold should map to u~0.5, got u={u:.4f}"
    print(f"  [PASS] Fold center -> u={u:.4f} (expected ~0.5), "
          f"hit=({hit[0]:.3f}, {hit[1]:.3f}, {hit[2]:.3f})")

    # Test 3: left edge (-W,0,H/2) maps to u=0
    left_edge = np.array([-W, 0.0, H / 2.0], dtype=np.float64)
    u, vz, hit = project_wall_to_virtual_plane(left_edge, sweet, W)
    assert abs(u) < 0.01, f"Left edge should map to u~0, got u={u:.4f}"
    print(f"  [PASS] Left edge (-W,0) -> u={u:.4f} (expected ~0)")

    # Test 4: right edge (0,-W,H/2) maps to u=1
    right_edge = np.array([0.0, -W, H / 2.0], dtype=np.float64)
    u, vz, hit = project_wall_to_virtual_plane(right_edge, sweet, W)
    assert abs(u - 1.0) < 0.01, \
        f"Right edge should map to u~1, got u={u:.4f}"
    print(f"  [PASS] Right edge (0,-W) -> u={u:.4f} (expected ~1)")

    # Test 5: synthetic BG — continuity at the fold
    H_img, W_img = 120, 200
    bg_synth = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    bg_synth[:, W_img // 2 - 2:W_img // 2 + 2, :] = 255   # center stripe
    bg_synth[30:90, :, 0] = 128                             # horizontal band

    bg_left, bg_right = map_background_to_panels(
        bg_synth, W, H, pixel_W=100, pixel_H=120, sweet_spot=sweet)

    # Center stripe (u=0.5) should appear at the fold on both panels
    left_fold = bg_left[:, -1, :].mean()
    right_fold = bg_right[:, 0, :].mean()
    assert left_fold > 20, \
        f"Left fold column should be bright, got {left_fold:.1f}"
    assert right_fold > 20, \
        f"Right fold column should be bright, got {right_fold:.1f}"
    print(f"  [PASS] Fold continuity: L-edge={left_fold:.1f}, "
          f"R-edge={right_fold:.1f}")

    # Test 6: vertical structure preserved
    mid_l = bg_left[60, :, 0].mean()
    top_l = bg_left[10, :, 0].mean()
    assert mid_l > top_l + 10, \
        "Mid-row (band) should be brighter than top row"
    print(f"  [PASS] Vertical structure: mid={mid_l:.1f}, top={top_l:.1f}")

    # Test 7: output dimensions
    assert bg_left.shape == (120, 100, 3), f"Left shape {bg_left.shape}"
    assert bg_right.shape == (120, 100, 3), f"Right shape {bg_right.shape}"
    print(f"  [PASS] Shapes: left={bg_left.shape}, right={bg_right.shape}")

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    ok = _harness()
    import sys
    sys.exit(0 if ok else 1)
