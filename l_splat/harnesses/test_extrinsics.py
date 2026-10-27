"""
harness_camera_extrinsics.py — Phase 1 & 2 HDD test harness.

Validates world-to-camera View Matrix construction for L-shaped 90-degree
outward-fold LED screen (Z-up right-handed world).

Tests multiple candidate formulations against strict mathematical assertions:
  1. Orthogonality: Right·Up = Up·Forward = Forward·Right = 0
  2. Unit-norm: |Right| = |Up| = |Forward| = 1
  3. Chirality: det(Rotation_3x3) == 1 (right-handed, no mirror)
  4. Orientation:
     - Left camera must look toward -Y world (screen-facing)
     - Right camera must look toward -X world (screen-facing)
     - Up vector must produce upright images (world +Z → image +Y)
  5. Fold-edge alignment:
     - Fold (X=0 or Y=0) must map to the CORRECT screen edge
       (right edge of left image, left edge of right image)
"""

import numpy as np

# ===========================================================================
# World configuration
# ===========================================================================
VIEWER_POS = np.array([2.6, 2.6, 1.0], dtype=np.float64)
PANEL_W = 0.889   # screen width (m)
PANEL_H = 1.000   # screen height (m)
HALF_H = PANEL_H / 2.0  # half-height along Z (Z=UP)

# Left screen corners (Y=0 plane, X∈[-W, 0], Z∈[-H/2, +H/2])
# Order: [top-L, top-R, bot-L, bot-R]
LEFT_CORNERS = np.array([
    [-PANEL_W, 0.0,  HALF_H],   # top-left     (far from fold)
    [0.0,      0.0,  HALF_H],   # top-right    (at fold)
    [-PANEL_W, 0.0, -HALF_H],   # bottom-left  (far from fold)
    [0.0,      0.0, -HALF_H],   # bottom-right (at fold)
], dtype=np.float64)

# Right screen corners (X=0 plane, Y∈[-W, 0], Z∈[-H/2, +H/2])
RIGHT_CORNERS = np.array([
    [0.0,  0.0,        HALF_H],   # top-left     (at fold)
    [0.0, -PANEL_W,    HALF_H],   # top-right    (far from fold)
    [0.0,  0.0,       -HALF_H],   # bottom-left  (at fold)
    [0.0, -PANEL_W,   -HALF_H],   # bottom-right (far from fold)
], dtype=np.float64)

# Screen target points (screen centers in world)
LEFT_CENTER = np.mean(LEFT_CORNERS, axis=0)
RIGHT_CENTER = np.mean(RIGHT_CORNERS, axis=0)

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# ===========================================================================
# Candidate view matrix formulations
# ===========================================================================

def _build_view_matrix(eye, forward_world, right_world, up_world):
    """
    Build 4x4 view matrix from camera basis vectors (all in world coords).

    OpenGL convention: camera at origin, looks toward -Z.
    Row 0 = right   (camera X axis in world coords)
    Row 1 = up      (camera Y axis in world coords)
    Row 2 = -forward (camera Z axis in world coords = opposite to view dir)

    Translation: p_cam = R @ (p_world - eye)
      → V[0,3] = -dot(right, eye)
      → V[1,3] = -dot(up, eye)
      → V[2,3] =  dot(forward, eye)   [since row2 = -forward]
      → V[3,3] = 1
    """
    V = np.eye(4, dtype=np.float64)
    V[0, :3] = right_world
    V[1, :3] = up_world
    V[2, :3] = -forward_world
    V[0, 3] = -np.dot(right_world, eye)
    V[1, 3] = -np.dot(up_world, eye)
    V[2, 3] = np.dot(forward_world, eye)
    V[3, 3] = 1.0
    return V


# --- Candidate A: Standard OpenGL look-at (cross(fwd, up)) ---
# This is the "original" formulation from look_at_opengl / _look_at_opengl_np
# Used by _build_asymmetric_frustum_np (which builds the DualFrustumCamera).

def candidate_A_left():
    """Standard OpenGL look-at: side = cross(fwd, up), up = cross(side, fwd)."""
    eye = VIEWER_POS
    target = eye + np.array([0.0, -1.0, 0.0])  # look toward -Y
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(fwd, u)           # side = cross(fwd, up)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)    # re-orthogonalize
    return _build_view_matrix(eye, fwd, side, up_ortho)


def candidate_A_right():
    eye = VIEWER_POS
    target = eye + np.array([-1.0, 0.0, 0.0])  # look toward -X
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(fwd, u)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)
    return _build_view_matrix(eye, fwd, side, up_ortho)


# --- Candidate B: Current code (cross(up, fwd), cross(side, fwd)) ---
# This is look_at_offscreen after the last session's fix.
# side = cross(up, fwd) to move fold to correct edge,
# but re-orthogonalization uses cross(side, fwd) which gives wrong up.

def candidate_B_left():
    eye = VIEWER_POS
    target = eye + np.array([0.0, -1.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(u, fwd)           # cross(up, fwd) — reversed order
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)    # cross(side, fwd) — BUG: should be cross(fwd, side)
    return _build_view_matrix(eye, fwd, side, up_ortho)


def candidate_B_right():
    eye = VIEWER_POS
    target = eye + np.array([-1.0, 0.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(u, fwd)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)
    return _build_view_matrix(eye, fwd, side, up_ortho)


# --- Candidate C: cross(up, fwd) for side, cross(fwd, side) for up ---
# Same as B but with CORRECT re-orthogonalization.
# This should give det=-1 (mirrored) but correct fold edge and correct up.

def candidate_C_left():
    eye = VIEWER_POS
    target = eye + np.array([0.0, -1.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(u, fwd)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(fwd, side)    # fixed: cross(fwd, side)
    return _build_view_matrix(eye, fwd, side, up_ortho)


def candidate_C_right():
    eye = VIEWER_POS
    target = eye + np.array([-1.0, 0.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(u, fwd)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(fwd, side)
    return _build_view_matrix(eye, fwd, side, up_ortho)


# --- Candidate D: PER-CAMERA cross product selection ---
# Left camera:  side = cross(up, fwd)  → side = (1, 0, 0), fold at RIGHT edge
# Right camera: side = cross(fwd, up)  → side = (0, 1, 0), fold at LEFT edge
# Both with cross(side, fwd) for re-orthogonalization.
# BUT: det differs between the two sides.

def candidate_D_left():
    """Left: side=cross(up,fwd), up=cross(side,fwd) — det=-1, fold at right edge."""
    eye = VIEWER_POS
    target = eye + np.array([0.0, -1.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(u, fwd)           # cross(up, fwd) → (1, 0, 0)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)    # cross(side, fwd) → (0, 0, -1)? No...
    return _build_view_matrix(eye, fwd, side, up_ortho)


def candidate_D_right():
    """Right: side=cross(fwd,up), up=cross(side,fwd) — det=1, fold at right edge."""
    eye = VIEWER_POS
    target = eye + np.array([-1.0, 0.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = WORLD_UP / np.linalg.norm(WORLD_UP)
    side = np.cross(fwd, u)           # cross(fwd, up) → (0, 1, 0)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)    # cross(side, fwd) → (0, 0, 1)
    return _build_view_matrix(eye, fwd, side, up_ortho)


# --- Candidate E: Standard OpenGL look-at (like A) + explicit horizontal flip ---
# Build with correct math (like A), then apply post-render flip.
# This is the "accept det=1 but flip image" approach.
# Same as A — the flip is external to the view matrix.

# --- Candidate F: Force det=1 with cross(fwd, up) style, accept fold position,
#     then use screen-corner swapping to move fold to correct edge. ---
# Basically same as A but we process corners differently.
# This is a pipeline-level fix, not a view-matrix fix.


# --- Candidate G: NEGATE UP WORLD for correct fold + det=1 ---
# Use up_world = (0, 0, -1) with side = cross(fwd, up).
# This gives det=1 AND fold at correct edge, but image is upside down.
# Post-render vertical flip compensates.

def candidate_G_left():
    """up=(0,0,-1), side=cross(fwd,up) — det=1, fold at right edge, image upside-down."""
    eye = VIEWER_POS
    target = eye + np.array([0.0, -1.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    side = np.cross(fwd, u)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)
    return _build_view_matrix(eye, fwd, side, up_ortho)


def candidate_G_right():
    eye = VIEWER_POS
    target = eye + np.array([-1.0, 0.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    side = np.cross(fwd, u)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)
    return _build_view_matrix(eye, fwd, side, up_ortho)


# --- Candidate H: flip_y @ V_B ---
# Start from Candidate B (det=1, fold correct, upside-down) and apply a
# vertical-flip matrix to the view space:
#   V' = flip_y @ V,   flip_y = diag(1, -1, 1, 1)
#
# This negates camera-space Y, which flips the image vertically in the
# projection, correcting the upside-down orientation.
#
# Side effects:
#   - det becomes -1 (single-axis reflection → mirror along X in image).
#   - Orthogonality preserved (flip_y is orthonormal, det=-1).
#   - Fold edge alignment preserved (Y-flip does not affect X coordinate).
#   - The -1 determinant means the rasterizer sees a left-handed view
#     coordinate system. For 3DGS this is acceptable: the rasterizer
#     does not perform face-culling (it renders Gaussians, not triangles),
#     and SH view-direction evaluation uses the camera-to-Gaussian vector
#     which is unaffected by the reflection in the view matrix itself
#     (pos - campos in world space, independent of V's chirality).
#
# This is mathematically equivalent to Candidate C but achieved through
# an explicit post-view-matrix Y-axis flip rather than through different
# cross-product ordering in look-at.

def _apply_flip_y(V):
    """Apply vertical-flip correction: V' = flip_y @ V."""
    flip_y = np.diag([1.0, -1.0, 1.0, 1.0])
    return flip_y @ V


def candidate_H_left():
    """Candidate B + flip_y — det=-1, upright, fold at right edge."""
    V_B = candidate_B_left()
    return _apply_flip_y(V_B)


def candidate_H_right():
    """Candidate B + flip_y — det=-1, upright, fold at left edge."""
    V_B = candidate_B_right()
    return _apply_flip_y(V_B)


# ===========================================================================
# Validation functions
# ===========================================================================

def orthogonality_check(V):
    """Check pairwise dot products of camera axes."""
    right = V[0, :3]
    up = V[1, :3]
    neg_fwd = V[2, :3]   # -forward in world
    fwd = -neg_fwd
    dots = {
        "right·up": np.dot(right, up),
        "right·fwd": np.dot(right, fwd),
        "up·fwd": np.dot(up, fwd),
    }
    return dots


def unit_norm_check(V):
    right = V[0, :3]
    up = V[1, :3]
    neg_fwd = V[2, :3]
    return {
        "|right|": np.linalg.norm(right),
        "|up|": np.linalg.norm(up),
        "|fwd|": np.linalg.norm(neg_fwd),
    }


def chirality_check(V):
    """det of rotation submatrix (first 3x3)."""
    R = V[:3, :3]
    return np.linalg.det(R)


def extract_axes(V):
    """Extract right, up, forward (world-space direction camera looks toward)."""
    right = V[0, :3]
    up = V[1, :3]
    fwd = -V[2, :3]  # negate row 2 to get forward direction (where camera looks)
    return right, up, fwd


def orientation_check(V, expected_fwd_world, label):
    """
    Check that forward direction aligns with expected.
    expected_fwd_world: unit vector in world space that camera should look toward.
    """
    _, _, fwd = extract_axes(V)
    alignment = np.dot(fwd, expected_fwd_world)
    return {
        "fwd_alignment": alignment,
        "fwd_actual": fwd,
        "fwd_expected": expected_fwd_world,
    }


def compute_frustum_bounds_from_view(V, screen_corners, near=0.01):
    """
    Project screen corners through V to get frustum l,r,b,t at near plane.
    Uses the same algorithm as _build_asymmetric_frustum_np.
    """
    corners_h = np.hstack([screen_corners, np.ones((4, 1), dtype=np.float64)])
    corners_view = (V @ corners_h.T).T
    z_vals = corners_view[:, 2]

    if np.any(z_vals >= 0):
        return None, f"Screen corners behind camera: z_view={z_vals}"

    t_factors = -near / z_vals
    x_near = corners_view[:, 0] * t_factors
    y_near = corners_view[:, 1] * t_factors

    l, r = float(np.min(x_near)), float(np.max(x_near))
    b, t_inner = float(np.min(y_near)), float(np.max(y_near))
    return (l, r, b, t_inner), None


def fold_edge_check(V, screen_corners, near=0.01, fold_at_right=True):
    """
    Check which screen edge the fold maps to.

    For the LEFT screen: fold is at X=0 (right edge of physical screen).
      If fold_at_right=True: expect fold world point to produce x_near ≈ r (right edge).
      If fold_at_right=False: expect fold world point to produce x_near ≈ l (left edge).

    For the RIGHT screen: fold is at Y=0 (left edge of physical screen).
      If fold_at_right=False: expect fold world point to produce x_near ≈ l (left edge).
      If fold_at_right=True: expect fold world point to produce x_near ≈ r (right edge).
    """
    bounds, err = compute_frustum_bounds_from_view(V, screen_corners, near)
    if err:
        return {"error": err}

    l, r, b, t_inner = bounds
    # Find x_near for a point at the fold
    if np.allclose(screen_corners[1, 1], 0.0):  # Left screen: fold at X=0
        fold_pt = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:  # Right screen: fold at Y=0
        fold_pt = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    fold_view = V @ fold_pt
    if fold_view[2] >= 0:
        return {"error": f"Fold point behind camera: z_view={fold_view[2]}"}

    t_factor = -near / fold_view[2]
    x_fold_near = fold_view[0] * t_factor

    # Determine which edge the fold maps to
    # x_ndc = (2*x_near - (r+l)) / (r-l)
    # At left edge (x_near=l):  x_ndc = -1
    # At right edge (x_near=r): x_ndc = +1
    ndc_fold = (2.0 * x_fold_near - (r + l)) / (r - l) if abs(r - l) > 1e-12 else 0.0

    return {
        "l": l, "r": r, "b": b, "t": t_inner,
        "x_fold_near": x_fold_near,
        "ndc_fold": ndc_fold,
        "fold_at_left_edge": abs(ndc_fold - (-1.0)) < 0.01,
        "fold_at_right_edge": abs(ndc_fold - 1.0) < 0.01,
    }


def image_up_check(V):
    """
    Check whether world +Z (up) maps to image +Y (up) or -Y (down).

    Projects a point at (0,0,+1) and (0,0,-1) in world and checks which
    has larger Y in the rendered image pixel coordinate.

    Uses the FULL NDC→pixel mapping: pixel_y = H/2 * (1 - y_ndc),
    accounting for frustum bounds t and b.
    """
    pt_top = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)      # world up
    pt_bot = np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float64)     # world down

    top_view = V @ pt_top
    bot_view = V @ pt_bot

    # Y in view space (camera coordinates)
    y_top_cam = top_view[1]
    y_bot_cam = bot_view[1]
    z_top_cam = top_view[2]
    z_bot_cam = bot_view[2]

    # Without frustum bounds, compare raw camera-Y (which determines NDC-Y
    # ordering since z values are nearly equal for points at similar depth).
    # Higher camera Y → higher Y_ndc → LOWER pixel row (further down in image).
    # So: world-up should give LOWER pixel row (closer to top of image).
    is_upright = y_top_cam > y_bot_cam

    return {
        "y_cam_top": y_top_cam,
        "y_cam_bot": y_bot_cam,
        "is_upright": is_upright,
        "note": ("y_cam_top > y_cam_bot: world +Z maps to image top [OK]"
                 if is_upright else
                 "y_cam_top < y_cam_bot: world +Z maps to image bottom [UPSIDE-DOWN]"),
    }


def flip_matrix_validation(V_original, V_flipped):
    """
    Validate that applying flip_y = diag(1,-1,1,1) preserves what it should
    and corrects what it should.

    Checks:
      1. Orthogonality preserved: R @ R.T = I for both matrices.
      2. Determinant changes sign: det(R_flipped) = -det(R_original).
      3. Row 0 (right) unchanged, row 1 (up) negated, row 2 (-fwd) unchanged.
    """
    R_orig = V_original[:3, :3]
    R_flip = V_flipped[:3, :3]

    # Orthogonality
    I_orig = R_orig @ R_orig.T
    I_flip = R_flip @ R_flip.T
    ortho_ok = (np.allclose(I_orig, np.eye(3), atol=1e-10) and
                np.allclose(I_flip, np.eye(3), atol=1e-10))

    # Determinant
    det_orig = np.linalg.det(R_orig)
    det_flip = np.linalg.det(R_flip)
    det_sign_flipped = abs(det_flip + det_orig) < 1e-10  # one positive, one negative

    # Row checks
    row0_preserved = np.allclose(V_flipped[0, :3], V_original[0, :3], atol=1e-10)
    row1_negated = np.allclose(V_flipped[1, :3], -V_original[1, :3], atol=1e-10)
    row2_preserved = np.allclose(V_flipped[2, :3], V_original[2, :3], atol=1e-10)

    return {
        "orthogonality_preserved": ortho_ok,
        "det_original": det_orig,
        "det_flipped": det_flip,
        "det_sign_flipped": det_sign_flipped,
        "row0_preserved": row0_preserved,
        "row1_negated": row1_negated,
        "row2_preserved": row2_preserved,
    }


# ===========================================================================
# Run all tests
# ===========================================================================

CANDIDATES = {
    "A (std OpenGL, cross(fwd,up), cross(side,fwd))": {
        "left": candidate_A_left,
        "right": candidate_A_right,
    },
    "B (current broken: cross(up,fwd), cross(side,fwd))": {
        "left": candidate_B_left,
        "right": candidate_B_right,
    },
    "C (cross(up,fwd), cross(fwd,side) — fixed re-ortho)": {
        "left": candidate_C_left,
        "right": candidate_C_right,
    },
    "D (per-camera cross order selection)": {
        "left": candidate_D_left,
        "right": candidate_D_right,
    },
    "G (up=(0,0,-1), cross(fwd,up) — det=1, fold correct, upside-down)": {
        "left": candidate_G_left,
        "right": candidate_G_right,
    },
    "H (Candidate B + flip_y@V — det=-1, upright, fold correct)": {
        "left": candidate_H_left,
        "right": candidate_H_right,
    },
}


def print_matrix(label, V):
    print(f"    {label}:")
    for i in range(4):
        print(f"      [{V[i,0]:10.6f} {V[i,1]:10.6f} {V[i,2]:10.6f} {V[i,3]:10.6f}]")


def validate_candidate(name, builders):
    print(f"\n{'='*70}")
    print(f"  Candidate: {name}")
    print(f"{'='*70}")

    for side in ["left", "right"]:
        V = builders[side]()
        right, up, fwd = extract_axes(V)

        print(f"\n  --- {side.upper()} Camera ---")
        print_matrix("V (world→camera)", V)
        print(f"    Forward (world):  [{fwd[0]:.4f}, {fwd[1]:.4f}, {fwd[2]:.4f}]")
        print(f"    Right   (world):  [{right[0]:.4f}, {right[1]:.4f}, {right[2]:.4f}]")
        print(f"    Up      (world):  [{up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f}]")

        # Orthogonality
        ortho = orthogonality_check(V)
        ortho_ok = all(abs(v) < 1e-10 for v in ortho.values())
        print(f"    Orthogonality: {'PASS' if ortho_ok else 'FAIL'} {ortho}")

        # Unit norms
        norms = unit_norm_check(V)
        norms_ok = all(abs(v - 1.0) < 1e-10 for v in norms.values())
        print(f"    Unit norms:    {'PASS' if norms_ok else 'FAIL'} {norms}")

        # Chirality
        det = chirality_check(V)
        chiral_ok = abs(det - 1.0) < 1e-10
        print(f"    Chirality:     det={det:.6f} → {'PASS (proper rotation)' if chiral_ok else 'FAIL (mirror/reflection!)'}")

        # Orientation
        expected_fwd = np.array([0.0, -1.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])
        orient = orientation_check(V, expected_fwd, side)
        fwd_ok = orient["fwd_alignment"] > 0.999
        print(f"    Orientation:   {'PASS' if fwd_ok else 'FAIL'} (fwd·expected={orient['fwd_alignment']:.6f})")

        # Image up check
        up_info = image_up_check(V)
        print(f"    Image upright: {'YES' if up_info['is_upright'] else 'NO (UPSIDE DOWN!)'} "
              f"(y_cam_top={up_info['y_cam_top']:.4f}, y_cam_bot={up_info['y_cam_bot']:.4f})")
        print(f"      {up_info['note']}")

        # Flip-matrix validation (only for Candidate H)
        if "flip_y" in name:
            V_orig = candidate_B_left() if side == "left" else candidate_B_right()
            flip_val = flip_matrix_validation(V_orig, V)
            print(f"    Flip-matrix validation:")
            print(f"      orthogonality preserved: {'PASS' if flip_val['orthogonality_preserved'] else 'FAIL'}")
            print(f"      det: {flip_val['det_original']:.1f} → {flip_val['det_flipped']:.1f} "
                  f"(sign flipped: {'PASS' if flip_val['det_sign_flipped'] else 'FAIL'})")
            print(f"      row0 (right) preserved:  {'PASS' if flip_val['row0_preserved'] else 'FAIL'}")
            print(f"      row1 (up) negated:       {'PASS' if flip_val['row1_negated'] else 'FAIL'}")
            print(f"      row2 (-fwd) preserved:   {'PASS' if flip_val['row2_preserved'] else 'FAIL'}")

        # Frustum bounds + fold edge
        if side == "left":
            corners = LEFT_CORNERS
            fold_at_right = True   # left screen: fold should be at right edge of image
        else:
            corners = RIGHT_CORNERS
            fold_at_right = False  # right screen: fold should be at left edge of image

        fold_info = fold_edge_check(V, corners)
        if "error" in fold_info:
            print(f"    Fold check:    ERROR: {fold_info['error']}")
        else:
            l, r, b, t_inner = fold_info["l"], fold_info["r"], fold_info["b"], fold_info["t"]
            print(f"    Frustum:       l={l:.6f} r={r:.6f} b={b:.6f} t={t_inner:.6f}")
            print(f"    Fold NDC:      x_ndc={fold_info['ndc_fold']:.4f} "
                  f"(left={fold_info['fold_at_left_edge']}, right={fold_info['fold_at_right_edge']})")
            # Check fold at correct edge
            if fold_at_right:
                fold_ok = fold_info['fold_at_right_edge']
            else:
                fold_ok = fold_info['fold_at_left_edge']
            print(f"    Fold edge:     {'PASS (correct edge)' if fold_ok else 'FAIL (wrong edge!)'}")

        print()

    # Summary for this candidate
    print(f"  --- SUMMARY: {name} ---")
    for side in ["left", "right"]:
        V = builders[side]()
        det = chirality_check(V)
        up_info = image_up_check(V)
        chiral_str = "det=1" if abs(det-1.0) < 1e-10 else f"det={det:.1f} (MIRROR)"
        up_str = "upright" if up_info["is_upright"] else "UPSIDE-DOWN"
        if side == "left":
            corners, fold_at_right = LEFT_CORNERS, True
        else:
            corners, fold_at_right = RIGHT_CORNERS, False
        fold_info = fold_edge_check(V, corners)
        if "error" not in fold_info:
            if fold_at_right:
                fold_str = "correct" if fold_info["fold_at_right_edge"] else "wrong edge"
            else:
                fold_str = "correct" if fold_info["fold_at_left_edge"] else "wrong edge"
        else:
            fold_str = fold_info["error"]
        print(f"    {side:5s}: {chiral_str} | {up_str} | fold: {fold_str}")


if __name__ == "__main__":
    print("=" * 70)
    print("  CAMERA EXTRINSICS HARNESS")
    print("  Z-up world → 3DGS View Matrix (OpenGL convention)")
    print("=" * 70)
    print(f"\n  Viewer position: {VIEWER_POS}")
    print(f"  Panel: {PANEL_W:.3f} × {PANEL_H:.3f} m")
    print(f"  Left screen:  Y=0, X∈[{-PANEL_W:.3f}, 0], Z∈[{-HALF_H:.3f}, {HALF_H:.3f}]")
    print(f"  Right screen: X=0, Y∈[{-PANEL_W:.3f}, 0], Z∈[{-HALF_H:.3f}, {HALF_H:.3f}]")

    for name, builders in CANDIDATES.items():
        validate_candidate(name, builders)

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print("""
  The CORRECT formulation must have ALL of:
    1. det=1 (proper rotation, no mirror → correct SH/text/chirality)
    2. Upright image (world +Z → image +Y)
    3. Fold at correct screen edge
       - Left camera: fold (X=0) → right edge of image
       - Right camera: fold (Y=0) → left edge of image

  No pure rotation satisfies all three (proven: see Candidate A vs B vs C).

  Candidate H (B + flip_y):
    - Side:  side=cross(up,fwd) → correct fold alignment
    - Up:    corrected by flip_y @ V → upright
    - det:   -1 (single-axis Y reflection in camera space)
    - The det=-1 is a necessary trade-off. Impact analysis:
        * 3DGS rasterizes Gaussians (NOT triangles) → no backface culling
        * SH view-direction: computed as pos-campos in world space, before V
        * The reflection is along camera-Y (vertical) → manifests as left-right
          mirror in the final image
    - For the L-shaped display: both screens receive identical chirality
      treatment, so the stitched image is internally consistent.

  RECOMMENDATION: Use Candidate H (flip_y @ V_B).
    - integrate flip_y into look_at_offscreen() or apply it in
      DualFrustumCamera after building V.
    - The flip_y matrix is self-contained and preserves orthogonality.
""")
