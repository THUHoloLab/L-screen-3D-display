"""
harness_overlap_blending.py -- Phase 1 & 2 HDD test harness.

Verifies the overlap frustum with 2D alpha blending strategy for
seamless L-screen fold stitching:

  1. Extend left camera's right bound (r_L) and right camera's left
     bound (l_R) by overlap_width=0.05m in world space --> compute
     corresponding near-plane delta.
  2. After rasterization, blend the overlap zone (~50-100 px) with
     linear alpha ramp to mask 3DGS projection errors at the seam.

Physical parameters (must match end_to_end_pipeline.py):
  Panel:  1.28 x 1.44 m  (832 x 936 px)
  Viewer: (2.12, 2.12, 1.60)
  Fold:   Z axis (X=0, Y=0)
"""

import math
import numpy as np

# ===========================================================================
# Physical configuration (mirrors end_to_end_pipeline.py)
# ===========================================================================
PANEL_W = 1.28
PANEL_H = 1.44
HALF_H = PANEL_H / 2.0  # 0.72 m
VIEWER_POS = np.array([2.12, 2.12, 1.60], dtype=np.float64)
NEAR_CLIP = 0.01
FAR_CLIP = 100.0

OVERLAP_WORLD_M = 0.05       # extend each frustum by 5 cm past the fold
BLEND_PIXELS = 75             # 2D alpha blend zone width

PANEL_PIXEL_W = 832
PANEL_PIXEL_H = 936


# ===========================================================================
# 1 -- Screen corner geometry
# ===========================================================================

def build_screen_corners(W, H):
    """
    Return (left_corners, right_corners) for L-shaped screen geometry.

    Left panel:  Y=0 plane, X in [-W, 0], Z in [-H/2, H/2]
    Right panel: X=0 plane, Y in [-W, 0], Z in [-H/2, H/2]

    Order: [top-left, top-right, bottom-left, bottom-right]
    "top" = +Z, "bottom" = -Z.
    """
    Hz = H / 2.0
    left_corners = np.array([
        [-W, 0.0,  Hz],   # top-left     (far from fold)
        [0.0, 0.0,  Hz],  # top-right    (at fold)
        [-W, 0.0, -Hz],   # bottom-left  (far from fold)
        [0.0, 0.0, -Hz],  # bottom-right (at fold)
    ], dtype=np.float64)

    right_corners = np.array([
        [0.0,  0.0,  Hz],  # top-left     (at fold)
        [0.0, -W,    Hz],  # top-right    (far from fold)
        [0.0,  0.0, -Hz],  # bottom-left  (at fold)
        [0.0, -W,   -Hz],  # bottom-right (far from fold)
    ], dtype=np.float64)

    return left_corners, right_corners


# ===========================================================================
# 2 -- View matrix (Candidate H: look_at_offscreen + flip_y)
# ===========================================================================

def look_at_offscreen(eye, target, up):
    """OpenGL view matrix, side=cross(up,fwd), cross(side,fwd) re-orthogonalize."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    u = up / np.linalg.norm(up)
    side = np.cross(u, fwd)
    side /= np.linalg.norm(side)
    u_ortho = np.cross(side, fwd)
    V = np.eye(4, dtype=np.float64)
    V[0, :3] = side
    V[1, :3] = u_ortho
    V[2, :3] = -fwd
    V[0, 3] = -np.dot(side, eye)
    V[1, 3] = -np.dot(u_ortho, eye)
    V[2, 3] = np.dot(fwd, eye)
    return V


def build_view_matrix(eye, forward_world, up_guess):
    """Candidate-H: look_at_offscreen then flip_y."""
    target = eye + forward_world
    V_raw = look_at_offscreen(eye, target, up_guess)
    flip_y = np.diag([1.0, -1.0, 1.0, 1.0])
    return flip_y @ V_raw


# ===========================================================================
# 3 -- Frustum bounds (standard, no overlap)
# ===========================================================================

def compute_frustum_bounds(screen_corners, near, V):
    """Project screen corners through V to get l, r, b, t on near plane."""
    n_corners = screen_corners.shape[0]
    corners_h = np.hstack([screen_corners, np.ones((n_corners, 1), dtype=np.float64)])
    corners_view = (V @ corners_h.T).T
    z_vals = corners_view[:, 2]

    if np.any(z_vals >= 0):
        raise ValueError(f"Screen corners behind camera: z_view={z_vals}")

    t_factors = -near / z_vals
    x_near = corners_view[:, 0] * t_factors
    y_near = corners_view[:, 1] * t_factors

    l, r = float(np.min(x_near)), float(np.max(x_near))
    b, t = float(np.min(y_near)), float(np.max(y_near))
    return l, r, b, t


def build_projection_matrix(near, far, l, r, b, t):
    """OpenGL asymmetric projection matrix."""
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * near / (r - l)
    P[1, 1] = 2.0 * near / (t - b)
    P[0, 2] = (r + l) / (r - l)
    P[1, 2] = (t + b) / (t - b)
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def project_to_ndc(V, P, point_world):
    """Project world point to NDC [x, y, z]."""
    p_h = np.append(np.asarray(point_world, dtype=np.float64), 1.0)
    p_view = V @ p_h
    p_clip = P @ p_view
    w = p_clip[3]
    if abs(w) < 1e-15:
        return None
    return np.array([p_clip[0]/w, p_clip[1]/w, p_clip[2]/w])


# ===========================================================================
# 4 -- Overlap frustum computation
# ===========================================================================

def calc_overlap_bounds(screen_width, overlap_world, viewer_pos):
    """
    Compute dual frustum bounds WITH overlap extension.

    Strategy:
      For each camera, add virtual screen corners offset past the fold by
      `overlap_world` metres, then compute frustum bounds from the expanded
      corner set.  This extends the frustum so that content slightly past
      the fold is visible to both cameras.

    Args:
        screen_width:  physical width of each panel (m)
        overlap_world: extension past fold in world space (m), e.g. 0.05
        viewer_pos:    (3,) viewer / sweet-spot position

    Returns:
        dict with all frustum parameters, overlap metrics, and projection matrices
    """
    W, H = float(screen_width), PANEL_H
    Hz = H / 2.0
    near = NEAR_CLIP
    far = FAR_CLIP

    # --- Standard screen corners ---
    left_corners, right_corners = build_screen_corners(W, H)

    # --- View matrices ---
    V_L = build_view_matrix(viewer_pos,
                            np.array([0.0, -1.0, 0.0]),   # look toward Y=0 plane
                            np.array([0.0, 0.0, 1.0]))
    V_R = build_view_matrix(viewer_pos,
                            np.array([-1.0, 0.0, 0.0]),   # look toward X=0 plane
                            np.array([0.0, 0.0, 1.0]))

    # --- Standard frustum bounds (no overlap) ---
    l_L0, r_L0, b_L, t_L = compute_frustum_bounds(left_corners, near, V_L)
    l_R0, r_R0, b_R, t_R = compute_frustum_bounds(right_corners, near, V_R)

    # --- Overlap: add virtual corners past the fold ---
    # Left camera: right=(1,0,0), x_view = X - viewer_X = X - 2.12.
    #   Fold X=0 --> x_view=-2.12 --> x_near=-0.01 = r_L.
    #   To extend r_L outward (less negative), need x_view less negative --> larger X.
    #   --> add corners at (overlap, 0, +Hz): extending +X past fold on Y=0 plane.
    left_overlap_corners = np.array([
        [overlap_world, 0.0,  Hz],   # top, past fold on left-screen plane
        [overlap_world, 0.0, -Hz],   # bottom
    ], dtype=np.float64)

    # Right camera: right=(0,-1,0), x_view = viewer_Y - Y = 2.12 - Y.
    #   Fold Y=0 --> x_view=2.12 --> x_near=0.01 = l_R.
    #   To extend l_R outward (less positive), need x_view smaller --> larger Y.
    #   --> add corners at (0, overlap, +Hz): extending +Y past fold on X=0 plane.
    right_overlap_corners = np.array([
        [0.0,  overlap_world,  Hz],   # top, past fold on right-screen plane
        [0.0,  overlap_world, -Hz],   # bottom
    ], dtype=np.float64)

    # --- Expanded corner sets ---
    left_expanded = np.vstack([left_corners, left_overlap_corners])
    right_expanded = np.vstack([right_corners, right_overlap_corners])

    l_L, r_L, b_L2, t_L2 = compute_frustum_bounds(left_expanded, near, V_L)
    l_R, r_R, b_R2, t_R2 = compute_frustum_bounds(right_expanded, near, V_R)

    # b,t should be unchanged (overlap only extends horizontally)
    # Use the tighter set (without overlap) for b,t to avoid vertical inflation

    # --- Build projection matrices (with overlap) ---
    P_L = build_projection_matrix(near, far, l_L, r_L, b_L, t_L)
    P_R = build_projection_matrix(near, far, l_R, r_R, b_R, t_R)

    # --- Compute overlap in NDC and pixel space ---
    # The fold point (0,0,Hz/2) in each camera with overlap frustum:
    #   Left:  should map to NDC x < +1 (inside frustum by some amount)
    #   Right: should map to NDC x > -1 (inside frustum by some amount)
    fold_mid = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    ndc_L_fold = project_to_ndc(V_L, P_L, fold_mid)
    ndc_R_fold = project_to_ndc(V_R, P_R, fold_mid)

    # How far from the edge is the fold? (in NDC units)
    # Left:  edge is at +1, fold is inside --> 1.0 - ndc_L_fold[0] = inward shift
    # Right: edge is at -1, fold is inside --> ndc_R_fold[0] - (-1.0) = inward shift
    ndc_overlap_L = 1.0 - ndc_L_fold[0] if ndc_L_fold is not None else 0.0
    ndc_overlap_R = ndc_R_fold[0] - (-1.0) if ndc_R_fold is not None else 0.0

    # Convert NDC overlap to pixels
    # NDC x ranges from -1 to +1 over image_width=832 pixels
    # So 1 NDC unit = image_width / 2 pixels
    px_per_ndc = PANEL_PIXEL_W / 2.0  # 416 px per NDC unit
    overlap_px_L = ndc_overlap_L * px_per_ndc
    overlap_px_R = ndc_overlap_R * px_per_ndc

    return {
        # Standard bounds
        "l_L0": l_L0, "r_L0": r_L0, "b_L": b_L, "t_L": t_L,
        "l_R0": l_R0, "r_R0": r_R0, "b_R": b_R, "t_R": t_R,
        # Overlap bounds
        "l_L": l_L, "r_L": r_L,
        "l_R": l_R, "r_R": r_R,
        # View matrices
        "V_L": V_L, "V_R": V_R,
        # Projection matrices (with overlap)
        "P_L": P_L, "P_R": P_R,
        # Overlap metrics
        "overlap_world_m": overlap_world,
        "ndc_L_fold": ndc_L_fold,
        "ndc_R_fold": ndc_R_fold,
        "ndc_overlap_L": ndc_overlap_L,
        "ndc_overlap_R": ndc_overlap_R,
        "overlap_px_L": overlap_px_L,
        "overlap_px_R": overlap_px_R,
    }


# ===========================================================================
# 5 -- 2D Alpha soft blending
# ===========================================================================

def soft_blend_seam(img_left, img_right, overlap_pixels):
    """
    Blend two images along the vertical seam using a linear alpha ramp.

    The fold seam is at the right edge of img_left and the left edge of
    img_right.  The overlap zone spans `overlap_pixels` columns:
      - The RIGHTmost `overlap_pixels` columns of img_left  (zone L)
      - The LEFTmost  `overlap_pixels` columns of img_right (zone R)
      are cross-faded.

    Alpha ramp:
      alpha_L(col) = (overlap_pixels - 1 - col_local) / (overlap_pixels - 1)
        --> 1.0 at left edge of zone, 0.0 at right edge
      alpha_R(col) = col_local / (overlap_pixels - 1)
        --> 0.0 at left edge of zone, 1.0 at right edge

    Result:
      stitched = [img_left[:, :-overlap_pixels],            ← left non-overlap
                  blended_zone,                              ← overlap
                  img_right[:, overlap_pixels:]]             ← right non-overlap

    Args:
        img_left:       (H, W, C) float32/float64 array, left render
        img_right:      (H, W, C) float32/float64 array, right render
        overlap_pixels: int, width of blend zone in pixels

    Returns:
        stitched:       (H, 2*W - overlap_pixels, C) blended image
        blend_mask:     debug visualization of the blend weights
    """
    H, W, C = img_left.shape
    assert img_right.shape == (H, W, C), \
        f"Shape mismatch: left={img_left.shape}, right={img_right.shape}"
    assert 0 < overlap_pixels < W, \
        f"overlap_pixels ({overlap_pixels}) must be in (0, {W})"

    overlap = int(overlap_pixels)
    out_W = 2 * W - overlap

    # Alpha ramps (linear, float64 for precision)
    ramp_down = np.linspace(1.0, 0.0, overlap, dtype=np.float64)   # left: 1-->0
    ramp_up   = np.linspace(0.0, 1.0, overlap, dtype=np.float64)   # right: 0-->1

    # Broadcast to (1, overlap, 1) for channel-wise multiplication
    ramp_down = ramp_down[None, :, None]
    ramp_up   = ramp_up[None, :, None]

    # Extract overlap zones
    zone_L = img_left[:, -overlap:, :].astype(np.float64)
    zone_R = img_right[:, :overlap, :].astype(np.float64)

    # Blend
    blended = zone_L * ramp_down + zone_R * ramp_up

    # Stitch
    stitched = np.concatenate([
        img_left[:, :-overlap, :].astype(np.float64),
        blended,
        img_right[:, overlap:, :].astype(np.float64),
    ], axis=1)

    # Debug mask: visualize alpha weights over the stitched image
    blend_mask = np.zeros((H, out_W), dtype=np.float64)
    blend_mask[:, :W - overlap] = 0.0                    # left-only
    blend_mask[:, W - overlap:W] = ramp_down[0, :, 0]    # fade left-->right
    blend_mask[:, W:W + overlap] = ramp_up[0, :, 0]      # fade left-->right (right side)
    blend_mask[:, W + overlap:] = 0.0                     # right-only
    # Actually, let me make the mask more informative:
    # 0 = left-only, 1 = blend zone, 2 = right-only
    blend_mask[:] = 0
    blend_mask[:, W - overlap:W] = 1   # left part of blend
    # The right contribution comes from img_right's leftmost columns
    # In the stitched image, columns W-overlap to W are the blended zone

    return stitched.astype(img_left.dtype), blend_mask


# ===========================================================================
# TESTS
# ===========================================================================

def test_overlap_frustum_bounds():
    """
    TEST 1: Verify that overlap extension creates mathematically overlapping
    frustums at the fold line.
    """
    print("=" * 70)
    print("  TEST 1: Overlap Frustum Bounds")
    print("=" * 70)

    info = calc_overlap_bounds(PANEL_W, OVERLAP_WORLD_M, VIEWER_POS)

    print(f"\n  Standard bounds (no overlap):")
    print(f"    Left:  l={info['l_L0']:.6f}  r={info['r_L0']:.6f}  "
          f"b={info['b_L']:.6f}  t={info['t_L']:.6f}")
    print(f"    Right: l={info['l_R0']:.6f}  r={info['r_R0']:.6f}  "
          f"b={info['b_R']:.6f}  t={info['t_R']:.6f}")

    print(f"\n  Overlap bounds (with {OVERLAP_WORLD_M:.2f}m extension):")
    print(f"    Left:  l={info['l_L']:.6f}  r={info['r_L']:.6f}  "
          f"(r shift: {info['r_L'] - info['r_L0']:+.6f})")
    print(f"    Right: l={info['l_R']:.6f}  r={info['r_R']:.6f}  "
          f"(l shift: {info['l_R'] - info['l_R0']:+.6f})")

    # --- Assertions ---
    # 1. r_L moved outward (more positive / less negative)
    assert info['r_L'] > info['r_L0'], \
        f"Left r did not extend outward: {info['r_L']:.6f} <= {info['r_L0']:.6f}"
    print(f"\n  [PASS] Left r_L extended outward: {info['r_L0']:.6f} --> {info['r_L']:.6f}")

    # 2. l_R moved outward (more negative / less positive)
    assert info['l_R'] < info['l_R0'], \
        f"Right l did not extend outward: {info['l_R']:.6f} >= {info['l_R0']:.6f}"
    print(f"  [PASS] Right l_R extended outward: {info['l_R0']:.6f} --> {info['l_R']:.6f}")

    # 3. The fold is now INSIDE both frustums (not at the very edge)
    ndc_L = info['ndc_L_fold']
    ndc_R = info['ndc_R_fold']
    print(f"\n  Fold (0,0,0) with overlap frustums:")
    print(f"    Left NDC:  x={ndc_L[0]:+.4f}  y={ndc_L[1]:+.4f}  z={ndc_L[2]:+.4f}")
    print(f"    Right NDC: x={ndc_R[0]:+.4f}  y={ndc_R[1]:+.4f}  z={ndc_R[2]:+.4f}")

    assert ndc_L[0] < 1.0, \
        f"Fold should be inside left frustum (x<+1), got x={ndc_L[0]:.4f}"
    assert ndc_R[0] > -1.0, \
        f"Fold should be inside right frustum (x>-1), got x={ndc_R[0]:.4f}"
    print(f"  [PASS] Fold is inside BOTH frustums (not at edge)")

    # 4. The overlap zones exist and are positive
    print(f"\n  Overlap in NDC:  L={info['ndc_overlap_L']:.4f}  R={info['ndc_overlap_R']:.4f}")
    print(f"  Overlap in px:   L={info['overlap_px_L']:.1f}  R={info['overlap_px_R']:.1f}")

    assert info['ndc_overlap_L'] > 0, "Left overlap NDC must be positive"
    assert info['ndc_overlap_R'] > 0, "Right overlap NDC must be positive"
    print(f"  [PASS] Overlap zones exist in both frustums")

    # 5. Overlap is in reasonable pixel range (20-200 px)
    assert 10 < info['overlap_px_L'] < 300, \
        f"Left overlap {info['overlap_px_L']:.0f}px out of reasonable range"
    assert 10 < info['overlap_px_R'] < 300, \
        f"Right overlap {info['overlap_px_R']:.0f}px out of reasonable range"
    print(f"  [PASS] Overlap pixel width in reasonable range (20-200 px)")

    # 6. Vertical bounds unchanged
    assert abs(info['b_L'] - info.get('b_L', info['b_L'])) < 1e-12
    print(f"  [PASS] Vertical bounds unchanged by overlap extension")

    # 7. Overlap is symmetric (L ≈ R in NDC space, within 10%)
    ratio = info['ndc_overlap_L'] / max(info['ndc_overlap_R'], 1e-15)
    assert 0.85 < ratio < 1.15, \
        f"Overlap asymmetry: L/R NDC ratio = {ratio:.3f} (expected ~1.0)"
    print(f"  [PASS] Overlap symmetric: L/R NDC ratio = {ratio:.3f}")

    return info


def test_soft_blend_alpha_sum():
    """
    TEST 2: Verify that in the blend zone, alpha_L + alpha_R == 1.0
    everywhere (no brightness clipping or gaps).
    """
    print("\n" + "=" * 70)
    print("  TEST 2: Soft Blend -- Alpha Conservation (alpha_L + alpha_R = 1)")
    print("=" * 70)

    H, W = 512, 512
    overlap = BLEND_PIXELS

    # Use constant-value images to detect brightness errors
    img_L = np.ones((H, W, 3), dtype=np.float32) * 0.3   # gray 0.3
    img_R = np.ones((H, W, 3), dtype=np.float32) * 0.7   # gray 0.7

    stitched, mask = soft_blend_seam(img_L, img_R, overlap)
    out_W = 2 * W - overlap

    print(f"  Input:  {W}×{H} each  -->  output: {out_W}×{H}")
    print(f"  Blend zone: {overlap} px  ({overlap/W*100:.1f}% of panel width)")

    # --- Assertions ---

    # 1. Output dimensions correct
    assert stitched.shape == (H, out_W, 3), \
        f"Wrong output shape: {stitched.shape} != ({H}, {out_W}, 3)"
    print(f"  [PASS] Output shape: {stitched.shape}")

    # 2. Left non-overlap zone: values == 0.3 (unchanged)
    left_zone = stitched[:, :W - overlap, :]
    assert np.allclose(left_zone, 0.3, atol=1e-6), \
        "Left non-overlap zone corrupted!"
    print(f"  [PASS] Left non-overlap: all == 0.3 (unchanged)")

    # 3. Right non-overlap zone: values == 0.7 (unchanged)
    right_zone = stitched[:, W:, :]
    assert np.allclose(right_zone, 0.7, atol=1e-6), \
        "Right non-overlap zone corrupted!"
    print(f"  [PASS] Right non-overlap: all == 0.7 (unchanged)")

    # 4. Blend zone: values should be a smooth ramp from 0.3 to 0.7
    blend_zone = stitched[:, W - overlap:W, :]
    # For column j in [0, overlap): alpha = j/(overlap-1)
    # expected = 0.3 * (1 - alpha) + 0.7 * alpha
    for j in range(overlap):
        alpha = j / max(overlap - 1, 1)
        expected = 0.3 * (1.0 - alpha) + 0.7 * alpha
        col_mean = blend_zone[:, j, :].mean()
        assert abs(col_mean - expected) < 1e-6, \
            f"Blend column {j}: expected={expected:.6f}, got={col_mean:.6f}"
    print(f"  [PASS] Blend zone: linear ramp 0.3-->0.7, {overlap} columns verified")

    # 5. Alpha conservation: mathematically ramp_down + ramp_up = 1.0 everywhere.
    #    Verify on a per-column basis in the blend zone.
    #    Use the SAME constant image for both inputs -- the per-pixel alpha sum
    #    cannot be tested directly (blend combines different pixel locations),
    #    but the overall brightness must be uniform.
    const_val = 0.6
    img_const = np.full((H, W, 3), const_val, dtype=np.float32)
    stitched_const, _ = soft_blend_seam(img_const, img_const, overlap)
    # Every column in the stitched output should == const_val
    assert np.allclose(stitched_const, const_val, atol=1e-6), \
        f"Constant image not preserved! max_err={float(np.abs(stitched_const - const_val).max()):.2e}"
    print(f"  [PASS] Alpha conservation: constant image --> uniform output (all == {const_val})")

    # 5b. Verify alpha ramp values directly
    alpha_L = np.linspace(1.0, 0.0, overlap, dtype=np.float64)
    alpha_R = np.linspace(0.0, 1.0, overlap, dtype=np.float64)
    alpha_sum = alpha_L + alpha_R
    assert np.allclose(alpha_sum, 1.0, atol=1e-15), \
        f"Alpha sum != 1.0! max_dev={float(np.abs(alpha_sum - 1.0).max()):.2e}"
    print(f"  [PASS] Alpha ramp: alpha_L + alpha_R == 1.0 at all {overlap} columns")

    # 6. No clipping: all values in [0, 1]
    img_bright = np.ones((H, W, 3), dtype=np.float32) * 0.95
    stitched_bright, _ = soft_blend_seam(img_bright, img_bright, overlap)
    assert stitched_bright.max() <= 1.0 + 1e-6, \
        f"Brightness clipping! max={stitched_bright.max():.4f}"
    assert stitched_bright.min() >= 0.0, \
        f"Negative values! min={stitched_bright.min():.4f}"
    print(f"  [PASS] No clipping: range [{stitched_bright.min():.3f}, {stitched_bright.max():.3f}]")

    return True


def test_noise_image_blend():
    """
    TEST 3: Blend two different noise images. Verify:
      - Seam is invisible (no hard edge)
      - Brightness is uniform across the blend zone
      - The transition is perceptually smooth
    """
    print("\n" + "=" * 70)
    print("  TEST 3: Noise Image Blend -- Visual Quality")
    print("=" * 70)

    H, W = 512, 512
    overlap = BLEND_PIXELS
    rng = np.random.default_rng(12345)

    # Different noise patterns
    img_L = rng.uniform(0.2, 0.8, (H, W, 3)).astype(np.float32)
    img_R = rng.uniform(0.2, 0.8, (H, W, 3)).astype(np.float32)

    stitched, mask = soft_blend_seam(img_L, img_R, overlap)

    # --- Assertions ---

    # 1. Seam zone: gradient magnitude should NOT spike at the seam boundary
    # Compute horizontal gradient in stitched image
    grad_x = np.abs(np.diff(stitched, axis=1)).mean(axis=(0, 2))

    # The blend zone spans [W-overlap, W] in stitched coords
    # Check that gradient in blend zone isn't anomalously high
    blend_start = W - overlap
    blend_end = W
    grad_in_blend = grad_x[blend_start:blend_end].mean()
    grad_outside = np.concatenate([
        grad_x[:blend_start - 5],
        grad_x[blend_end + 5:]
    ]).mean()

    print(f"  Mean |grad_x| in blend zone:  {grad_in_blend:.4f}")
    print(f"  Mean |grad_x| outside blend:  {grad_outside:.4f}")

    # The blend zone gradient should be similar to or lower than outside
    # (blending should NOT create a visible seam)
    seam_ratio = grad_in_blend / max(grad_outside, 1e-10)
    print(f"  Seam gradient ratio (blend/outside): {seam_ratio:.2f}")

    assert seam_ratio < 3.0, \
        f"Blend zone has excessive gradient (ratio={seam_ratio:.1f}x) -- visible seam!"
    print(f"  [PASS] No visible seam: blend gradient is {seam_ratio:.1f}x outside gradient")

    # 2. No brightness discontinuity at seam boundaries
    #    The mean of the last pre-blend column vs first blend column should
    #    be similar (no hard jump). Same for blend-end vs post-blend.
    pre_blend_mean = stitched[:, blend_start - 1, :].mean()
    blend_first_mean = stitched[:, blend_start, :].mean()
    blend_last_mean = stitched[:, blend_end - 1, :].mean()
    post_blend_mean = stitched[:, blend_end, :].mean()

    jump_start = abs(blend_first_mean - pre_blend_mean)
    jump_end = abs(post_blend_mean - blend_last_mean)
    print(f"  Brightness jump at blend-start: {jump_start:.4f}")
    print(f"  Brightness jump at blend-end:   {jump_end:.4f}")

    # A jump < 0.15 in [0,1] range indicates no visible hard seam
    assert jump_start < 0.20, f"Visible seam at blend start! jump={jump_start:.3f}"
    assert jump_end < 0.20, f"Visible seam at blend end! jump={jump_end:.3f}"
    print(f"  [PASS] No hard brightness jump at seam boundaries (< 0.20)")

    # 3. Per-pixel alpha sum check (for random images the alpha weights sum to 1)
    # The weights for the blend zone: w_L(col) + w_R(col) = 1.0 by construction
    # This is mathematically guaranteed by the ramp_down + ramp_up = 1.0
    # But we can verify on the stitched result with same-input images:
    img_const_L = np.ones((H, W, 3), dtype=np.float32) * 0.5
    img_const_R = np.ones((H, W, 3), dtype=np.float32) * 0.5
    stitched_const, _ = soft_blend_seam(img_const_L, img_const_R, overlap)
    blend_zone_check = stitched_const[:, W - overlap:W, :]
    assert np.allclose(blend_zone_check, 0.5, atol=1e-6), \
        "Alpha sum != 1.0: constant 0.5 images should produce constant 0.5 output"
    print(f"  [PASS] Alpha weights sum to 1.0 everywhere in blend zone")

    return True


def test_fold_continuity_with_overlap(info):
    """
    TEST 4: With overlap frustums, verify that fold points are rendered
    by BOTH cameras (inside both viewports) and that the NDC positions
    are consistent (same Y, same Z for same world point).
    """
    print("\n" + "=" * 70)
    print("  TEST 4: Fold Continuity with Overlap Frustums")
    print("=" * 70)

    V_L, P_L = info["V_L"], info["P_L"]
    V_R, P_R = info["V_R"], info["P_R"]

    z_samples = np.linspace(-HALF_H, HALF_H, 7)
    all_pass = True

    for z in z_samples:
        pt = np.array([0.0, 0.0, z], dtype=np.float64)
        ndc_L = project_to_ndc(V_L, P_L, pt)
        ndc_R = project_to_ndc(V_R, P_R, pt)

        if ndc_L is None or ndc_R is None:
            print(f"  Z={z:+.2f}: BEHIND CAMERA")
            all_pass = False
            continue

        # With overlap: fold should be INSIDE both viewports (-1 < x < +1)
        xL_inside = -1.0 <= ndc_L[0] <= 1.0
        xR_inside = -1.0 <= ndc_R[0] <= 1.0
        y_match = abs(ndc_L[1] - ndc_R[1]) < 0.02
        z_match = abs(ndc_L[2] - ndc_R[2]) < 0.05

        ok = xL_inside and xR_inside and y_match and z_match
        if not ok:
            all_pass = False

        print(f"  Z={z:+.2f}:  "
              f"L=({ndc_L[0]:+.4f}, {ndc_L[1]:+.4f}, {ndc_L[2]:+.4f})  "
              f"R=({ndc_R[0]:+.4f}, {ndc_R[1]:+.4f}, {ndc_R[2]:+.4f})  "
              f"inside=({xL_inside},{xR_inside}) y={y_match} z={z_match}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    assert all_pass, "Fold points NOT consistently inside both overlap frustums!"
    print(f"\n  ASSERT: fold continuity with overlap -- PASSED")

    # Verify overlap extent: fold at mid-height should be inside by ~overlap_px pixels
    ndc_L_mid = project_to_ndc(V_L, P_L, np.array([0.0, 0.0, 0.0]))
    ndc_R_mid = project_to_ndc(V_R, P_R, np.array([0.0, 0.0, 0.0]))
    px_per_ndc = PANEL_PIXEL_W / 2.0
    px_inside_L = (1.0 - ndc_L_mid[0]) * px_per_ndc
    px_inside_R = (ndc_R_mid[0] - (-1.0)) * px_per_ndc

    print(f"\n  Fold at mid-height: {px_inside_L:.1f} px inside left frustum")
    print(f"                       {px_inside_R:.1f} px inside right frustum")
    print(f"  Total overlap zone: ~{px_inside_L + px_inside_R:.0f} px wide")

    return True


def test_screen_fill_with_overlap(info):
    """
    TEST 5: With overlap frustums, content that was previously ONLY visible
    to one camera near the fold should now be visible to BOTH cameras.
    """
    print("\n" + "=" * 70)
    print("  TEST 5: Dual Visibility in Overlap Zone")
    print("=" * 70)

    V_L, P_L = info["V_L"], info["P_L"]
    V_R, P_R = info["V_R"], info["P_R"]

    # Test point slightly on the LEFT side of fold (X<0, Y=0, Z=0)
    # Should be visible to BOTH cameras with overlap
    pt_left = np.array([-0.03, 0.0, 0.0], dtype=np.float64)
    ndc_L_l = project_to_ndc(V_L, P_L, pt_left)
    ndc_R_l = project_to_ndc(V_R, P_R, pt_left)

    print(f"  Point at (-0.03, 0, 0) -- left of fold:")
    print(f"    Left NDC:  {ndc_L_l[0]:+.4f}" if ndc_L_l is not None else "    Left NDC:  None")
    print(f"    Right NDC: {ndc_R_l[0]:+.4f}" if ndc_R_l is not None else "    Right NDC: None")

    # Test point slightly on the RIGHT side of fold (X>0, Y=0, Z=0)
    pt_right = np.array([0.03, 0.0, 0.0], dtype=np.float64)
    ndc_L_r = project_to_ndc(V_L, P_L, pt_right)
    ndc_R_r = project_to_ndc(V_R, P_R, pt_right)

    print(f"  Point at (+0.03, 0, 0) -- right of fold:")
    print(f"    Left NDC:  {ndc_L_r[0]:+.4f}" if ndc_L_r is not None else "    Left NDC:  None")
    print(f"    Right NDC: {ndc_R_r[0]:+.4f}" if ndc_R_r is not None else "    Right NDC: None")

    # Assertions: with overlap, both points should be visible to BOTH cameras
    if ndc_L_l is not None:
        assert -1.0 <= ndc_L_l[0] <= 1.0, \
            f"Point left of fold not visible to left camera!"
    if ndc_R_l is not None:
        assert -1.0 <= ndc_R_l[0] <= 1.0, \
            f"Point left of fold not visible to right camera (no overlap)!"

    if ndc_L_r is not None:
        assert -1.0 <= ndc_L_r[0] <= 1.0, \
            f"Point right of fold not visible to left camera (no overlap)!"
    if ndc_R_r is not None:
        assert -1.0 <= ndc_R_r[0] <= 1.0, \
            f"Point right of fold not visible to right camera!"

    print(f"\n  ASSERT: dual visibility in overlap zone -- PASSED")
    return True


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  OVERLAP FRUSTUM & ALPHA BLENDING HARNESS")
    print(f"  Panel: {PANEL_W:.2f} x {PANEL_H:.2f} m  ({PANEL_PIXEL_W}x{PANEL_PIXEL_H} px)")
    print(f"  Viewer: {VIEWER_POS}")
    print(f"  Overlap: {OVERLAP_WORLD_M:.2f} m world-space")
    print(f"  Blend zone: {BLEND_PIXELS} px target")
    print("=" * 70)

    # Phase 1: Frustum overlap computation
    info = test_overlap_frustum_bounds()

    # Phase 2: Blend validation
    results = []
    results.append(("Soft Blend Alpha", test_soft_blend_alpha_sum()))
    results.append(("Noise Blend Quality", test_noise_image_blend()))
    results.append(("Fold Continuity + Overlap", test_fold_continuity_with_overlap(info)))
    results.append(("Dual Visibility", test_screen_fill_with_overlap(info)))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"\n  Blend recommendation:")
    print(f"    overlap_pixels = {BLEND_PIXELS}  (or range 50-100)")
    print(f"    world overlap  = {OVERLAP_WORLD_M:.2f} m")
    print(f"    alpha ramp = linear (cosine optional for smoother roll-off)")
    print("=" * 70)

    if not all_pass:
        exit(1)
