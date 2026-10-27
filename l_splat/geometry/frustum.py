"""
utils/geometry_utils.py
Shared geometry functions for L-shaped 90-degree outward-fold screen rendering.

Canonical source of truth for:
  - View matrix construction (Candidate H: look_at_offscreen + flip_y)
  - Asymmetric frustum boundary computation
  - Asymmetric projection matrix building (numpy backend)
  - L-screen corner geometry

All functions are numpy-based (no torch dependency) so harness files can
import them without CUDA.  This module replaces the local copies scattered
across harness_physical_frustum.py, harness_overlap_blending.py,
harness_camera_extrinsics.py, and smoke_test_orientation.py.
"""

import numpy as np

# ===========================================================================
# Flip-Y Correction
# ===========================================================================
# look_at_offscreen produces det=+1 with correct fold alignment, but the
# image comes out upside-down (up = (0,0,-1) in view space).
# FLIP_Y = diag(1, -1, 1, 1) negates camera-space Y (row 1), giving an
# upright image.  Trade-off: det becomes -1.
#
# This is the "flip_y" step in Candidate H, applied as: V_final = FLIP_Y @ V.

FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])


def apply_flip_y(V):
    """Apply flip-y correction to a view matrix.  Returns FLIP_Y @ V."""
    return FLIP_Y @ np.asarray(V, dtype=np.float64)


# ===========================================================================
# Flip-X (Horizontal Mirror) Correction
# ===========================================================================
# For the patent-aligned L-screen, the left panel (Y=0) is viewed from +Y, and
# the right panel (X=0) is viewed from +X.  Without horizontal mirroring, content
# on one panel may appear reversed from the viewer's perspective.
#
# FLIP_X = diag(-1, 1, 1, 1) negates camera-space X (row 0), providing a
# horizontal mirror.  det becomes -1.  Applies as: V_mirrored = FLIP_X @ V.
#
# Properties:
#   - self-inverse: FLIP_X @ FLIP_X = I
#   - commutes with FLIP_Y: FLIP_X @ FLIP_Y = FLIP_Y @ FLIP_X = diag(-1,-1,1,1)
#   - combined det = +1 (proper rotation after both flips)

FLIP_X = np.diag([-1.0, 1.0, 1.0, 1.0])


def apply_flip_x(V):
    """Apply horizontal mirror to a view matrix.  Returns FLIP_X @ V."""
    return FLIP_X @ np.asarray(V, dtype=np.float64)


# ===========================================================================
# View Matrix
# ===========================================================================

def look_at_offscreen(eye, target, up):
    """
    Right-handed look-at view matrix (world -> view).  **Candidate B.**

    Camera at `eye`, looking at `target`.
    View-space -Z axis points toward `target`.

    Uses ``side = cross(up, fwd)`` for proper right-handed coordinates
    and correct fold alignment on L-shaped screens.  Re-orthogonalises
    up via ``cross(side, fwd)`` for numerical stability.

    Args:
        eye:    (3,) camera position in world space.
        target: (3,) point the camera looks at.
        up:     (3,) world up direction (typically (0,0,1) for Z-up).

    Returns:
        V: (4, 4) view matrix, column-vector convention: p_view = V @ p_world.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    u = up / np.linalg.norm(up)
    side = np.cross(u, fwd)
    side = side / np.linalg.norm(side)
    u_ortho = np.cross(side, fwd)

    V = np.eye(4, dtype=np.float64)
    V[0, :3] = side
    V[1, :3] = u_ortho
    V[2, :3] = -fwd
    V[:3, 3] = [-np.dot(side, eye), -np.dot(u_ortho, eye), np.dot(fwd, eye)]
    return V


# ===========================================================================
# Frustum Boundaries
# ===========================================================================

def compute_frustum_bounds(viewer_pos, screen_corners, near, view_matrix=None):
    """
    Compute asymmetric frustum boundaries (l, r, b, t) on the near plane.

    Projects rays from viewer through each screen corner onto the near plane.
    Uses OpenGL convention: camera looks down -Z, near plane at z = -near.

    Args:
        viewer_pos:     (3,) viewer position in world space.
        screen_corners: (N, 3) screen corners in world space (N >= 4).
        near:           near plane distance (> 0).
        view_matrix:    optional (4, 4) pre-computed view matrix.  If None,
                        computed via look_at_offscreen toward screen centre.

    Returns:
        (l, r, b, t): frustum boundaries on the near plane.
        view_matrix:   (4, 4) the view matrix used.
    """
    viewer_pos = np.asarray(viewer_pos, dtype=np.float64)
    screen_corners = np.asarray(screen_corners, dtype=np.float64)
    near = float(near)

    screen_center = np.mean(screen_corners, axis=0)
    screen_up = screen_corners[0] - screen_corners[2]
    screen_up = screen_up / np.linalg.norm(screen_up)

    if view_matrix is None:
        view_matrix = look_at_offscreen(viewer_pos, screen_center, screen_up)

    n_corners = screen_corners.shape[0]
    corners_h = np.hstack(
        [screen_corners, np.ones((n_corners, 1), dtype=np.float64)]
    )
    corners_view = (view_matrix @ corners_h.T).T

    z_vals = corners_view[:, 2]
    if np.any(z_vals >= 0):
        raise ValueError(
            f"Some screen corners behind camera: z_view={z_vals}. "
            f"Adjust viewer position or near distance."
        )

    t_vals = -near / z_vals
    x_near = corners_view[:, 0] * t_vals
    y_near = corners_view[:, 1] * t_vals

    return (
        float(np.min(x_near)),
        float(np.max(x_near)),
        float(np.min(y_near)),
        float(np.max(y_near)),
    ), view_matrix


# ===========================================================================
# Projection Matrix
# ===========================================================================

def build_asymmetric_proj_matrix(near, far, l, r, b, t):
    """
    Build an OpenGL-style asymmetric perspective projection matrix.

    Uses z_sign = -1 (OpenGL convention).  This is REQUIRED for off-axis
    asymmetric frustums --- the z_sign = +1 convention only works for
    symmetric projections.

    Matrix form (column-vector: p_clip = P @ p_view)::

        P = [[2n/(r-l),  0,        (r+l)/(r-l),  0         ],
             [0,          2n/(t-b), (t+b)/(t-b),  0         ],
             [0,          0,        -(f+n)/(f-n), -2fn/(f-n)],
             [0,          0,        -1,           0         ]]

    Args:
        near, far: near / far clip distances.
        l, r, b, t: frustum boundaries on the near plane.

    Returns:
        P:    (4, 4) projection matrix as numpy array.
        info: dict with A_x, A_y, offset_x, offset_y for CUDA Jacobian.
    """
    near, far = float(near), float(far)
    l, r, b, t = float(l), float(r), float(b), float(t)

    if r - l < 1e-12 or t - b < 1e-12:
        raise ValueError(f"Degenerate frustum: r-l={r - l:.3e}, t-b={t - b:.3e}")

    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * near / (r - l)
    P[1, 1] = 2.0 * near / (t - b)
    P[0, 2] = (r + l) / (r - l)
    P[1, 2] = (t + b) / (t - b)
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0

    info = {
        "l": l,
        "r": r,
        "b": b,
        "t": t,
        "z_sign": -1.0,
        "A_x": P[0, 0],
        "A_y": P[1, 1],
        "offset_x": P[0, 2],
        "offset_y": P[1, 2],
    }
    return P, info


# ===========================================================================
# Screen Geometry
# ===========================================================================

def build_l_screen_corners(physical_width, physical_height):
    """
    Build corner arrays for a 90-degree outward-fold L-shaped screen.

    Coordinate system (right-handed, **Z-up**):
        Fold line = Z axis (x=0, y=0).  +Z = UP.
        Left screen:  y=0 plane, extends in -x from the fold.
        Right screen: x=0 plane, extends in -y from the fold.

    Corner ordering: [top-left, top-right, bottom-left, bottom-right]
    where "top" = +Z, "bottom" = -Z.

    Args:
        physical_width:  physical width of each screen panel (meters).
        physical_height: physical height of each screen panel (meters).

    Returns:
        left_corners:  (4, 3) float64
        right_corners: (4, 3) float64
    """
    W, H = float(physical_width), float(physical_height)
    Hz = H / 2.0

    left_corners = np.array([
        [-W, 0,  Hz],
        [0,  0,  Hz],
        [-W, 0, -Hz],
        [0,  0, -Hz],
    ], dtype=np.float64)

    right_corners = np.array([
        [0,  0,  Hz],
        [0, -W,  Hz],
        [0,  0, -Hz],
        [0, -W, -Hz],
    ], dtype=np.float64)

    return left_corners, right_corners
