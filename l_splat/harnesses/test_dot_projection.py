"""
test_dot_projection.py v3 — Geometric Round-Trip with corrected u_norm.

After the 2026-05-19 u_norm fix: input left edge → X<0 (left half-space),
so a black square at input top-left should appear in the LEFT panel's top-left.
"""

import os, sys, tempfile, numpy as np, cv2


def generate_test_image(H=600, W=800, corner_size=60):
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    img[0:corner_size, 0:corner_size] = 0
    return img


def test_dot_projection():
    print("=" * 64)
    print("  DOT PROJECTION TEST v3 — u_norm FIXED")
    print("=" * 64)

    from l_splat.core.pipeline import process_frame, build_dual_camera
    from l_splat.depth.loader import load_dav3
    from l_splat.depth.anchor import DepthAnchorStream
    from l_splat.core.pipeline import DEPTH_FWD_M, DEPTH_BACK_M

    H, W = 600, 800
    corner_sz = 60
    rgb = generate_test_image(H, W, corner_size=corner_sz)
    cv2.imwrite("_dot_input.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  Input: {W}x{H}, black square at TOP-LEFT [0:{corner_sz}, 0:{corner_sz}]")

    model_dav3 = load_dav3("da3-small", device="cuda")
    camera = build_dual_camera()
    depth_anchor = DepthAnchorStream(
        physical_near=DEPTH_FWD_M, physical_far=DEPTH_BACK_M, alpha=0.05)

    with tempfile.TemporaryDirectory() as tmpdir:
        process_frame(rgb, model_dav3, camera,
                      out_dir=tmpdir, frame_idx=None, depth_anchor=depth_anchor)
        left = cv2.imread(os.path.join(tmpdir, "render_left.png"))
        right = cv2.imread(os.path.join(tmpdir, "render_right.png"))
        cv2.imwrite("_dot_left.png", left)
        cv2.imwrite("_dot_right.png", right)

    h, w = left.shape[:2]
    tl_u, tl_v = w // 4, h // 4
    br_u, br_v = w * 3 // 4, h * 3 // 4

    all_pass = True

    for label, img in [("Left", left), ("Right", right)]:
        gray = img.mean(axis=2)
        tl_mean = gray[0:tl_v, 0:tl_u].mean()
        tr_mean = gray[0:tl_v, w//2:w].mean()
        bl_mean = gray[h//2:h, 0:tl_u].mean()
        br_mean = gray[h//2:h, br_u:w].mean()

        print(f"\n  --- {label} Panel ---")
        print(f"    Top-Left:    {tl_mean:5.0f}")
        print(f"    Top-Right:   {tr_mean:5.0f}")
        print(f"    Bottom-Left: {bl_mean:5.0f}")
        print(f"    Bottom-Right:{br_mean:5.0f}")

        # After fix: black square (input top-left) → LEFT panel top-left
        if label == "Left":
            tl_dark = tl_mean < tr_mean  # top-left darker than top-right
            print(f"    Top-Left < Top-Right (black square here): {tl_dark} "
                  f"[{'PASS' if tl_dark else 'FAIL - wrong side!'}]")
            if not tl_dark:
                all_pass = False

    print(f"\n  OVERALL: {'ALL PASSED' if all_pass else 'FAILED - mirror still present!'}")
    print("=" * 64)
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    test_dot_projection()
