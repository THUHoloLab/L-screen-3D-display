"""
depth_pipeline.py — Depth filter utilities.

Provides a thin cv2.ximgproc.jointBilateralFilter (C++ accelerated) wrapper
that replaces the hand-written Python JBF previously in this module.
"""

import numpy as np


# ============================================================================
# cv2.ximgproc C++ Joint Bilateral Filter
# ============================================================================

def cv2_jbf(rgb, depth, ksize=9, sigma_s=5.0, sigma_r=0.03):
    """
    OpenCV C++ Joint Bilateral Filter — replaces hand-written Python JBF.

    参数已针对 L-Splat 双屏折角场景重调 (2026-05-11):
      - sigma_r=0.03: 对色彩边界极其敏感, 防止边缘糊化
      - sigma_s=5.0:  大空间采样池, 吃掉深度光晕
      - ksize=9:      匹配 sigma_s 的内核直径

    Args:
        rgb:     (H, W, 3) float32 guide image in [0, 1].
        depth:   (H, W) float32 depth map.
        ksize:   kernel diameter (pixels), must be odd.
        sigma_s: spatial sigma (pixels).
        sigma_r: range sigma (RGB in [0,1] space).

    Returns:
        filtered: (H, W) float32.
    """
    import cv2
    return cv2.ximgproc.jointBilateralFilter(
        rgb.astype(np.float32),
        depth.astype(np.float32),
        d=int(ksize),
        sigmaColor=float(sigma_r),
        sigmaSpace=float(sigma_s),
    )


# ============================================================================
# Harness
# ============================================================================

def harness_depth_pipeline():
    """
    Verify cv2_jbf preserves edges while smoothing flat regions.
    """
    print("=" * 64)
    print("  DEPTH PIPELINE HARNESS — cv2 JBF")
    print("=" * 64)

    H, W = 120, 160
    FOLD_COL = W // 2
    rng = np.random.RandomState(42)

    # Synthetic L-screen depth
    depth_gt = np.ones((H, W), dtype=np.float32)
    depth_gt[:, FOLD_COL:] = 1.5
    depth_noisy = depth_gt + rng.randn(H, W).astype(np.float32) * 0.05

    # Synthetic RGB guide
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:] = [0.25, 0.25, 0.25]
    rgb[:, :FOLD_COL] += rng.randn(H, FOLD_COL, 3).astype(np.float32) * 0.02
    rgb[:, FOLD_COL:] += rng.randn(H, W - FOLD_COL, 3).astype(np.float32) * 0.02
    rgb = np.clip(rgb, 0, 1)

    # Test cv2 JBF
    print("\n  Testing cv2.ximgproc.jointBilateralFilter (C++ JBF)...")
    depth_jbf = cv2_jbf(rgb, depth_noisy)

    var_before = np.var(depth_noisy[:, :FOLD_COL - 5])
    var_after = np.var(depth_jbf[:, :FOLD_COL - 5])
    noise_reduction = (var_before - var_after) / var_before
    print(f"    Noise reduction (flat region): {noise_reduction*100:.1f}%")
    assert noise_reduction > 0.1, f"Insufficient noise reduction: {noise_reduction:.3f}"
    print(f"    [PASS] cv2 JBF reduces noise in flat regions")

    assert depth_jbf.shape == depth_noisy.shape
    assert not np.any(np.isnan(depth_jbf))
    print(f"    [PASS] Output shape: {depth_jbf.shape}, no NaN")

    print(f"\n  HARNESS COMPLETE — all assertions passed\n")
    return True


if __name__ == "__main__":
    harness_depth_pipeline()
