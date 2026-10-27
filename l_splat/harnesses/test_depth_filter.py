"""
test_depth_filter.py — HDD Phase 1 & 2 for cv2_jbf (C++ Joint Bilateral Filter).

Phase 1: Functional correctness — noise reduction, edge preservation, shape/NaN.
Phase 2: Performance — latency budget for interactive use.
"""

import time
import numpy as np


def test_noise_reduction():
    """Verify cv2 JBF reduces noise in flat regions by >10%."""
    from l_splat.core.depth_filter import cv2_jbf

    H, W = 120, 160
    rng = np.random.RandomState(42)

    depth_gt = np.ones((H, W), dtype=np.float32)
    depth_noisy = depth_gt + rng.randn(H, W).astype(np.float32) * 0.05

    rgb = np.full((H, W, 3), 0.5, dtype=np.float32)

    depth_jbf = cv2_jbf(rgb, depth_noisy)

    var_before = np.var(depth_noisy)
    var_after = np.var(depth_jbf)
    reduction = (var_before - var_after) / var_before
    assert reduction > 0.1, f"Noise reduction {reduction*100:.1f}% <= 10%"
    print(f"  [PASS] Noise reduction: {reduction*100:.1f}% > 10%")
    return True


def test_edge_preservation():
    """Verify cv2 JBF preserves a sharp depth edge."""
    from l_splat.core.depth_filter import cv2_jbf

    H, W = 120, 160
    fold = W // 2

    depth = np.ones((H, W), dtype=np.float32)
    depth[:, fold:] = 2.0  # sharp step

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :fold] = [0.2, 0.2, 0.2]
    rgb[:, fold:] = [0.8, 0.8, 0.8]

    depth_jbf = cv2_jbf(rgb, depth)

    # Gradient magnitude at the fold should still be strong
    edge_grad = np.abs(np.diff(depth_jbf[:, fold-2:fold+2], axis=1)).max()
    assert edge_grad > 0.3, f"Edge gradient {edge_grad:.3f} too weak (edge blurred)"
    print(f"  [PASS] Edge gradient at fold: {edge_grad:.3f} > 0.3")
    return True


def test_output_shape_and_nan():
    """Verify output has correct shape and contains no NaN/Inf."""
    from l_splat.core.depth_filter import cv2_jbf

    H, W = 64, 96
    rng = np.random.RandomState(123)
    depth = rng.randn(H, W).astype(np.float32) * 0.1 + 1.0
    rgb = rng.rand(H, W, 3).astype(np.float32)

    depth_jbf = cv2_jbf(rgb, depth)

    assert depth_jbf.shape == (H, W), f"Shape {depth_jbf.shape} != ({H}, {W})"
    assert not np.any(np.isnan(depth_jbf)), "NaN in output"
    assert not np.any(np.isinf(depth_jbf)), "Inf in output"
    print(f"  [PASS] Shape {depth_jbf.shape}, no NaN/Inf")
    return True


def test_ksize_validation():
    """Verify cv2 JBF works with different kernel sizes."""
    from l_splat.core.depth_filter import cv2_jbf

    H, W = 32, 48
    rng = np.random.RandomState(7)
    depth = rng.rand(H, W).astype(np.float32)
    rgb = rng.rand(H, W, 3).astype(np.float32)

    for ksize in [5, 7, 9, 11]:
        result = cv2_jbf(rgb, depth, ksize=ksize)
        assert result.shape == (H, W), f"ksize={ksize}: wrong shape"
        assert not np.any(np.isnan(result)), f"ksize={ksize}: NaN in output"
    print(f"  [PASS] All ksize values (5,7,9,11) produce valid output")
    return True


def test_performance():
    """Verify cv2 JBF latency is within budget for typical frame sizes."""
    from l_splat.core.depth_filter import cv2_jbf

    # Typical post-DAV3 frame: 378x504 (resized) or original ~600x800
    H, W = 378, 504
    rng = np.random.RandomState(99)
    depth = rng.rand(H, W).astype(np.float32)
    rgb = rng.rand(H, W, 3).astype(np.float32)

    # Warm-up
    _ = cv2_jbf(rgb, depth)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = cv2_jbf(rgb, depth)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    budget_ms = 200  # generous: C++ JBF is fast
    assert avg_ms < budget_ms, f"JBF too slow: {avg_ms:.1f}ms >= {budget_ms}ms"
    print(f"  [PASS] JBF latency: {avg_ms:.1f}ms < {budget_ms}ms (avg of 5)")
    return True


def harness():
    print("=" * 64)
    print("  HARNESS: Depth Filter — cv2 JBF")
    print("=" * 64)

    results = []

    print("\n  Phase 1 — Functional Correctness")
    results.append(("Noise Reduction", test_noise_reduction()))
    results.append(("Edge Preservation", test_edge_preservation()))
    results.append(("Output Shape & NaN", test_output_shape_and_nan()))
    results.append(("Kernel Size Variants", test_ksize_validation()))

    print("\n  Phase 2 — Performance")
    results.append(("Performance Budget", test_performance()))

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 64)
    return all_pass


if __name__ == "__main__":
    ok = harness()
    import sys
    sys.exit(0 if ok else 1)
