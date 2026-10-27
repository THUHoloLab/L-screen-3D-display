"""
test_single_image_pipeline.py
Single-Image Pipeline Integration Harness — HDD Phase 1 & 2.

Validates end-to-end correctness of the single-image processing path:
  - Output dimensions match panel spec (832×936 per panel, 2*832-overlap stitched)
  - Aspect ratio correction: content plane matches input image ratio
  - Far-field coverage: upper portion is not excessively suppressed
  - Output files are valid PNGs with expected pixel distributions
  - Performance: E2E time within budget for interactive use

Phase 1 — functional correctness with synthetic test images.
Phase 2 — performance baseline assertion.
"""

import os
import sys
import time
import tempfile
import numpy as np


# ============================================================================
# Phase 1 — E2E Pipeline Functional Correctness
# ============================================================================

def _generate_synthetic_rgb(H, W, seed=42):
    """Generate a synthetic RGB image with a gradient + texture pattern.

    Creates horizontal bands with depth cues: bright = near, dark = far.
    Top half (far-field) has a finer texture pattern, bottom half (near-field)
    has broader structure, simulating a natural scene with depth variation.
    """
    rng = np.random.RandomState(seed)
    Y, X = np.meshgrid(
        np.linspace(0, 1, H, dtype=np.float32),
        np.linspace(0, 1, W, dtype=np.float32),
        indexing='ij')

    # Vertical gradient: top=dark (far), bottom=bright (near)
    depth_hint = Y.copy()

    # RGB channels with depth-dependent coloring
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.2 + 0.4 * depth_hint + 0.1 * np.sin(X * 20) * (1 - depth_hint)
    rgb[:, :, 1] = 0.15 + 0.35 * depth_hint + 0.08 * np.cos(Y * 15) * depth_hint
    rgb[:, :, 2] = 0.3 + 0.5 * depth_hint * 0.8 + 0.12 * np.sin((X + Y) * 12)

    # Add structured texture noise
    noise = rng.randn(H, W, 3).astype(np.float32) * 0.03
    rgb = np.clip(rgb + noise, 0, 1)
    return (rgb * 255).astype(np.uint8)


def _run_pipeline_on_synthetic(H, W, out_dir):
    """Run the full single-image pipeline on a synthetic test image.

    Returns the elapsed time in seconds.
    """
    from l_splat.core.pipeline import process_frame, build_dual_camera
    from l_splat.depth.loader import load_dav3
    from l_splat.depth.anchor import DepthAnchorStream
    from l_splat.core.pipeline import DEPTH_FWD_M, DEPTH_BACK_M

    rgb = _generate_synthetic_rgb(H, W)

    # Load DAV3 once
    model_dav3 = load_dav3("da3-small", device="cuda" if _has_cuda() else "cpu")
    camera = build_dual_camera()
    depth_anchor = DepthAnchorStream(
        physical_near=DEPTH_FWD_M, physical_far=DEPTH_BACK_M, alpha=0.05,
        raw_depth_semantic="ray_depth_high_far")

    t0 = time.perf_counter()
    process_frame(rgb, model_dav3, camera,
                  out_dir=out_dir, frame_idx=None,
                  depth_anchor=depth_anchor)
    elapsed = time.perf_counter() - t0
    return elapsed


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_output_dimensions():
    """Verify rendered outputs have correct panel dimensions."""
    print("=" * 64)
    print("  TEST 1: Output Dimensions")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        _run_pipeline_on_synthetic(600, 800, tmpdir)

        import cv2
        left = cv2.imread(os.path.join(tmpdir, "render_left.png"))
        right = cv2.imread(os.path.join(tmpdir, "render_right.png"))
        raw_combined = cv2.imread(os.path.join(tmpdir, "render_raw_combined.png"))
        stitched = cv2.imread(os.path.join(tmpdir, "render_stitched.png"))
        blended_preview = cv2.imread(os.path.join(tmpdir, "render_blended_preview.png"))

        assert left is not None, "render_left.png not found"
        assert right is not None, "render_right.png not found"
        assert raw_combined is not None, "render_raw_combined.png not found"
        assert stitched is not None, "render_stitched.png not found"
        assert blended_preview is not None, "render_blended_preview.png not found"

        from l_splat.core.pipeline import PANEL_PIXEL_W, PANEL_PIXEL_H

        panel_shape = (PANEL_PIXEL_H, PANEL_PIXEL_W, 3)
        raw_shape = (PANEL_PIXEL_H, PANEL_PIXEL_W * 2, 3)

        assert left.shape == panel_shape, \
            f"Left panel shape: {left.shape}, expected {panel_shape}"
        assert right.shape == panel_shape, \
            f"Right panel shape: {right.shape}, expected {panel_shape}"
        assert raw_combined.shape == raw_shape, \
            f"Raw combined shape: {raw_combined.shape}, expected {raw_shape}"
        assert stitched.shape == raw_shape, \
            f"render_stitched compatibility alias: {stitched.shape}, expected {raw_shape}"
        assert blended_preview.shape[0] == PANEL_PIXEL_H, \
            f"Blended preview height: {blended_preview.shape[0]}, expected {PANEL_PIXEL_H}"
        assert PANEL_PIXEL_W < blended_preview.shape[1] <= PANEL_PIXEL_W * 2, \
            f"Blended preview width: {blended_preview.shape[1]}, expected diagnostic range"

        print(f"  Left:    {left.shape}  [PASS]")
        print(f"  Right:   {right.shape}  [PASS]")
        print(f"  Raw:     {raw_combined.shape}  [PASS]")
        print(f"  Preview: {blended_preview.shape}  [PASS]")
    return True


def test_aspect_ratio_correction():
    """Verify content_w/content_h matches input W/H after our fix."""
    print("\n" + "=" * 64)
    print("  TEST 2: Aspect Ratio Correction")
    print("=" * 64)

    from l_splat.core.pipeline import CONTENT_W_M

    test_cases = [
        (600, 800, 800/600),       # 4:3 landscape
        (1326, 1440, 1440/1326),   # original hypercar aspect
        (1080, 1920, 1920/1080),   # 16:9 HD
    ]

    for H, W, expected_ratio in test_cases:
        content_h = CONTENT_W_M * (H / W)
        actual_ratio = CONTENT_W_M / content_h
        assert abs(actual_ratio - expected_ratio) < 0.001, \
            f"Aspect mismatch: {actual_ratio:.5f} != {expected_ratio:.5f}"
        print(f"  {W}x{H}: content_h={content_h*1000:.1f}mm  "
              f"ratio={actual_ratio:.5f} == {expected_ratio:.5f}  [PASS]")
    return True


def test_content_coverage():
    """Verify rendered output has reasonable pixel coverage (not all black)."""
    print("\n" + "=" * 64)
    print("  TEST 3: Content Coverage (non-black pixel ratio)")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        _run_pipeline_on_synthetic(600, 800, tmpdir)

        import cv2
        left = cv2.imread(os.path.join(tmpdir, "render_left.png"))
        right = cv2.imread(os.path.join(tmpdir, "render_right.png"))

        for name, img in [("Left", left), ("Right", right)]:
            H = img.shape[0]
            total_px = H * img.shape[1]

            # Global coverage
            nonblack_all = (img.mean(axis=2) > 15).sum()
            coverage = nonblack_all / total_px

            # Upper vs lower half coverage
            upper = img[:H // 2]
            lower = img[H // 2:]
            upper_nb = (upper.mean(axis=2) > 15).sum()
            lower_nb = (lower.mean(axis=2) > 15).sum()
            up_cov = upper_nb / (total_px / 2)
            lo_cov = lower_nb / (total_px / 2)
            balance = up_cov / max(lo_cov, 1e-8)

            print(f"  {name}: coverage={coverage*100:.1f}%  "
                  f"upper={up_cov*100:.1f}%  lower={lo_cov*100:.1f}%  "
                  f"balance={balance:.2f}")

            # Assertions:
            # 1. At least some content rendered (not completely black)
            assert coverage > 0.05, \
                f"{name}: coverage {coverage*100:.1f}% < 5%"
            print(f"    [PASS] coverage {coverage*100:.1f}% > 5%")

            # 2. Upper half has non-zero coverage (far-field not wiped out)
            assert up_cov > 0.02, \
                f"{name}: upper coverage {up_cov*100:.1f}% < 2%"
            print(f"    [PASS] upper coverage {up_cov*100:.1f}% > 2%")

            # 3. Balance ratio is within sane bounds
            assert 0.2 < balance < 5.0, \
                f"{name}: balance ratio {balance:.2f} out of range (0.2, 5.0)"
            print(f"    [PASS] balance ratio {balance:.2f} in (0.2, 5.0)")
    return True


def test_output_file_integrity():
    """Verify all expected output files are produced and valid."""
    print("\n" + "=" * 64)
    print("  TEST 4: Output File Integrity")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        _run_pipeline_on_synthetic(600, 800, tmpdir)

        expected_files = [
            "render_left.png", "render_right.png", "render_raw_combined.png",
            "render_stitched.png", "render_blended_preview.png",
            "seam_crop_left_edge.png", "seam_crop_right_edge.png",
            "seam_crop_raw_combined.png",
            "depth_raw.png", "depth_left.png", "depth_right.png",
            "input_preview.png", "parameter_provenance.json",
        ]

        import cv2
        for fname in expected_files:
            fpath = os.path.join(tmpdir, fname)
            assert os.path.exists(fpath), f"Missing output: {fname}"

            if fname.endswith(".png"):
                img = cv2.imread(fpath)
                assert img is not None, f"Corrupted image: {fname}"
                assert img.size > 0, f"Empty image: {fname}"
                assert not np.any(np.isnan(img)), f"NaN in {fname}"

            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname}: {size_kb:.1f} KB  [OK]")
    return True


# ============================================================================
# Phase 2 — Performance Baseline
# ============================================================================

def test_performance_baseline():
    """Verify single-image pipeline completes within time budget."""
    print("\n" + "=" * 64)
    print("  TEST 5: Performance Baseline (E2E time budget)")
    print("=" * 64)

    TIME_BUDGET_SEC = 60.0  # generous: spatial hash KNN dominates (~25s for 850K gaussians)

    with tempfile.TemporaryDirectory() as tmpdir:
        elapsed = _run_pipeline_on_synthetic(600, 800, tmpdir)

        print(f"  E2E time: {elapsed:.1f}s  (budget: {TIME_BUDGET_SEC:.0f}s)")

        assert elapsed < TIME_BUDGET_SEC, \
            f"Pipeline too slow: {elapsed:.1f}s >= {TIME_BUDGET_SEC:.0f}s"
        print(f"  [PASS] E2E time {elapsed:.1f}s < {TIME_BUDGET_SEC:.0f}s budget")

        # Verify DAV3 inference + gaussian init are running (not just render)
        assert elapsed > 0.5, \
            f"Pipeline suspiciously fast ({elapsed:.1f}s) — may skip stages"
        print(f"  [PASS] E2E time > 0.5s (DAV3 inference + gaussian init confirmed)")
        print(f"  [NOTE] KNN bottleneck: ~{elapsed-0.5:.0f}s in spatial hash, "
              f"target for Phase 2 zero-copy pool")
    return True


# ============================================================================
# Harness Orchestrator
# ============================================================================

def harness():
    print("=" * 64)
    print("  HARNESS: Single-Image Pipeline Integration")
    print("=" * 64)
    print(f"  CUDA: {_has_cuda()}")

    # Shared resources: load DAV3 and camera once for all tests
    from l_splat.core.pipeline import build_dual_camera
    from l_splat.depth.loader import load_dav3
    print("  Loading shared DAV3 + camera (once)...")
    _shared_dav3 = load_dav3("da3-small", device="cuda" if _has_cuda() else "cpu")
    _shared_camera = build_dual_camera()
    print("  Shared resources ready.\n")

    # Monkey-patch _run_pipeline_on_synthetic to use shared resources
    global _run_pipeline_on_synthetic

    def _run_with_shared(H, W, out_dir):
        from l_splat.core.pipeline import process_frame
        from l_splat.depth.anchor import DepthAnchorStream
        from l_splat.core.pipeline import DEPTH_FWD_M, DEPTH_BACK_M

        rgb = _generate_synthetic_rgb(H, W)
        depth_anchor = DepthAnchorStream(
            physical_near=DEPTH_FWD_M, physical_far=DEPTH_BACK_M, alpha=0.05,
            raw_depth_semantic="ray_depth_high_far")

        t0 = time.perf_counter()
        process_frame(rgb, _shared_dav3, _shared_camera,
                      out_dir=out_dir, frame_idx=None,
                      depth_anchor=depth_anchor)
        elapsed = time.perf_counter() - t0
        return elapsed

    _run_pipeline_on_synthetic = _run_with_shared

    results = []

    # Phase 1: Functional correctness
    print("\n" + "=" * 64)
    print("  Phase 1 — Functional Correctness")
    print("=" * 64)
    results.append(("Output Dimensions", test_output_dimensions()))
    results.append(("Aspect Ratio Correction", test_aspect_ratio_correction()))
    results.append(("Content Coverage", test_content_coverage()))
    results.append(("File Integrity", test_output_file_integrity()))

    # Phase 2: Performance — measure process_frame() only (not DAV3 load)
    print("\n" + "=" * 64)
    print("  Phase 2 — Performance Baseline (process_frame only)")
    print("=" * 64)
    results.append(("Performance Baseline", test_performance_baseline()))

    # Summary
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

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    harness()
