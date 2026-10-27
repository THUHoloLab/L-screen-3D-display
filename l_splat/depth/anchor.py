"""
harness_temporal_depth_anchor.py
Temporal depth anchoring via EMA-smoothed percentile bounds.

Phase 1 — DepthAnchorStream functional correctness.
Phase 2 — 50-frame synthetic drift validation with assertions.

Problem: Per-frame Min-Max normalization of DAV3 depth causes "breathing jelly
effect" in video — the depth scale drifts frame-to-frame as the scene content
changes, producing visible pulsation in the rendered 3DGS output.

Solution: Lock the scene scale by maintaining EMA-smoothed p5/p95 percentile
bounds.  A very low alpha (0.05) ensures the anchor adapts slowly to genuine
scene changes while suppressing frame-to-frame jitter.
"""

import numpy as np


RAW_DEPTH_NEAR_HIGH = "near_high"
RAW_DEPTH_RAY_DEPTH_HIGH_FAR = "ray_depth_high_far"
VALID_RAW_DEPTH_SEMANTICS = {
    RAW_DEPTH_NEAR_HIGH,
    RAW_DEPTH_RAY_DEPTH_HIGH_FAR,
}


class DepthAnchorStream:
    """
    EMA-smoothed percentile-based depth normalizer.

    Maintains running estimates of the 5th and 95th depth percentiles via
    exponential moving average.  Normalizes each incoming depth frame by
    these smoothed bounds, then maps to a fixed physical range.

    Parameters
    ----------
    physical_near : float
        Physical depth at normalized=1.0 (e.g. +0.08 m).
    physical_far : float
        Physical depth at normalized=0.0 (e.g. -0.12 m).
    alpha : float
        EMA update rate for min/max bounds.  Low values (0.02–0.10) give
        strong temporal smoothing; higher values adapt faster.
    p_lo : float
        Lower percentile for robust min (default 5).
    p_hi : float
        Upper percentile for robust max (default 95).
    raw_depth_semantic : {"near_high", "ray_depth_high_far"}
        Meaning of larger raw input values. Returned depth_norm always keeps
        L-Splat's contract: 0 = far, 1 = near.
    """

    def __init__(self, physical_near=0.08, physical_far=-0.12,
                 alpha=0.05, p_lo=5.0, p_hi=95.0,
                 raw_depth_semantic=RAW_DEPTH_NEAR_HIGH):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= p_lo < p_hi <= 100.0:
            raise ValueError(f"require 0 <= p_lo < p_hi <= 100, got p_lo={p_lo}, p_hi={p_hi}")
        if raw_depth_semantic not in VALID_RAW_DEPTH_SEMANTICS:
            raise ValueError(
                "raw_depth_semantic must be one of "
                f"{sorted(VALID_RAW_DEPTH_SEMANTICS)}, got {raw_depth_semantic!r}")

        self.physical_near = float(physical_near)
        self.physical_far = float(physical_far)
        self.alpha = float(alpha)
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.raw_depth_semantic = raw_depth_semantic

        # EMA state — None until first frame seeds them
        self._ema_min = None
        self._ema_max = None
        self._n_frames = 0

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def ema_min(self):
        """Current EMA-smoothed lower percentile bound (or None before 1st frame)."""
        return self._ema_min

    @property
    def ema_max(self):
        """Current EMA-smoothed upper percentile bound (or None before 1st frame)."""
        return self._ema_max

    @property
    def n_frames(self):
        """Number of frames processed so far."""
        return self._n_frames

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def process(self, depth_raw):
        """
        Normalize a raw depth frame using EMA-smoothed percentile bounds.

        Parameters
        ----------
        depth_raw : np.ndarray
            (H, W) float32 raw depth from DAV3.

        Returns
        -------
        depth_norm : np.ndarray
            (H, W) float32 normalized depth in [0, 1].
            0 = farthest (physical_far), 1 = nearest (physical_near).
        depth_phys : np.ndarray
            (H, W) float32 depth mapped to the physical range.
        """
        valid = depth_raw[np.isfinite(depth_raw)]
        if valid.size == 0:
            raise ValueError("depth_raw has no finite values")

        cur_min = float(np.percentile(valid, self.p_lo))
        cur_max = float(np.percentile(valid, self.p_hi))

        if cur_max - cur_min < 1e-8:
            cur_max = cur_min + 1e-8

        # Seed EMA on first frame
        if self._ema_min is None:
            self._ema_min = cur_min
            self._ema_max = cur_max
        else:
            self._ema_min = self.alpha * cur_min + (1.0 - self.alpha) * self._ema_min
            self._ema_max = self.alpha * cur_max + (1.0 - self.alpha) * self._ema_max

        self._n_frames += 1

        # Normalize by EMA-smoothed bounds
        denom = self._ema_max - self._ema_min
        if denom < 1e-8:
            denom = 1e-8
        depth_norm = (depth_raw - self._ema_min) / denom
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        if self.raw_depth_semantic == RAW_DEPTH_RAY_DEPTH_HIGH_FAR:
            depth_norm = 1.0 - depth_norm

        # Map to physical range
        depth_phys = self.physical_far + depth_norm * (self.physical_near - self.physical_far)

        return depth_norm, depth_phys

    def reset(self):
        """Reset EMA state for re-use on a new sequence."""
        self._ema_min = None
        self._ema_max = None
        self._n_frames = 0


# ============================================================================
# Harness
# ============================================================================

def _generate_drift_sequence(n_frames=50, H=128, W=160, seed=42):
    """
    Generate synthetic depth frames simulating a vehicle slowly approaching.

    Depth values start centered around 0.5 and gradually shift up to ~0.65
    (realistic drift over ~1.7s at 30fps).  Gaussian noise is added per
    frame to simulate real sensor variation.
    """
    rng = np.random.RandomState(seed)
    frames = []
    for i in range(n_frames):
        # Baseline drift: 0.5 → 0.65 over n_frames (realistic scene change)
        baseline = 0.5 + 0.15 * (i / max(n_frames - 1, 1))
        # Add spatial texture (gradient + noise)
        Y, X_ = np.mgrid[0:H, 0:W].astype(np.float32)
        texture = 0.05 * np.sin(X_ * 0.1) * np.cos(Y * 0.08)
        noise = rng.randn(H, W).astype(np.float32) * 0.03
        depth = baseline + texture + noise
        depth = np.clip(depth, 0.0, 1.0)
        frames.append(depth.astype(np.float32))
    return frames


def harness():
    print("=" * 64)
    print("  HARNESS: Temporal Depth Anchor (EMA Percentile Bounds)")
    print("=" * 64)

    # ================================================================
    # Phase 1 — Functional Correctness
    # ================================================================
    print("\n" + "=" * 64)
    print("  Phase 1 — Functional Correctness")
    print("=" * 64)

    anchor = DepthAnchorStream(physical_near=0.08, physical_far=-0.12, alpha=0.05)

    # Single-frame test
    H, W = 128, 160
    rng = np.random.RandomState(42)
    d0 = rng.rand(H, W).astype(np.float32) * 0.5 + 0.25  # range [0.25, 0.75]

    d_norm, d_phys = anchor.process(d0)

    # Assertion 1: output shape
    assert d_norm.shape == (H, W), f"norm shape: {d_norm.shape}"
    assert d_phys.shape == (H, W), f"phys shape: {d_phys.shape}"
    print(f"  [PASS] Output shapes: norm={d_norm.shape}, phys={d_phys.shape}")

    # Assertion 2: first frame seeds EMA from current percentiles
    p5 = float(np.percentile(d0, 5))
    p95 = float(np.percentile(d0, 95))
    assert abs(anchor.ema_min - p5) < 1e-6, f"ema_min {anchor.ema_min} != p5 {p5}"
    assert abs(anchor.ema_max - p95) < 1e-6, f"ema_max {anchor.ema_max} != p95 {p95}"
    print(f"  [PASS] Frame 1 seeds EMA: min={anchor.ema_min:.4f} (p5={p5:.4f}), "
          f"max={anchor.ema_max:.4f} (p95={p95:.4f})")

    # Assertion 3: normalized depth is in [0,1]
    assert d_norm.min() >= -1e-6 and d_norm.max() <= 1.0 + 1e-6
    mid_val = d_norm[H // 2, W // 2]
    print(f"  [PASS] d_norm range: [{d_norm.min():.4f}, {d_norm.max():.4f}], "
          f"center sample={mid_val:.4f}")

    # Assertion 4: physical mapping is correct
    # normalized=0 → physical_far=-0.12, normalized=1 → physical_near=+0.08
    assert d_phys.min() >= -0.13 and d_phys.max() <= 0.09
    print(f"  [PASS] d_phys range: [{d_phys.min():.4f}, {d_phys.max():.4f}] "
          f"(target: [-0.12, +0.08])")

    # Assertion 5: no NaN in output
    assert not np.any(np.isnan(d_norm))
    assert not np.any(np.isnan(d_phys))
    print(f"  [PASS] No NaN in outputs")

    # Assertion 6: second identical frame → EMA barely moves
    d_norm2, d_phys2 = anchor.process(d0)
    expected_min = 0.05 * p5 + 0.95 * p5  # = p5 (unchanged)
    assert abs(anchor.ema_min - p5) < 1e-6, \
        f"EMA min shifted on identical frame: {anchor.ema_min} != {p5}"
    print(f"  [PASS] EMA stable on identical frame: "
          f"min={anchor.ema_min:.4f}, max={anchor.ema_max:.4f}")

    # Assertion 7: n_frames counter
    assert anchor.n_frames == 2
    print(f"  [PASS] n_frames={anchor.n_frames}")

    # ================================================================
    # Phase 2 — 50-Frame Drift Validation
    # ================================================================
    print("\n" + "=" * 64)
    print("  Phase 2 — 50-Frame Drift Validation")
    print("=" * 64)

    anchor2 = DepthAnchorStream(physical_near=0.08, physical_far=-0.12, alpha=0.05)
    frames = _generate_drift_sequence(50)

    history = {"frame": [], "cur_min": [], "cur_max": [],
               "ema_min": [], "ema_max": [], "norm_range": []}

    for i, depth in enumerate(frames):
        valid = depth[np.isfinite(depth)]
        cur_min = float(np.percentile(valid, 5))
        cur_max = float(np.percentile(valid, 95))
        _, d_phys = anchor2.process(depth)

        history["frame"].append(i)
        history["cur_min"].append(cur_min)
        history["cur_max"].append(cur_max)
        history["ema_min"].append(anchor2.ema_min)
        history["ema_max"].append(anchor2.ema_max)
        history["norm_range"].append(float(d_phys.max() - d_phys.min()))

    # Assertion 8: EMA tracks the drift (not frozen)
    ema_min_start = history["ema_min"][0]
    ema_max_start = history["ema_max"][0]
    ema_min_end = history["ema_min"][-1]
    ema_max_end = history["ema_max"][-1]

    assert ema_max_end > ema_max_start, \
        f"EMA max did not track upward drift: {ema_max_start:.4f} → {ema_max_end:.4f}"
    assert ema_min_end > ema_min_start, \
        f"EMA min did not track upward drift: {ema_min_start:.4f} → {ema_min_end:.4f}"
    print(f"  [PASS] EMA tracks drift: "
          f"min {ema_min_start:.4f}→{ema_min_end:.4f}, "
          f"max {ema_max_start:.4f}→{ema_max_end:.4f}")

    # Assertion 9: EMA is smoother than raw percentiles (lower std)
    cur_min_std = float(np.std(history["cur_min"]))
    ema_min_std = float(np.std(history["ema_min"]))
    assert ema_min_std < cur_min_std * 0.8, \
        f"EMA not smoother: ema_std={ema_min_std:.6f} >= cur_std*0.8={cur_min_std*0.8:.6f}"
    print(f"  [PASS] EMA smoother than instant: "
          f"std(ema_min)={ema_min_std:.6f} < std(cur_min)={cur_min_std:.6f}")

    # Assertion 10: first 5 frames → EMA barely moves from seed
    early_ema_range = max(history["ema_max"][:5]) - min(history["ema_max"][:5])
    late_ema_range = max(history["ema_max"][-5:]) - min(history["ema_max"][-5:])
    # In early frames, the true baseline is ~0.5-0.53 so p95 changes are small
    # and EMA with alpha=0.05 responds very slowly
    assert early_ema_range < 0.02, \
        f"EMA max jittered too much in first 5 frames: range={early_ema_range:.4f}"
    print(f"  [PASS] First 5 frames EMA range: {early_ema_range:.5f} (tight)")

    # Assertion 11: physical range stabilizes after EMA convergence
    # During drift, the range narrows as EMA lags; after convergence it
    # approaches 0.20m.  Check that the last 20 frames' range is stable.
    phys_ranges = history["norm_range"]
    late_ranges = phys_ranges[-20:]
    late_range_mean = float(np.mean(late_ranges))
    late_range_std = float(np.std(late_ranges))
    assert late_range_std < 0.005, \
        f"Physical range jitter in steady state: std={late_range_std:.6f}"
    assert 0.18 < late_range_mean <= 0.20 + 0.01, \
        f"Physical range in steady state: mean={late_range_mean:.4f}m"
    print(f"  [PASS] Physical range steady-state: mean={late_range_mean:.4f}m, "
          f"std={late_range_std:.6f}m (near 0.20m)")

    # Assertion 12: no frame-to-frame jumps in physical depth
    phys_medians = [float(np.median(f)) for f in [
        anchor2.process(f)[1] for f in frames[:5]]]  # re-process first 5 for median trend
    # Instead, check consecutive frames have small median jumps
    anchor3 = DepthAnchorStream(physical_near=0.08, physical_far=-0.12, alpha=0.05)
    medians = []
    for f in frames:
        _, d_phys = anchor3.process(f)
        medians.append(float(np.median(d_phys)))
    max_jump = max(abs(medians[i] - medians[i-1]) for i in range(1, len(medians)))
    assert max_jump < 0.02, \
        f"Frame-to-frame median jump too large: {max_jump:.4f}m"
    print(f"  [PASS] Max frame-to-frame median jump: {max_jump:.5f}m (< 0.02m)")

    # Assertion 13: reset clears state
    anchor3.reset()
    assert anchor3.ema_min is None
    assert anchor3.ema_max is None
    assert anchor3.n_frames == 0
    print(f"  [PASS] reset() clears EMA state")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 64)
    print("  ALL TESTS PASSED")
    print("=" * 64)
    print(f"  Frames processed: 50")
    print(f"  Alpha:            0.05")
    print(f"  EMA min drift:    {ema_min_start:.4f} → {ema_min_end:.4f}")
    print(f"  EMA max drift:    {ema_max_start:.4f} → {ema_max_end:.4f}")
    print(f"  Physical range:   {late_range_mean:.4f} ± {late_range_std:.5f} m")
    print(f"  Max median jump:  {max_jump:.5f} m")


if __name__ == "__main__":
    harness()
