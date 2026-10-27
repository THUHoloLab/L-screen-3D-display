"""
harness_video_reader_ema.py
Video Depth Streamer with EMA Temporal Smoothing — HDD Phase 1 & 2.

Multi-threaded video reader that:
  1. Reads frames in a background daemon thread (cv2.VideoCapture → Queue).
  2. Computes per-frame depth (via pluggable depth_fn or simulated proxy).
  3. Applies per-pixel Exponential Moving Average (EMA) across time:
         D_smoothed(t) = α · D_curr(t) + (1-α) · D_smoothed(t-1)
     with α = 0.4 to suppress monocular depth temporal jitter.
  4. Returns (rgb_tensor, smoothed_depth_tensor) on the target device.

The EMA filter is inspired by temporally-consistent depth estimation
literature (e.g. CVPR stereo / video-depth works) where a first-order
IIR low-pass in time domain is the simplest effective defence against
frame-to-frame flicker.

Integrated performance validation (Phase 2):
  - Spike suppression: a +3σ depth spike at frame 5 is attenuated >2× by EMA.
  - Throughput: ≥ 100 FPS frame read + depth proxy + EMA.
"""

import cv2
import numpy as np
import torch
import threading
import queue
import time
import os
import tempfile

from l_splat.temporal.depth_fusion import FlowGuidedDepthFusion


# ===========================================================================
# Phase 1 — VideoDepthStreamer
# ===========================================================================

class VideoDepthStreamer:
    """
    Multi-threaded video reader with EMA-smoothed depth output.

    A daemon background thread reads BGR frames via cv2.VideoCapture,
    converts to RGB uint8, and enqueues into a bounded queue (maxsize=3).
    The main thread dequeues, runs an optional depth estimator, and applies
    per-pixel EMA to suppress temporal flicker from monocular depth networks.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    alpha : float
        EMA smoothing factor (0 < alpha ≤ 1).  Smaller = more smoothing.
        alpha=1.0 disables smoothing (raw depth).
    frame_stride : int
        Extract every Nth frame.
    depth_fn : callable or None
        depth_fn(rgb_uint8_np) → depth_float32_np(H,W).
        If None, a simulated proxy (luminance) is used for testing.
    """

    def __init__(self, video_path, alpha=0.4, frame_stride=1, depth_fn=None,
                 flow_estimator=None, flow_history_weight=0.5,
                 flow_fail_open=True):
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0 <= flow_history_weight <= 1):
            raise ValueError(
                "flow_history_weight must be in [0, 1], got "
                f"{flow_history_weight}")

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._alpha = float(alpha)
        self._frame_stride = max(1, int(frame_stride))
        self._depth_fn = depth_fn
        self._flow_estimator = flow_estimator
        self._flow_fail_open = bool(flow_fail_open)
        self._flow_fusion = (
            FlowGuidedDepthFusion(history_weight=flow_history_weight)
            if flow_estimator is not None else None)
        self._prev_rgb = None
        self._flow_failure_reported = False
        self._last_temporal_metrics = {
            "mode": "ema",
            "temporal_support_frames": None,
        }

        self.expected_frames = (
            self._total_frames + self._frame_stride - 1
        ) // self._frame_stride

        # Internal state
        self._queue = queue.Queue(maxsize=3)
        self._prev_depth = None   # (H, W) float32 — EMA accumulator
        self._eof = False
        self._frames_yielded = 0

        # Daemon reader thread
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _reader_loop(self):
        idx = 0
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                if idx % self._frame_stride == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._queue.put((idx, rgb))      # blocks when queue full → backpressure
                idx += 1
        finally:
            self._queue.put(None)   # EOF sentinel
            self._cap.release()

    # ------------------------------------------------------------------
    # Main-thread consumer
    # ------------------------------------------------------------------

    def _apply_legacy_ema(self, depth_curr):
        """Apply the pre-integration recursive EMA path unchanged."""
        if self._prev_depth is None:
            self._prev_depth = depth_curr.copy()
            return depth_curr
        depth_smoothed = (
            self._alpha * depth_curr
            + (1.0 - self._alpha) * self._prev_depth)
        self._prev_depth = depth_smoothed.copy()
        return depth_smoothed

    def _apply_flow_fusion(self, rgb_np, depth_curr):
        """Align the previous raw depth and fuse it in the current domain."""
        flow_device = self._flow_estimator.device
        depth_tensor = torch.from_numpy(
            np.ascontiguousarray(depth_curr)).to(flow_device)
        if self._prev_rgb is None:
            fused = self._flow_fusion.update(depth_tensor)
            metrics = {
                "mode": "flow_warmup",
                **self._flow_fusion.last_stats,
                "flow_total_ms": 0.0,
                "fusion_ms": 0.0,
            }
        else:
            flow_pair = self._flow_estimator.estimate_pair(
                self._prev_rgb, rgb_np, reuse_previous=True)
            if tuple(flow_pair.backward.shape[-2:]) != tuple(depth_tensor.shape):
                raise ValueError(
                    "flow/depth resolution mismatch: "
                    f"{tuple(flow_pair.backward.shape[-2:])} vs "
                    f"{tuple(depth_tensor.shape)}")
            fusion_start = time.perf_counter()
            fused = self._flow_fusion.update(
                depth_tensor,
                backward_flow=flow_pair.backward[0].to(depth_tensor.dtype),
                confidence=flow_pair.confidence[0].to(depth_tensor.dtype),
            )
            if flow_device.type == "cuda":
                torch.cuda.synchronize(flow_device)
            fusion_ms = (time.perf_counter() - fusion_start) * 1000.0
            metrics = {
                "mode": "flow_two_frame",
                **self._flow_fusion.last_stats,
                **flow_pair.timings_ms,
                "flow_total_ms": flow_pair.timings_ms["total_ms"],
                "fusion_ms": fusion_ms,
                "flow_working_height": flow_pair.working_resolution[0],
                "flow_working_width": flow_pair.working_resolution[1],
            }

        depth_smoothed = fused.cpu().numpy().astype(np.float32, copy=False)
        self._prev_rgb = rgb_np.copy()
        self._prev_depth = depth_smoothed.copy()
        self._last_temporal_metrics = metrics
        return depth_smoothed

    def get_smoothed_frame(self, device="cuda"):
        """
        Dequeue next frame, compute depth, apply EMA, return tensors.

        Returns
        -------
        (rgb_tensor, smoothed_depth_tensor) or None at EOF
            rgb_tensor:              (H, W, 3) float32 on `device`, [0, 1].
            smoothed_depth_tensor:   (H, W) float32 on `device`.
        """
        if self._eof:
            return None

        item = self._queue.get()
        if item is None:
            self._eof = True
            return None

        frame_idx, rgb_np = item

        # --- Compute current depth ---
        if self._depth_fn is not None:
            depth_curr = self._depth_fn(rgb_np).astype(np.float32)
        else:
            # Proxy: mean luminance (stands in for real MiDaS during testing)
            depth_curr = rgb_np.astype(np.float32).mean(axis=-1) / 255.0

        # --- Temporal smoothing ---
        if self._flow_estimator is None:
            depth_smoothed = self._apply_legacy_ema(depth_curr)
            self._last_temporal_metrics = {
                "mode": "ema",
                "temporal_support_frames": None,
            }
        else:
            try:
                depth_smoothed = self._apply_flow_fusion(rgb_np, depth_curr)
            except Exception as exc:
                if not self._flow_fail_open:
                    raise
                if not self._flow_failure_reported:
                    print(
                        "  [FlowConfidence] runtime failure; falling back "
                        f"to legacy EMA: {exc}")
                    self._flow_failure_reported = True
                if hasattr(self._flow_estimator, "reset_sequence"):
                    self._flow_estimator.reset_sequence()
                self._flow_estimator = None
                if self._flow_fusion is not None:
                    self._flow_fusion.reset()
                depth_smoothed = self._apply_legacy_ema(depth_curr)
                self._last_temporal_metrics = {
                    "mode": "ema_after_flow_failure",
                    "temporal_support_frames": None,
                    "error": str(exc),
                }

        self._frames_yielded += 1

        # --- To tensors (pinned → async H2D) ---
        rgb_tensor = torch.from_numpy(
            rgb_np.astype(np.float32) * (1.0 / 255.0))
        depth_tensor = torch.from_numpy(depth_smoothed)

        if device != "cpu":
            rgb_tensor = rgb_tensor.pin_memory().to(device, non_blocking=True)
            depth_tensor = depth_tensor.pin_memory().to(device, non_blocking=True)

        return rgb_tensor, depth_tensor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def resolution(self):
        return (self._width, self._height)

    @property
    def video_fps(self):
        return self._fps

    @property
    def frames_yielded(self):
        return self._frames_yielded

    @property
    def alpha(self):
        return self._alpha

    @property
    def prev_depth(self):
        """Last EMA state (may be None before first frame)."""
        return self._prev_depth

    @property
    def flow_enabled(self):
        return self._flow_estimator is not None

    @property
    def temporal_mode(self):
        return self._last_temporal_metrics.get("mode", "ema")

    @property
    def last_temporal_metrics(self):
        return dict(self._last_temporal_metrics)

    def close(self):
        self._eof = True
        if self._flow_fusion is not None:
            self._flow_fusion.reset()
        if (
            self._flow_estimator is not None
            and hasattr(self._flow_estimator, "reset_sequence")
        ):
            self._flow_estimator.reset_sequence()
        if self._cap.isOpened():
            self._cap.release()


# ===========================================================================
# Phase 2 — Performance & Logic Validation Harness
# ===========================================================================

def _generate_depth_test_video(path, num_frames=30, fps=30, size=(320, 240)):
    """
    Generate a synthetic video where each frame has a known luminance
    pattern that maps to a predictable "depth" proxy.

    Frames are numbered blocks so the depth proxy (mean luminance) varies
    in a controlled way, enabling spike-injection verification.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    W, H = size
    rng = np.random.RandomState(42)

    for i in range(num_frames):
        # Base frame: gradient from dark (left) to bright (right)
        base = np.linspace(50, 200, W, dtype=np.uint8)
        frame_gray = np.tile(base, (H, 1))  # (H, W)

        # Add subtle per-frame variation so we can track it
        frame_gray = np.clip(
            frame_gray.astype(np.int32) + int((i - num_frames // 2) * 2), 0, 255
        ).astype(np.uint8)

        # Small random texture
        noise = rng.randint(-8, 9, (H, W), dtype=np.int32)
        frame_gray = np.clip(frame_gray.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)

    writer.release()
    return num_frames


def harness():
    print("=" * 64)
    print("  HARNESS: VideoDepthStreamer + EMA — Phase 1 & 2")
    print("=" * 64)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("  Device: CPU (CUDA not available)")

    # ── Synthetic test video ──
    NUM_FRAMES = 20
    tmp_path = os.path.join(tempfile.gettempdir(),
                            "_harness_video_ema_test.mp4")
    _generate_depth_test_video(tmp_path, num_frames=NUM_FRAMES,
                                fps=30, size=(320, 240))
    print(f"\n  Test video: {tmp_path} ({NUM_FRAMES} frames, 320×240)")

    try:
        # ================================================================
        # Phase 1 — Basic Functionality
        # ================================================================
        print("\n" + "=" * 64)
        print("  Phase 1 — Functional Correctness")
        print("=" * 64)

        alpha = 0.4
        streamer = VideoDepthStreamer(tmp_path, alpha=alpha)
        print(f"  Resolution:  {streamer.resolution[0]}×{streamer.resolution[1]}")
        print(f"  Total frames:{streamer.total_frames}")
        print(f"  Alpha (EMA): {streamer.alpha}")

        # Consume all frames, collect depths for analysis
        raw_depths = []      # per-frame mean of depth_curr (before EMA)
        smoothed_means = []  # per-frame mean of smoothed depth
        frames = []

        idx = 0
        while True:
            result = streamer.get_smoothed_frame(device=device)
            if result is None:
                break
            rgb_t, depth_t = result

            # Collect diagnostics
            depth_cpu = depth_t.cpu().numpy()
            smoothed_means.append(float(depth_cpu.mean()))

            # Also collect the "raw" depth by temporarily computing it
            # (same proxy as inside streamer: mean luminance)
            rgb_cpu = rgb_t.cpu().numpy()
            raw_depth = rgb_cpu.mean(axis=-1)
            raw_depths.append(float(raw_depth.mean()))

            frames.append((rgb_t, depth_t))
            idx += 1

        actual_frames = len(frames)
        print(f"\n  Frames consumed: {actual_frames} / {NUM_FRAMES}")

        # --- Assertion 1: All frames read ---
        assert actual_frames == NUM_FRAMES, \
            f"Frame count mismatch: {actual_frames} != {NUM_FRAMES}"
        print(f"  [PASS] All {NUM_FRAMES} frames read")

        # --- Assertion 2: Tensor shapes ---
        H, W = streamer.resolution[1], streamer.resolution[0]
        assert frames[0][0].shape == (H, W, 3), \
            f"RGB shape mismatch: {frames[0][0].shape}"
        assert frames[0][1].shape == (H, W), \
            f"Depth shape mismatch: {frames[0][1].shape}"
        print(f"  [PASS] Tensor shapes: RGB={tuple(frames[0][0].shape)}, "
              f"Depth={tuple(frames[0][1].shape)}")

        # --- Assertion 3: EMA state accumulates ---
        assert streamer.prev_depth is not None, \
            "prev_depth should be set after first frame"
        print(f"  [PASS] EMA state (prev_depth) initialized & maintained: "
              f"shape={streamer.prev_depth.shape}, "
              f"range=[{streamer.prev_depth.min():.3f}, {streamer.prev_depth.max():.3f}]")

        streamer.close()

        # ================================================================
        # Phase 2 — Spike Suppression & Throughput
        # ================================================================
        print("\n" + "=" * 64)
        print("  Phase 2 — Spike Suppression & Throughput")
        print("=" * 64)

        # --- Spike suppression test ---
        # Create a depth_fn that simulates a stable signal + a spike at frame 5
        ground_depth_value = 0.5  # stable "true" depth
        spike_value = 0.9        # extreme outlier
        spike_frame = 5
        frame_counter = [0]  # mutable counter for closure

        def spiky_depth_fn(rgb_np):
            i = frame_counter[0]
            frame_counter[0] += 1
            base = np.full(rgb_np.shape[:2], ground_depth_value,
                           dtype=np.float32)
            if i == spike_frame:
                # Inject a +3σ spike at frame 5
                base += spike_value - ground_depth_value  # → 0.9
            return base

        streamer2 = VideoDepthStreamer(tmp_path, alpha=0.4,
                                       depth_fn=spiky_depth_fn)

        raw_means_2 = []
        ema_means_2 = []
        while True:
            result = streamer2.get_smoothed_frame(device="cpu")
            if result is None:
                break
            _, depth_t = result
            ema_means_2.append(float(depth_t.mean()))

            # Reconstruct raw depth for comparison
            i = len(ema_means_2) - 1
            if i == spike_frame:
                raw_means_2.append(spike_value)
            else:
                raw_means_2.append(ground_depth_value)

        streamer2.close()

        print(f"\n  EMA smoothing (α={alpha}):")

        # Compute spike suppression ratio
        # Raw jump at spike: |raw[5] - raw[4]|
        # EMA jump at spike: |ema[5] - ema[4]|
        raw_jump = abs(raw_means_2[spike_frame] - raw_means_2[spike_frame - 1])
        ema_jump = abs(ema_means_2[spike_frame] - ema_means_2[spike_frame - 1])
        suppression_ratio = raw_jump / max(ema_jump, 1e-12)

        print(f"    Raw depth at frame {spike_frame - 1}: {raw_means_2[spike_frame - 1]:.4f}")
        print(f"    Raw depth at frame {spike_frame}:     {raw_means_2[spike_frame]:.4f} "
              f"(spike injected)")
        print(f"    Raw jump:    {raw_jump:.4f}")
        print(f"    EMA jump:    {ema_jump:.4f}")
        print(f"    Suppression: {suppression_ratio:.1f}×")

        # Theoretical: EMA for α=0.4 on a step from 0.5→0.9→0.5
        # After spike at t=5, EMA at t=5: D5 = 0.4*0.9 + 0.6*0.5 = 0.36+0.30 = 0.66
        # Jump from prev: |0.66 - 0.50| = 0.16
        # Raw jump: |0.90 - 0.50| = 0.40
        # Suppression ratio: 0.40 / 0.16 = 2.5×
        expected_ema_at_spike = alpha * spike_value + (1 - alpha) * ground_depth_value
        expected_raw_jump = 0.40
        expected_ema_jump = abs(expected_ema_at_spike - ground_depth_value)
        expected_ratio = expected_raw_jump / expected_ema_jump

        print(f"\n    Theoretical:")
        print(f"      EMA at spike frame: D = {alpha}×{spike_value} + "
              f"{1-alpha}×{ground_depth_value} = {expected_ema_at_spike:.4f}")
        print(f"      Expected suppression: {expected_ratio:.1f}×")

        # --- Assertion 4: EMA significantly suppresses the spike (> 2×) ---
        assert suppression_ratio > 2.0, \
            f"EMA SUPPRESSION TOO WEAK: {suppression_ratio:.1f}× ≤ 2.0×"
        print(f"\n  [PASS] EMA suppresses depth spike by {suppression_ratio:.1f}× "
              f"(> 2.0× threshold)")

        # --- Assertion 5: EMA returns toward baseline after spike passes ---
        # Frame 6 (1 frame after spike): D6 = 0.4*0.5 + 0.6*0.66 = 0.20+0.396 = 0.596
        # Should be lower than at spike frame (0.66)
        post_spike_ema = ema_means_2[spike_frame + 1]
        ema_at_spike = ema_means_2[spike_frame]
        assert post_spike_ema < ema_at_spike, \
            f"EMA should decay after spike: post={post_spike_ema:.4f} >= spike={ema_at_spike:.4f}"
        print(f"  [PASS] EMA decays after spike: "
              f"{ema_at_spike:.4f} → {post_spike_ema:.4f}")

        # --- Assertion 6: Steady-state convergence ---
        # After many frames at constant depth, EMA should converge to ground truth
        # D_n = α * D_gt + (1-α) * D_{n-1}
        # As n→∞: D_n → D_gt
        late_ema = ema_means_2[-1]
        convergence_err = abs(late_ema - ground_depth_value)
        # For 20 frames after spike (from frame 5→20 = 15 recovery frames):
        # D_20 = 0.5 * (1 - 0.6^15) ≈ 0.5 * 0.9995 ≈ 0.4998
        # Error < 0.01 should hold easily
        assert convergence_err < 0.02, \
            f"EMA failed to converge: error {convergence_err:.4f} at final frame"
        print(f"  [PASS] EMA converges to baseline: "
              f"final={late_ema:.4f}, err={convergence_err:.4f}")

        # ================================================================
        # Throughput Benchmark
        # ================================================================
        print("\n" + "=" * 64)
        print("  Throughput Benchmark")
        print("=" * 64)

        streamer3 = VideoDepthStreamer(tmp_path, alpha=0.4)
        torch.cuda.synchronize() if device == "cuda" else None

        t0 = time.perf_counter()
        count = 0
        while True:
            result = streamer3.get_smoothed_frame(device=device)
            if result is None:
                break
            count += 1
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        fps = count / elapsed
        ms_per_frame = elapsed / count * 1000
        print(f"  Frames:     {count}")
        print(f"  Elapsed:    {elapsed:.4f} s")
        print(f"  Throughput: {fps:.1f} FPS ({ms_per_frame:.2f} ms/frame)")

        # --- Assertion 7: Minimum throughput (read + depth proxy + EMA) ---
        assert fps >= 100.0, \
            f"THROUGHPUT TOO LOW: {fps:.1f} FPS < 100 FPS target"
        print(f"  [PASS] Throughput {fps:.1f} FPS ≥ 100 FPS target "
              f"(headroom: {fps - 100:.0f} FPS)")

        streamer3.close()

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 64)
        print("  ALL TESTS PASSED")
        print("=" * 64)
        print(f"  EMA α={alpha}: {suppression_ratio:.1f}× spike suppression, "
              f"converges to baseline within {convergence_err:.4f}")
        print(f"  Throughput:   {fps:.1f} FPS (read + depth proxy + EMA)")

    finally:
        # Windows may still hold a file handle; retry with a short delay
        if os.path.exists(tmp_path):
            for _ in range(5):
                try:
                    os.remove(tmp_path)
                    print(f"\n  Cleaned up: {tmp_path}")
                    break
                except PermissionError:
                    time.sleep(0.05)


if __name__ == "__main__":
    harness()
