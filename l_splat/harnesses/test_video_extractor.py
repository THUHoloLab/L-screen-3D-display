"""
harness_video_extractor.py
Fast Video Frame Extractor — HDD Phase 1 & 2.

Multi-threaded video frame streaming with:
  - Background daemon-thread cv2.VideoCapture reader
  - Frame striding (skip-N sampling)
  - Bounded queue.Queue(maxsize=3) backpressure
  - Async H2D transfer via torch non_blocking pinned memory
  - Integrated performance validation harness (Phase 2)
"""

import cv2
import json
import numpy as np
import torch
import threading
import queue
import time
import os


# ===========================================================================
# Phase 1 — FastVideoExtractor
# ===========================================================================

class FastVideoExtractor:
    """
    Multi-threaded video frame extractor with GPU async transfer.

    A daemon background thread reads frames via cv2.VideoCapture, applies
    frame_stride decimation, and enqueues BGR→RGB uint8 frames into a
    bounded queue.Queue.  The main thread calls get_next_frame_tensor()
    which blocks on the queue, converts to a float32 torch tensor, and
    fires an async H2D transfer via pinned memory + non_blocking copy.

    Parameters
    ----------
    video_path : str
        Path to video file.
    frame_stride : int
        Extract every Nth frame (1 = all frames, 2 = half, etc.).
    dtype : str
        Output tensor dtype: "float32" (default) or "half" (float16).
    """

    def __init__(self, video_path, frame_stride=1, dtype="float32"):
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_stride = max(1, int(frame_stride))
        self._dtype = dtype

        # Expected output frame count (ceil division)
        self.expected_frames = (
            self._total_frames + self._frame_stride - 1
        ) // self._frame_stride

        # Bounded queue — backpressure prevents reader runaway
        self._queue = queue.Queue(maxsize=3)
        self._eof = False
        self._frames_extracted = 0

        # Daemon thread — dies with main process, no explicit join needed
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Background reader (runs in daemon thread)
    # ------------------------------------------------------------------

    def _reader_loop(self):
        """Read frames, apply stride, enqueue RGB uint8.  Push None on EOF."""
        idx = 0
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                if idx % self._frame_stride == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._queue.put(frame_rgb)          # blocks when full → backpressure
                    self._frames_extracted += 1
                idx += 1
        finally:
            self._queue.put(None)   # EOF sentinel
            self._cap.release()

    # ------------------------------------------------------------------
    # Main-thread consumers
    # ------------------------------------------------------------------

    def get_next_frame_numpy(self):
        """
        Dequeue next frame as raw RGB uint8 numpy array.

        Zero-copy fast path for CPU-side consumers (e.g. MiDaS depth).
        The reader thread already does BGR→RGB, so this returns a
        ready-to-use (H, W, 3) uint8 array.

        Returns
        -------
        np.ndarray | None
            (H, W, 3) uint8 RGB frame, or None at EOF.
        """
        if self._eof:
            return None

        frame = self._queue.get()
        if frame is None:
            self._eof = True
            return None
        return frame

    def get_next_frame_tensor(self, device="cuda"):
        """
        Dequeue next frame and transfer to GPU asynchronously.

        Returns
        -------
        torch.Tensor | None
            (H, W, 3) float32 (or half) tensor on `device`, or None at EOF.
        """
        if self._eof:
            return None

        frame = self._queue.get()
        if frame is None:
            self._eof = True
            return None

        # uint8 [0,255] → float32 [0,1], then optional half cast
        arr = frame.astype(np.float32) * (1.0 / 255.0)
        tensor = torch.from_numpy(arr)
        if self._dtype == "half":
            tensor = tensor.half()

        # Pinned → async H2D; pin_memory() allocates page-locked host memory,
        # then .to(device, non_blocking=True) fires the DMA copy on a CUDA
        # stream without blocking the calling thread.
        tensor = tensor.pin_memory().to(device, non_blocking=True)
        return tensor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def frame_stride(self):
        return self._frame_stride

    @property
    def video_fps(self):
        return self._fps

    @property
    def resolution(self):
        return (self._width, self._height)

    @property
    def frames_extracted(self):
        return self._frames_extracted

    def close(self):
        """Release resources.  Safe to call multiple times."""
        self._eof = True
        if self._cap.isOpened():
            self._cap.release()


# ===========================================================================
# Config
# ===========================================================================

VIDEO_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "test.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ===========================================================================
# Phase 2 — Performance Validation Harness
# ===========================================================================

def _save_frame_preview(tensor, path):
    """Save a GPU float32 [0,1] tensor as an 8-bit PNG."""
    img = (tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def harness(video_path=None, output_dir=None):
    if video_path is None:
        video_path = VIDEO_PATH
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  HARNESS: FastVideoExtractor — Phase 1 & 2")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available, testing on CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"  Device: {torch.cuda.get_device_name(0)}")

    # --- Open video ---
    print(f"\n  Video: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap_probe = cv2.VideoCapture(video_path)
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    print(f"  Total frames in video: {total_frames}")

    report = {
        "video_path": video_path,
        "device": device,
        "total_frames_in_file": total_frames,
    }

    try:
        # ==================================================================
        # Phase 1 — Functional Correctness
        # ==================================================================
        print("\n" + "=" * 60)
        print("  Phase 1 — Functional Tests")
        print("=" * 60)

        stride = 2
        extractor = FastVideoExtractor(video_path, frame_stride=stride)

        res_w, res_h = extractor.resolution
        print(f"  Resolution: {res_w}x{res_h}")
        print(f"  Total frames: {extractor.total_frames}")
        print(f"  Video FPS: {extractor.video_fps:.1f}")
        print(f"  Frame stride: {extractor.frame_stride}")
        print(f"  Expected output frames: {extractor.expected_frames}")

        report["resolution"] = f"{res_w}x{res_h}"
        report["video_fps"] = extractor.video_fps
        report["frame_stride"] = stride
        report["expected_output_frames"] = extractor.expected_frames

        # Consume all frames
        frames = []
        while True:
            tensor = extractor.get_next_frame_tensor(device=device)
            if tensor is None:
                break
            frames.append(tensor)

        # --- Assertion 1: Exact frame count with stride=2 ---
        expected = total_frames // stride
        assert len(frames) == expected, \
            f"FRAME COUNT MISMATCH: got {len(frames)}, expected {expected}"
        print(f"\n  [PASS] Frame count: {len(frames)} == {expected} "
              f"(total_frames // stride)")
        report["extracted_frames_stride2"] = len(frames)

        # --- Assertion 2: Tensor shape & dtype ---
        H, W = res_h, res_w
        sample_indices = [0, len(frames) // 2, len(frames) - 1]
        sample_labels = ["first", "middle", "last"]
        for i, idx in enumerate(sample_indices):
            t = frames[idx]
            assert t.shape == (H, W, 3), \
                f"SHAPE MISMATCH: {tuple(t.shape)} != ({H}, {W}, 3)"
            assert t.dtype == torch.float32, \
                f"DTYPE MISMATCH: {t.dtype} != torch.float32"
            print(f"  [PASS] {sample_labels[i]} frame (idx={idx}): shape={tuple(t.shape)}, "
                  f"dtype={t.dtype}, range=[{t.min():.3f}, {t.max():.3f}]")

            # Save preview PNG to output/
            preview_path = os.path.join(
                output_dir, f"frame_stride2_{sample_labels[i]}_idx{idx:04d}.png"
            )
            _save_frame_preview(t, preview_path)

        print(f"  [INFO] Saved {len(sample_indices)} preview frames to {output_dir}/")

        # --- Assertion 3: Tensor on target device ---
        if device == "cuda":
            assert frames[0].is_cuda, "Tensor not on CUDA device"
            print(f"  [PASS] Tensor resides on CUDA device")

        # --- Assertion 4: EOF sentinel works ---
        extra = extractor.get_next_frame_tensor(device=device)
        assert extra is None, "get_next_frame_tensor should return None after EOF"
        print(f"  [PASS] EOF sentinel: returns None after video exhausted")

        extractor.close()

        # ==================================================================
        # Phase 2 — Throughput Benchmark
        # ==================================================================
        print("\n" + "=" * 60)
        print("  Phase 2 — Throughput Benchmark (stride=1, all frames)")
        print("=" * 60)

        # Warm-up: one extraction to initialise CUDA context / allocators
        warm = FastVideoExtractor(video_path, frame_stride=1)
        _ = warm.get_next_frame_tensor(device=device)
        warm.close()

        # Timed run
        extractor2 = FastVideoExtractor(video_path, frame_stride=1)
        torch.cuda.synchronize() if device == "cuda" else None

        t0 = time.perf_counter()
        count = 0
        while True:
            tensor = extractor2.get_next_frame_tensor(device=device)
            if tensor is None:
                break
            count += 1
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        fps = count / elapsed
        ms_per_frame = elapsed / count * 1000

        print(f"  Frames extracted: {count}")
        print(f"  Total elapsed:    {elapsed:.4f} s")
        print(f"  Throughput:       {fps:.1f} FPS  ({ms_per_frame:.2f} ms/frame)")
        print(f"  Video duration:   {count / extractor2.video_fps:.1f}s (real-time)")
        print(f"  Speed multiple:   {fps / extractor2.video_fps:.1f}x real-time")

        report["throughput"] = {
            "frames_extracted": count,
            "elapsed_sec": round(elapsed, 4),
            "fps": round(fps, 1),
            "ms_per_frame": round(ms_per_frame, 2),
            "video_duration_sec": round(count / extractor2.video_fps, 1),
            "speed_multiple_vs_realtime": round(fps / extractor2.video_fps, 1),
            "pixel_throughput_Mpx_per_s": round(fps * res_w * res_h / 1e6, 1),
            "target_Mpx_per_s": 46.08,
        }

        # --- Assertion 5: Pixel throughput >= 46 Mpx/s ---
        # Baseline: 150 FPS @ 640x480 = 46.08 Mpx/s.
        # The target scales with resolution so 720p (1280x720) requires
        # 46.08e6 / (1280*720) = 50 FPS minimum.  The extractor must never
        # be the real-time bottleneck (source is 30 fps).
        px_per_frame = res_w * res_h
        px_throughput = fps * px_per_frame
        target_px_throughput = 150 * 640 * 480  # 46.08 Mpx/s
        fps_min = target_px_throughput / px_per_frame

        assert px_throughput >= target_px_throughput, \
            f"PIXEL THROUGHPUT TOO LOW: {px_throughput/1e6:.1f} Mpx/s " \
            f"< {target_px_throughput/1e6:.1f} Mpx/s target " \
            f"(equivalent to {fps_min:.0f} FPS at {res_w}x{res_h})"
        print(f"  [PASS] Pixel throughput {px_throughput/1e6:.1f} Mpx/s "
              f">= {target_px_throughput/1e6:.1f} Mpx/s target "
              f"(min {fps_min:.0f} FPS @ {res_w}x{res_h})")

        extractor2.close()

        # ==================================================================
        # Summary
        # ==================================================================
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)

        report["status"] = "ALL TESTS PASSED"

    except Exception as e:
        report["status"] = f"FAILED: {e}"
        raise

    finally:
        # Write benchmark report
        report_path = os.path.join(output_dir, "benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Benchmark report: {report_path}")


if __name__ == "__main__":
    harness()
