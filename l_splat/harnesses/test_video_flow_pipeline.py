"""
test_video_flow_pipeline.py
Isolation harness: VideoIO + FlowConfidenceEstimator end-to-end.

Validates:
  1. VideoIO.extract_frames() — extract first 2 frames from test video
  2. FlowConfidenceEstimator.compute_mask() — RAFT forward-backward consistency
  3. Mask visualisation — save as grayscale heatmap PNG
  4. VideoIO.compile_video() — re-assemble extracted frames into .mp4

Usage:
    python tests/test_video_flow_pipeline.py [--video PATH]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Ensure l_splat is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from l_splat.video_io import VideoIO
from l_splat.flow_confidence import FlowConfidenceEstimator
from l_splat.depth.streamer import VideoDepthStreamer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "flow_test")


def _save_colormap(mask: np.ndarray, path: str):
    """Save (H,W) float32 mask as an 8-bit Inferno heatmap PNG."""
    # mask in [0, 1]
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, heatmap)
    print(f"  [SAVED] {path}  ({mask.shape[1]}x{mask.shape[0]} px)")


def _save_side_by_side(frame_t, frame_tp1, mask, path, label: str):
    """Concatenate [frame_t | frame_t+1 | mask_heatmap] horizontally."""
    H, W = frame_t.shape[:2]
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    mask_rgb = cv2.applyColorMap(mask_u8, cv2.COLORMAP_INFERNO)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

    # Convert frames back to RGB for display
    f_t = frame_t.copy()
    f_tp1 = frame_tp1.copy()

    # Ensure same height
    h_max = max(f_t.shape[0], f_tp1.shape[0], mask_rgb.shape[0])
    f_t = cv2.copyMakeBorder(f_t, 0, h_max - f_t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    f_tp1 = cv2.copyMakeBorder(f_tp1, 0, h_max - f_tp1.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    mask_rgb = cv2.copyMakeBorder(mask_rgb, 0, h_max - mask_rgb.shape[0], 0, 0, cv2.BORDER_CONSTANT)

    canvas = np.hstack([f_t, f_tp1, mask_rgb])
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  [SAVED] {path} ({label})")


# ===========================================================================
# Main Harness
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="VideoIO + FlowConfidence test harness")
    ap.add_argument("--video", type=str,
                    default=os.path.join(os.path.expanduser("~"), "Desktop", "test.mp4"),
                    help="Path to test video (default: ~/Desktop/test.mp4)")
    ap.add_argument("--max-frames", type=int, default=2,
                    help="Frames to extract for flow test (default: 2)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Frame stride for extraction (default: 1)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 64)
    print("  TEST HARNESS: VideoIO + FlowConfidenceEstimator")
    print("=" * 64)

    # ======================================================================
    # Step 1 — VideoIO: Extract frames
    # ======================================================================
    print("\n── Step 1: VideoIO.extract_frames() ──")

    vio = VideoIO(frame_stride=args.stride)

    video_path = os.path.expanduser(args.video) if args.video.startswith("~") \
                 else args.video
    if not os.path.exists(video_path):
        print(f"  ERROR: Video not found: {video_path}")
        print(f"  Provide a .mp4 via: python {__file__} --video <path>")
        sys.exit(1)

    info = VideoIO.probe(video_path)
    print(f"  Video: {video_path}")
    print(f"  Metadata: {info.get('total_frames')} frames, "
          f"{info.get('fps'):.1f} fps, "
          f"{info.get('width')}x{info.get('height')} px")

    t0 = time.perf_counter()
    frames = vio.extract_frames(video_path, max_frames=args.max_frames)
    t_extract = time.perf_counter() - t0
    print(f"  Extracted {len(frames)} frames in {t_extract:.3f}s "
          f"({t_extract/len(frames)*1000:.1f} ms/frame)" if frames else "")

    if len(frames) < 2:
        print(f"  ERROR: Need at least 2 frames for flow, got {len(frames)}")
        sys.exit(1)

    frame_t, frame_tp1 = frames[0], frames[1]
    print(f"  Frame shapes: t={frame_t.shape}, t+1={frame_tp1.shape}")

    # Save extracted frames
    cv2.imwrite(os.path.join(OUT_DIR, "frame_0000.png"),
                cv2.cvtColor(frame_t, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT_DIR, "frame_0001.png"),
                cv2.cvtColor(frame_tp1, cv2.COLOR_RGB2BGR))
    print(f"  [SAVED] frame_0000.png, frame_0001.png")

    # ======================================================================
    # Step 2 — FlowConfidenceEstimator: Compute mask
    # ======================================================================
    print("\n── Step 2: FlowConfidenceEstimator.compute_mask() ──")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    estimator = FlowConfidenceEstimator(model_size="small", device=device,
                                        fb_threshold=0.5)

    t0 = time.perf_counter()
    mask = estimator.compute_mask(frame_t, frame_tp1)
    t_flow = time.perf_counter() - t0

    print(f"  Mask shape: {mask.shape}  dtype: {mask.dtype}")
    print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"  Mean confidence (mask>0.5):  {(mask > 0.5).mean()*100:.1f}%")
    print(f"  Mean uncertainty (mask<0.5): {(mask < 0.5).mean()*100:.1f}%")
    print(f"  RAFT inference + FB consistency: {t_flow:.3f}s")

    # ======================================================================
    # Step 2b — Production video-streamer integration
    # ======================================================================
    print("\n── Step 2b: VideoDepthStreamer + two-frame fusion ──")

    def luminance_depth(rgb_np):
        return rgb_np.astype(np.float32).mean(axis=-1) / 255.0

    streamer = VideoDepthStreamer(
        video_path,
        alpha=0.4,
        frame_stride=args.stride,
        depth_fn=luminance_depth,
        flow_estimator=estimator,
        flow_history_weight=0.5,
        flow_fail_open=False,
    )
    first_result = streamer.get_smoothed_frame(device="cpu")
    second_result = streamer.get_smoothed_frame(device="cpu")
    temporal_metrics = streamer.last_temporal_metrics
    streamer.close()

    assert first_result is not None and second_result is not None
    assert temporal_metrics["mode"] == "flow_two_frame"
    assert temporal_metrics["temporal_support_frames"] == 2.0
    assert 0.0 <= temporal_metrics["mean_confidence"] <= 1.0
    print(
        "  [PASS] production streamer mode=flow_two_frame, "
        f"support={temporal_metrics['temporal_support_frames']:.0f} frames, "
        f"flow={temporal_metrics['flow_total_ms']:.1f} ms, "
        f"fusion={temporal_metrics['fusion_ms']:.1f} ms, "
        f"confidence={temporal_metrics['mean_confidence']:.3f}"
    )

    # ======================================================================
    # Step 3 — Visualise mask
    # ======================================================================
    print("\n── Step 3: Visualise mask ──")

    _save_colormap(mask, os.path.join(OUT_DIR, "confidence_mask.png"))
    _save_side_by_side(frame_t, frame_tp1, mask,
                       os.path.join(OUT_DIR, "comparison_triptych.png"),
                       "frame_t | frame_t+1 | confidence_mask")

    # ======================================================================
    # Step 4 — VideoIO: Compile back to video
    # ======================================================================
    print("\n── Step 4: VideoIO.compile_video() ──")

    all_frames = vio.extract_frames(video_path, max_frames=-1)
    print(f"  Re-extracted {len(all_frames)} frames for video compilation")

    output_video = os.path.join(OUT_DIR, "recompiled.mp4")
    original_fps = info.get("fps", 30.0) / args.stride
    t0 = time.perf_counter()
    VideoIO.compile_video(all_frames, output_video, fps=original_fps)
    t_compile = time.perf_counter() - t0
    print(f"  Compile time: {t_compile:.3f}s")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 64)
    print("  HARNESS COMPLETE — All steps passed")
    print("=" * 64)
    print(f"  Extract time:      {t_extract*1000:.0f} ms")
    print(f"  RAFT+FB time:      {t_flow*1000:.0f} ms")
    print(f"  Compile time:      {t_compile*1000:.0f} ms")
    print(f"  Output dir:        {OUT_DIR}")
    print(f"  Confidence mask:   confidence_mask.png")
    print(f"  Triptych:          comparison_triptych.png")
    print(f"  Recompiled video:  recompiled.mp4")


if __name__ == "__main__":
    main()
