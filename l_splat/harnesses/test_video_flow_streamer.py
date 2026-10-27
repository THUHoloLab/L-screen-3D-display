"""Integration tests for VideoDepthStreamer + two-frame flow fusion."""

from __future__ import annotations

from types import SimpleNamespace
import os
import tempfile
import time

import cv2
import numpy as np
import torch

from l_splat.depth.streamer import VideoDepthStreamer


class _ZeroFlowEstimator:
    device = torch.device("cpu")

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def estimate_pair(self, previous_rgb, current_rgb, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic flow failure")
        height, width = previous_rgb.shape[:2]
        return SimpleNamespace(
            backward=torch.zeros((1, 2, height, width), dtype=torch.float32),
            confidence=torch.ones((1, height, width), dtype=torch.float32),
            working_resolution=(height, width),
            timings_ms={
                "preprocess_ms": 0.0,
                "raft_bidirectional_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_ms": 0.0,
            },
        )


def _make_video(path, frames=3, size=(16, 16)):
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size
    )
    for value in range(frames):
        writer.write(np.full((size[1], size[0], 3), value * 40, dtype=np.uint8))
    writer.release()


def _depth_sequence(values):
    state = {"index": 0}

    def depth_fn(rgb):
        index = state["index"]
        state["index"] += 1
        return np.full(rgb.shape[:2], values[index], dtype=np.float32)

    return depth_fn


def _consume(streamer):
    outputs = []
    while True:
        result = streamer.get_smoothed_frame(device="cpu")
        if result is None:
            break
        outputs.append(result[1].numpy().copy())
    streamer.close()
    return outputs


def test_streamer_uses_exactly_one_previous_raw_depth():
    video_path = os.path.join(
        tempfile.gettempdir(), "_l_splat_flow_streamer_contract.mp4"
    )
    _make_video(video_path)
    estimator = _ZeroFlowEstimator()
    streamer = VideoDepthStreamer(
        video_path,
        alpha=0.4,
        depth_fn=_depth_sequence([0.0, 1.0, 3.0]),
        flow_estimator=estimator,
        flow_history_weight=0.5,
        flow_fail_open=False,
    )
    try:
        outputs = _consume(streamer)
        assert len(outputs) == 3
        np.testing.assert_array_equal(outputs[0], 0.0)
        np.testing.assert_allclose(outputs[1], 0.5, rtol=0.0, atol=1e-6)
        # Uses previous raw value 1.0, not previous fused value 0.5.
        np.testing.assert_allclose(outputs[2], 2.0, rtol=0.0, atol=1e-6)
        assert estimator.calls == 2
        assert streamer.temporal_mode == "flow_two_frame"
        assert streamer.last_temporal_metrics["temporal_support_frames"] == 2.0
    finally:
        if os.path.exists(video_path):
            for _ in range(5):
                try:
                    os.remove(video_path)
                    break
                except PermissionError:
                    time.sleep(0.05)


def test_fail_open_returns_to_legacy_ema():
    video_path = os.path.join(
        tempfile.gettempdir(), "_l_splat_flow_streamer_failopen.mp4"
    )
    _make_video(video_path, frames=2)
    estimator = _ZeroFlowEstimator(fail=True)
    streamer = VideoDepthStreamer(
        video_path,
        alpha=0.4,
        depth_fn=_depth_sequence([0.0, 1.0]),
        flow_estimator=estimator,
        flow_history_weight=0.5,
        flow_fail_open=True,
    )
    try:
        outputs = _consume(streamer)
        assert len(outputs) == 2
        np.testing.assert_array_equal(outputs[0], 0.0)
        np.testing.assert_allclose(outputs[1], 0.4, rtol=0.0, atol=1e-6)
        assert not streamer.flow_enabled
        assert streamer.temporal_mode == "ema_after_flow_failure"
    finally:
        if os.path.exists(video_path):
            for _ in range(5):
                try:
                    os.remove(video_path)
                    break
                except PermissionError:
                    time.sleep(0.05)


def test_flow_off_preserves_legacy_ema_values():
    video_path = os.path.join(
        tempfile.gettempdir(), "_l_splat_flow_streamer_legacy.mp4"
    )
    _make_video(video_path)
    streamer = VideoDepthStreamer(
        video_path,
        alpha=0.4,
        depth_fn=_depth_sequence([0.0, 1.0, 3.0]),
        flow_estimator=None,
    )
    try:
        outputs = _consume(streamer)
        assert len(outputs) == 3
        np.testing.assert_array_equal(outputs[0], np.float32(0.0))
        np.testing.assert_array_equal(outputs[1], np.float32(0.4))
        np.testing.assert_array_equal(outputs[2], np.float32(1.44))
        assert not streamer.flow_enabled
        assert streamer.temporal_mode == "ema"
    finally:
        if os.path.exists(video_path):
            for _ in range(5):
                try:
                    os.remove(video_path)
                    break
                except PermissionError:
                    time.sleep(0.05)


def main():
    tests = [
        test_streamer_uses_exactly_one_previous_raw_depth,
        test_fail_open_returns_to_legacy_ema,
        test_flow_off_preserves_legacy_ema_values,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)} VIDEO FLOW STREAMER TESTS PASSED")


if __name__ == "__main__":
    main()
