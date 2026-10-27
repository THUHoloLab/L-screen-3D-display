"""
Contract tests for the proposed flow-confidence-guided depth fusion stage.

These tests intentionally avoid RAFT weights and exercise only deterministic
fusion mathematics on CPU.  Before the production module is implemented, the
import below is expected to fail; that failure is the pre-change HDD evidence.

Proposed tensor conventions
---------------------------
- depth:          (H, W) float tensor
- backward_flow:  (2, H, W) float tensor mapping current pixels to previous
- confidence:     (H, W) float tensor, clamped to [0, 1]
- history_weight: scalar in [0, 1]
"""

from __future__ import annotations

import torch

from l_splat.temporal.depth_fusion import (
    FlowGuidedDepthFusion,
    fuse_depth_with_confidence,
)


def _zero_flow(height: int, width: int) -> torch.Tensor:
    return torch.zeros((2, height, width), dtype=torch.float32)


def test_static_high_confidence_blend() -> None:
    current = torch.full((3, 4), 1.0, dtype=torch.float32)
    previous = torch.full((3, 4), 0.2, dtype=torch.float32)
    confidence = torch.ones_like(current)

    fused = fuse_depth_with_confidence(
        current,
        previous,
        _zero_flow(3, 4),
        confidence,
        history_weight=0.25,
    )

    expected = torch.full_like(current, 0.8)
    torch.testing.assert_close(fused, expected, rtol=0.0, atol=1e-6)


def test_low_confidence_and_warp_holes_fall_back_to_current() -> None:
    current = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    previous = torch.full_like(current, -50.0)

    low_confidence = torch.zeros_like(current)
    fused_low = fuse_depth_with_confidence(
        current,
        previous,
        _zero_flow(3, 4),
        low_confidence,
        history_weight=1.0,
    )
    torch.testing.assert_close(fused_low, current, rtol=0.0, atol=0.0)

    out_of_bounds_flow = torch.full((2, 3, 4), 100.0, dtype=torch.float32)
    fused_hole = fuse_depth_with_confidence(
        current,
        previous,
        out_of_bounds_flow,
        torch.ones_like(current),
        history_weight=1.0,
    )
    torch.testing.assert_close(fused_hole, current, rtol=0.0, atol=0.0)


def test_backward_flow_aligns_a_one_pixel_translation() -> None:
    previous = torch.zeros((3, 6), dtype=torch.float32)
    previous[:, 2] = 1.0

    current = torch.zeros_like(previous)
    current[:, 3] = 1.0

    # Current x samples previous x - 1: the feature moved one pixel right.
    backward_flow = _zero_flow(3, 6)
    backward_flow[0].fill_(-1.0)

    fused = fuse_depth_with_confidence(
        current,
        previous,
        backward_flow,
        torch.ones_like(current),
        history_weight=1.0,
    )
    torch.testing.assert_close(fused, current, rtol=0.0, atol=1e-6)


def test_confidence_is_bounded_before_weighting() -> None:
    current = torch.ones((1, 2), dtype=torch.float32)
    previous = torch.zeros_like(current)
    confidence = torch.tensor([[-2.0, 2.0]], dtype=torch.float32)

    fused = fuse_depth_with_confidence(
        current,
        previous,
        _zero_flow(1, 2),
        confidence,
        history_weight=1.0,
    )

    # confidence -2 -> 0: current fallback; confidence 2 -> 1: previous.
    torch.testing.assert_close(
        fused,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_temporal_support_is_exactly_two_raw_frames() -> None:
    fusion = FlowGuidedDepthFusion(history_weight=0.5)
    confidence = torch.ones((2, 2), dtype=torch.float32)
    flow = _zero_flow(2, 2)

    first = fusion.update(torch.zeros((2, 2), dtype=torch.float32))
    second = fusion.update(
        torch.ones((2, 2), dtype=torch.float32),
        backward_flow=flow,
        confidence=confidence,
    )
    third = fusion.update(
        torch.full((2, 2), 3.0, dtype=torch.float32),
        backward_flow=flow,
        confidence=confidence,
    )

    torch.testing.assert_close(first, torch.zeros_like(first), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        second, torch.full_like(second, 0.5), rtol=0.0, atol=1e-6
    )
    # Must use raw frame 2 (=1), not fused frame 2 (=0.5):
    # 0.5 * current(3) + 0.5 * previous_raw(1) = 2.
    torch.testing.assert_close(
        third, torch.full_like(third, 2.0), rtol=0.0, atol=1e-6
    )


def test_30_frame_temporal_variance_threshold() -> None:
    generator = torch.Generator().manual_seed(20260724)
    height, width = 48, 64
    ground_truth = torch.linspace(
        0.0, 1.0, steps=height * width, dtype=torch.float32
    ).reshape(height, width)
    raw_frames = [
        ground_truth
        + torch.randn(
            (height, width), generator=generator, dtype=torch.float32
        ) * 0.006
        for _ in range(30)
    ]

    fusion = FlowGuidedDepthFusion(history_weight=0.5)
    zero_flow = _zero_flow(height, width)
    confidence = torch.ones((height, width), dtype=torch.float32)
    fused_frames = []
    for index, raw_depth in enumerate(raw_frames):
        if index == 0:
            fused_frames.append(fusion.update(raw_depth))
        else:
            fused_frames.append(
                fusion.update(
                    raw_depth,
                    backward_flow=zero_flow,
                    confidence=confidence,
                )
            )

    depth_range = float(ground_truth.max() - ground_truth.min())
    raw_ratio = float(
        torch.stack(raw_frames).std(dim=0, correction=0).mean()
    ) / depth_range
    fused_ratio = float(
        torch.stack(fused_frames).std(dim=0, correction=0).mean()
    ) / depth_range

    assert raw_ratio > 0.005, raw_ratio
    assert fused_ratio < 0.005, fused_ratio
    assert fused_ratio < raw_ratio
    print(
        "  30-frame mean temporal std/depth-range: "
        f"raw={raw_ratio:.6f}, fused={fused_ratio:.6f}"
    )


def main() -> None:
    tests = [
        test_static_high_confidence_blend,
        test_low_confidence_and_warp_holes_fall_back_to_current,
        test_backward_flow_aligns_a_one_pixel_translation,
        test_confidence_is_bounded_before_weighting,
        test_temporal_support_is_exactly_two_raw_frames,
        test_30_frame_temporal_variance_threshold,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)} FLOW-GUIDED DEPTH FUSION CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
