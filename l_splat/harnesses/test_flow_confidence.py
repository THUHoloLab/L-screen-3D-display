"""Deterministic contract tests for FlowConfidenceEstimator without weights."""

from __future__ import annotations

import numpy as np
import torch

from l_splat.flow_confidence import FlowConfidenceEstimator


class _ConstantBidirectionalRAFT(torch.nn.Module):
    """Return +2/-2 horizontal work-grid flow based on frame ordering."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image1, image2, num_flow_updates=12):
        self.calls.append(int(num_flow_updates))
        direction = torch.sign(image2.mean() - image1.mean())
        flow = torch.zeros(
            (image1.shape[0], 2, image1.shape[-2], image1.shape[-1]),
            device=image1.device,
            dtype=image1.dtype,
        )
        flow[:, 0].fill_(float(direction.item()) * 2.0)
        return [flow]


def _make_estimator(max_side=48):
    model = _ConstantBidirectionalRAFT()
    estimator = FlowConfidenceEstimator(
        model_size="small",
        device="cpu",
        fb_threshold=0.5,
        max_side=max_side,
        num_flow_updates=12,
        model=model,
    )
    return estimator, model


def test_preprocessing_matches_official_weight_transform():
    estimator, _ = _make_estimator()
    rng = np.random.default_rng(20260724)
    image = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    actual = estimator._img_to_tensor(image)

    from torchvision.models.optical_flow import Raft_Small_Weights

    official = Raft_Small_Weights.C_T_V2.transforms()
    chw = torch.from_numpy(image.copy()).permute(2, 0, 1)
    expected, _ = official(chw, chw)
    torch.testing.assert_close(actual[0], expected, rtol=0.0, atol=0.0)


def test_downscaled_flow_vectors_are_rescaled_to_source_pixels():
    estimator, model = _make_estimator(max_side=48)
    previous = np.zeros((64, 96, 3), dtype=np.uint8)
    current = np.full_like(previous, 255)
    pair = estimator.estimate_pair(previous, current)

    assert pair.forward.shape == (1, 2, 64, 96)
    assert pair.backward.shape == (1, 2, 64, 96)
    assert pair.confidence.shape == (1, 64, 96)
    assert pair.valid.shape == (1, 64, 96)
    assert pair.scaled_resolution == (32, 48)
    assert pair.working_resolution == (32, 48)
    assert model.calls == [12, 12]

    # Work-grid +2 px at half resolution must become +4 source pixels.
    torch.testing.assert_close(
        pair.forward[:, 0],
        torch.full_like(pair.forward[:, 0], 4.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        pair.backward[:, 0],
        torch.full_like(pair.backward[:, 0], -4.0),
        rtol=0.0,
        atol=0.0,
    )
    assert float(pair.confidence.min()) >= 0.0
    assert float(pair.confidence.max()) <= 1.0
    assert float(pair.confidence[:, :, 4:].mean()) == 1.0


def test_current_domain_invalid_border_and_legacy_mask():
    estimator, _ = _make_estimator(max_side=48)
    previous = np.zeros((64, 96, 3), dtype=np.uint8)
    current = np.full_like(previous, 255)
    pair = estimator.estimate_pair(previous, current)

    # Backward flow is -4 px: current columns [0,3] sample outside previous.
    assert not bool(pair.valid[:, :, :4].any())
    assert bool(pair.valid[:, :, 4:].all())
    mask = estimator.compute_mask(previous, current)
    assert mask.shape == (64, 96)
    assert mask.dtype == np.float32
    assert np.all(mask[:, :4] == 0.0)
    assert np.all(mask[:, 4:] == 1.0)


def test_sampling_grid_is_reused():
    estimator, _ = _make_estimator(max_side=48)
    previous = np.zeros((64, 96, 3), dtype=np.uint8)
    current = np.full_like(previous, 255)
    estimator.estimate_pair(previous, current)
    assert len(estimator._grid_cache) == 1
    first_grid = next(iter(estimator._grid_cache.values()))
    estimator.estimate_pair(previous, current)
    second_grid = next(iter(estimator._grid_cache.values()))
    assert first_grid.data_ptr() == second_grid.data_ptr()


def test_sequential_mode_reuses_one_preprocessed_frame():
    estimator, _ = _make_estimator(max_side=48)
    original_convert = estimator._img_to_tensor
    conversion_count = {"value": 0}

    def counted_convert(image):
        conversion_count["value"] += 1
        return original_convert(image)

    estimator._img_to_tensor = counted_convert
    first = np.zeros((64, 96, 3), dtype=np.uint8)
    second = np.full_like(first, 127)
    third = np.full_like(first, 255)
    estimator.estimate_pair(first, second, reuse_previous=True)
    reused = estimator.estimate_pair(second, third, reuse_previous=True)
    assert conversion_count["value"] == 3

    reference_estimator, _ = _make_estimator(max_side=48)
    reference = reference_estimator.estimate_pair(second, third)
    torch.testing.assert_close(
        reused.forward, reference.forward, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        reused.backward, reference.backward, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        reused.confidence, reference.confidence, rtol=0.0, atol=0.0)
    estimator.reset_sequence()
    assert estimator._sequence_previous_tensor is None


def main():
    tests = [
        test_preprocessing_matches_official_weight_transform,
        test_downscaled_flow_vectors_are_rescaled_to_source_pixels,
        test_current_domain_invalid_border_and_legacy_mask,
        test_sampling_grid_is_reused,
        test_sequential_mode_reuses_one_preprocessed_frame,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)} FLOW CONFIDENCE CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
