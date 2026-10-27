"""Regression harness for super-sampled Gaussian scale capping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l_splat.core import pipeline as pipe


def test_super_sampled_points_use_subpixel_scale_cap() -> None:
    points = np.asarray(
        [
            [-0.10, 0.00, 0.00],
            [-0.05, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    colors = np.ones((2, 3), dtype=np.float32)
    is_super = np.asarray([False, True], dtype=bool)
    owner = np.asarray([0, 0], dtype=np.uint8)
    img_width = 8
    content_w = 0.8
    overlap = 1.0

    model = pipe.create_gaussian_model(
        points,
        colors,
        sh_degree=0,
        img_width=img_width,
        content_w=content_w,
        overlap_factor=overlap,
        is_super_sampled=is_super,
        panel_owner=owner,
        target_opacity=0.99,
        fade_threshold=-1.0,
    )
    scales = model.get_scaling.detach().cpu().numpy()
    expected_super = (content_w / float(img_width)) * overlap / float(pipe.SUPER_SAMPLE_SIZE)

    assert np.allclose(scales[1], expected_super, rtol=1e-5, atol=1e-7), scales[1]
    assert float(scales[1, 0]) < float(scales[0, 0])


def harness() -> None:
    test_super_sampled_points_use_subpixel_scale_cap()
    print("Super-sample scale regression passed")


if __name__ == "__main__":
    harness()
