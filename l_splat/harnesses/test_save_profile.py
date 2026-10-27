"""Behavioral tests for production-minimal and diagnostic frame saving."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch

from l_splat.core.pipeline import save_results


def _render_output(height: int = 8, width: int = 10) -> dict:
    left = torch.linspace(0.0, 1.0, steps=3 * height * width).reshape(
        3, height, width
    )
    right = torch.flip(left, dims=[2])
    return {"render_left": left, "render_right": right}


def test_minimal_profile_writes_only_runtime_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        save_results(
            _render_output(),
            input_rgb=np.zeros((8, 20, 3), dtype=np.uint8),
            out_dir=tmp,
            save_profile="minimal",
        )
        names = {path.name for path in Path(tmp).iterdir()}

    assert names == {
        "parameter_provenance.json",
        "render_left.png",
        "render_raw_combined.png",
        "render_right.png",
    }


def test_diagnostic_profile_preserves_review_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        save_results(
            _render_output(),
            input_rgb=np.zeros((8, 20, 3), dtype=np.uint8),
            out_dir=tmp,
            save_profile="diagnostic",
        )
        names = {path.name for path in Path(tmp).iterdir()}

    required = {
        "input_preview.png",
        "render_blended_preview.png",
        "render_left.png",
        "render_raw_combined.png",
        "render_right.png",
        "render_stitched.png",
        "seam_crop_left_edge.png",
        "seam_crop_raw_combined.png",
        "seam_crop_right_edge.png",
    }
    assert required <= names


def main() -> None:
    tests = [
        test_minimal_profile_writes_only_runtime_artifacts,
        test_diagnostic_profile_preserves_review_artifacts,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)} SAVE PROFILE TESTS PASSED")


if __name__ == "__main__":
    main()
