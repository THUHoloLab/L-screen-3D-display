"""Source-level contracts for the traced production runtime cleanup.

This harness deliberately avoids importing CUDA.  It protects architectural
decisions established by the source trace of ``最终播放.mp4``.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_pipeline_uses_cli_sweet_spot_and_no_segmentation_stage() -> None:
    source = _text("l_splat/core/pipeline.py")

    assert "distance_cm=args.sweet_spot_distance_cm" in source
    assert "height_cm=args.sweet_spot_height_cm" in source
    assert "from l_splat.segmentation" not in source
    assert "segment_foreground(" not in source
    assert "load_sam3_model(" not in source


def test_frustum_is_canonical_and_dual_module_is_only_an_adapter() -> None:
    tree = ast.parse(_text("l_splat/core/dual_frustum.py"))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }

    assert "DualFrustumCamera" in definitions
    assert "get_asymmetric_frustum" in definitions
    assert "getProjectionMatrixAsymmetric" not in definitions
    assert "render_dual_frustum" not in definitions
    assert "make_default_dual_camera" not in definitions


def test_asymmetric_cuda_uses_a_common_forward_distance_guard() -> None:
    source = _text(
        "gaussian-splatting-main/submodules/diff-gaussian-rasterization/"
        "cuda_rasterizer/forward.cu"
    )

    assert "const float view_depth = use_asymmetric ? -p_view.z : p_view.z;" in source
    assert "if (view_depth <= 0.2f)" in source
    assert "fabsf(p_view.z) < 1.0f" not in source


def test_incomplete_zero_copy_prototype_is_not_active() -> None:
    assert not (ROOT / "l_splat/core/gaussian_pool.py").exists()
    assert not (ROOT / "l_splat/harnesses/test_zero_copy.py").exists()


def main() -> None:
    tests = [
        test_pipeline_uses_cli_sweet_spot_and_no_segmentation_stage,
        test_frustum_is_canonical_and_dual_module_is_only_an_adapter,
        test_asymmetric_cuda_uses_a_common_forward_distance_guard,
        test_incomplete_zero_copy_prototype_is_not_active,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)} CANONICAL RUNTIME CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
