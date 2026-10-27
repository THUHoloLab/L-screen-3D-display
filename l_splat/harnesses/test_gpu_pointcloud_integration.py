"""Production-contract tests for the CUDA pointcloud integration."""

from __future__ import annotations

import gc
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

import l_splat.core.pipeline as pipeline
from l_splat.core.gpu_pointcloud import depth_to_unified_pointcloud_cuda


HEIGHT_PX = 36
WIDTH_PX = 64
POINT_ATOL_M = 1.0e-6
COLOR_ATOL = 1.0e-7
MODEL_ATOL = 3.0e-6
OPACITY_LOGIT_ATOL = 2.0e-4
OPACITY_ALPHA_ATOL = 5.0e-5


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic RGB, raw depth, and externally anchored depth.

    Inputs: none; fixed named dimensions are used. Outputs: RGB uint8
    ``(H,W,3)``, raw depth float32 ``(H,W)``, and anchored normalized depth
    float32 ``(H,W)`` in ``[0,1]``. Formula: channel ramps plus a smooth
    linear/sinusoidal surface and a fixed external normalization. Failure modes:
    none for the positive fixed dimensions.
    """
    rows, cols = np.mgrid[0:HEIGHT_PX, 0:WIDTH_PX].astype(np.float32)
    rgb = np.stack([
        np.mod(cols * 7.0 + rows * 3.0, 256.0),
        np.mod(cols * 5.0 + rows * 11.0, 256.0),
        np.mod(cols * 13.0 + rows * 2.0, 256.0),
    ], axis=-1).astype(np.uint8)
    depth = (
        0.25
        + 0.015 * cols
        + 0.01 * rows
        + 0.05 * np.sin(cols * 0.21)
    ).astype(np.float32)
    anchored = np.clip((depth - 0.10) / 1.80, 0.0, 1.0).astype(np.float32)
    return rgb, depth, anchored


def _cpu_pointcloud(
        rgb: np.ndarray,
        depth: np.ndarray,
        depth_norm: np.ndarray | None) -> tuple[np.ndarray, ...]:
    """Run the unchanged NumPy production oracle for one synthetic frame.

    Inputs: RGB uint8, raw float32 depth, and optional anchored normalized depth.
    Outputs: seven production NumPy arrays. Formula: delegated to the canonical
    CPU pointcloud function. Failure modes: production validation/allocation
    errors propagate; stdout is captured only to keep test evidence concise.
    """
    with redirect_stdout(StringIO()):
        return pipeline.depth_to_unified_pointcloud(
            rgb,
            depth,
            pipeline.DEPTH_FWD_M,
            pipeline.DEPTH_BACK_M,
            pipeline.CONTENT_W_M,
            pipeline.PANEL_PHYS_H_M,
            depth_norm=depth_norm,
            fg_mask=None,
            return_edge_aware_mask=True,
        )


def _gpu_pointcloud(
        rgb: np.ndarray,
        depth: np.ndarray,
        depth_norm: np.ndarray | None,
        *,
        max_gaussians: int | None = None,
        ) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Run the production CUDA helper with current canonical constants.

    Inputs: synthetic RGB/depth, optional anchored depth and optional capacity.
    Outputs: seven CUDA tensors plus metadata. Formula: delegated to the new
    production helper using the same constants as the CPU oracle. Failure modes:
    CUDA/helper validation and allocation errors propagate.
    """
    capacity = (
        int(pipeline.MAX_GAUSSIANS)
        if max_gaussians is None
        else int(max_gaussians)
    )
    return depth_to_unified_pointcloud_cuda(
        rgb,
        depth,
        pipeline.DEPTH_FWD_M,
        pipeline.DEPTH_BACK_M,
        pipeline.CONTENT_W_M,
        pipeline.PANEL_PHYS_H_M,
        sweet_spot=tuple(float(value) for value in pipeline.SWEET_SPOT),
        super_sample_size=int(pipeline.SUPER_SAMPLE_SIZE),
        local_spacing_threshold=float(pipeline.LOCAL_SPACING_THRESHOLD),
        max_gaussians=capacity,
        depth_norm=depth_norm,
        device=pipeline.DEVICE,
    )


def _to_numpy(tensors: tuple[torch.Tensor, ...]) -> tuple[np.ndarray, ...]:
    """Copy a seven-tensor production pointcloud result to contiguous NumPy.

    Inputs: seven CUDA tensors in production order. Outputs: seven NumPy arrays
    preserving shape/dtype/order. Formula: detached D2H copy. Failure modes:
    tuple-size or CUDA transfer errors fail the test.
    """
    if len(tensors) != 7:
        raise AssertionError(f"expected seven tensors, got {len(tensors)}")
    return tuple(
        np.ascontiguousarray(value.detach().cpu().numpy()) for value in tensors)


def _assert_pointcloud_equivalent(
        expected: tuple[np.ndarray, ...],
        actual: tuple[np.ndarray, ...]) -> tuple[float, float]:
    """Assert continuous tolerances and exact discrete pointcloud contracts.

    Inputs: seven CPU-oracle and candidate NumPy arrays. Outputs: point/color
    maximum absolute errors. Formula: absolute float32 difference for the first
    two arrays and byte-exact equality for the remaining five. Failure modes:
    count, order, shape, dtype, tolerance, or metadata drift raises AssertionError.
    """
    names = (
        "points_world",
        "colors",
        "x_sign",
        "is_super_sampled",
        "panel_owner",
        "is_bg_point",
        "edge_aware_mask",
    )
    assert len(expected) == len(actual) == len(names)
    for name, left, right in zip(names, expected, actual):
        assert left.shape == right.shape, (name, left.shape, right.shape)
        assert left.dtype == right.dtype, (name, left.dtype, right.dtype)
    point_error = float(np.abs(expected[0] - actual[0]).max(initial=0.0))
    color_error = float(np.abs(expected[1] - actual[1]).max(initial=0.0))
    assert point_error <= POINT_ATOL_M, point_error
    assert color_error <= COLOR_ATOL, color_error
    for index, name in enumerate(names[2:], start=2):
        assert np.array_equal(expected[index], actual[index]), name
    return point_error, color_error


def test_anchored_cuda_pointcloud_matches_cpu() -> None:
    """Verify anchored depth semantics and every seven-array output contract.

    Inputs: deterministic synthetic frame with externally supplied depth_norm.
    Outputs: numerical stdout and assertions. Formula: canonical CPU versus CUDA
    comparison. Failure modes: CUDA absence, ignored anchoring, layout/dtype drift,
    or tolerance violation fails the test.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production integration test")
    rgb, depth, anchored = _inputs()
    cpu = _cpu_pointcloud(rgb, depth, anchored)
    gpu_tensors, metadata = _gpu_pointcloud(rgb, depth, anchored)
    point_error, color_error = _assert_pointcloud_equivalent(
        cpu, _to_numpy(gpu_tensors))
    assert metadata["depth_normalization"] == "provided"
    assert metadata["supersampling_skipped_for_capacity"] is False
    print(
        "anchored CUDA pointcloud: "
        f"points={cpu[0].shape[0]}, super={int(np.count_nonzero(cpu[3]))}, "
        f"point_max_abs={point_error:.3e}m, "
        f"color_max_abs={color_error:.3e}"
    )


def test_cuda_capacity_matches_cpu_skip_semantics() -> None:
    """Verify capacity overflow keeps base points instead of raising or allocating.

    Inputs: deterministic synthetic frame and capacity ``base+1``. Outputs:
    assertions and metadata. Formula: both CPU and CUDA must skip all 4x4 added
    candidates when expected total exceeds capacity. Failure modes: global-state
    restoration, allocation, count, array, or skip-flag drift fails the test.
    """
    rgb, depth, anchored = _inputs()
    base_count = HEIGHT_PX * WIDTH_PX
    capacity = base_count + 1
    original_capacity = pipeline.MAX_GAUSSIANS
    pipeline.MAX_GAUSSIANS = capacity
    try:
        cpu = _cpu_pointcloud(rgb, depth, anchored)
    finally:
        pipeline.MAX_GAUSSIANS = original_capacity
    gpu_tensors, metadata = _gpu_pointcloud(
        rgb, depth, anchored, max_gaussians=capacity)
    point_error, color_error = _assert_pointcloud_equivalent(
        cpu, _to_numpy(gpu_tensors))
    assert cpu[0].shape[0] == base_count
    assert int(np.count_nonzero(cpu[3])) == 0
    assert metadata["supersampling_skipped_for_capacity"] is True
    assert metadata["candidate_point_count"] > capacity
    print(
        "capacity skip: "
        f"base={base_count}, candidate={metadata['candidate_point_count']}, "
        f"limit={capacity}, point_max_abs={point_error:.3e}m, "
        f"color_max_abs={color_error:.3e}"
    )


def test_cuda_tensors_feed_gaussian_model_without_xyz_copy() -> None:
    """Verify CUDA point tensors flow directly into the canonical Gaussian model.

    Inputs: anchored synthetic CPU/CUDA pointclouds. Outputs: model tensor
    equivalence and data-pointer assertions. Formula: build isotropic models with
    identical parameters and compare xyz/features/scales/rotation/opacity/masks.
    Failure modes: an implicit D2H/H2D copy, unsupported tensor conversion, or
    numerical/model-contract drift fails the test.
    """
    rgb, depth, anchored = _inputs()
    cpu = _cpu_pointcloud(rgb, depth, anchored)
    gpu, _metadata = _gpu_pointcloud(rgb, depth, anchored)
    common = {
        "sh_degree": pipeline.SH_DEGREE,
        "img_width": WIDTH_PX,
        "content_w": pipeline.CONTENT_W_M,
        "overlap_factor": pipeline.GAUSSIAN_OVERLAP_FACTOR,
        "target_opacity": pipeline.GAUSSIAN_OPACITY_SIGMOID,
        "edge_aware_repair_enabled": False,
        "representation_contract_prototype_enabled": False,
        "surface_patch_level_prototype_enabled": False,
        "gaussian_representation": "isotropic",
    }
    with redirect_stdout(StringIO()):
        cpu_model = pipeline.create_gaussian_model(
            cpu[0], cpu[1], is_super_sampled=cpu[3], panel_owner=cpu[4],
            is_bg_point=cpu[5], edge_aware_mask=cpu[6], **common)
        gpu_model = pipeline.create_gaussian_model(
            gpu[0], gpu[1], is_super_sampled=gpu[3], panel_owner=gpu[4],
            is_bg_point=gpu[5], edge_aware_mask=gpu[6], **common)

    assert gpu_model.get_xyz.data_ptr() == gpu[0].data_ptr()
    assert gpu_model._panel_owner.data_ptr() == gpu[4].data_ptr()
    assert gpu_model._is_super_sampled.data_ptr() == gpu[3].data_ptr()
    for name in (
            "_xyz", "_features_dc", "_features_rest", "_scaling",
            "_rotation", "_opacity"):
        expected = getattr(cpu_model, name)
        actual = getattr(gpu_model, name)
        assert expected.shape == actual.shape, name
        max_abs = float((expected - actual).abs().max().item()) \
            if expected.numel() else 0.0
        tolerance = OPACITY_LOGIT_ATOL if name == "_opacity" else MODEL_ATOL
        assert max_abs <= tolerance, (name, max_abs, tolerance)
        if name == "_opacity":
            alpha_max_abs = float(
                (torch.sigmoid(expected) - torch.sigmoid(actual)).abs().max().item()
            ) if expected.numel() else 0.0
            assert alpha_max_abs <= OPACITY_ALPHA_ATOL, alpha_max_abs
    assert torch.equal(cpu_model._panel_owner, gpu_model._panel_owner)
    assert torch.equal(cpu_model._is_super_sampled, gpu_model._is_super_sampled)
    print(
        "Gaussian tensor boundary: "
        f"points={gpu_model.get_xyz.shape[0]}, xyz_ptr_reused=True, "
        f"owner_ptr_reused=True, super_ptr_reused=True"
    )


def test_backend_resolution_is_explicit_and_fail_closed() -> None:
    """Verify auto/CPU/strict-CUDA backend selection and unsupported fallback.

    Inputs: canonical supported flags plus one diagnostic-profile variant.
    Outputs: resolver assertions and source-import isolation check. Formula:
    explicit CPU wins, canonical auto selects CUDA, unsupported auto selects CPU,
    and strict CUDA raises. Failure modes: silent fallback or harness-to-production
    import coupling fails the test.
    """
    supported = {
        "save_profile": "minimal",
        "fg_mask_present": False,
        "depth_structure_mode": "off",
        "gaussian_representation": "isotropic",
        "edge_aware_gaussian_repair": False,
        "representation_contract_prototype": False,
        "surface_patch_level_prototype": False,
        "needs_representation_categories": False,
        "needs_geometry_metadata": False,
    }
    backend, reason = pipeline._resolve_pointcloud_backend("auto", **supported)
    assert backend == "cuda", reason
    backend, reason = pipeline._resolve_pointcloud_backend("cpu", **supported)
    assert backend == "cpu" and "explicit" in reason.lower()

    unsupported = dict(supported)
    unsupported["save_profile"] = "diagnostic"
    backend, reason = pipeline._resolve_pointcloud_backend("auto", **unsupported)
    assert backend == "cpu"
    assert "save_profile" in reason
    try:
        pipeline._resolve_pointcloud_backend("cuda", **unsupported)
    except RuntimeError as exc:
        assert "save_profile" in str(exc)
    else:
        raise AssertionError("strict CUDA accepted an unsupported diagnostic mode")

    original_sweet_spot = pipeline.SWEET_SPOT
    pipeline.SWEET_SPOT = np.array([0.0, 0.0, 0.4], dtype=np.float64)
    try:
        backend, reason = pipeline._resolve_pointcloud_backend(
            "auto", **supported)
        assert backend == "cpu" and "sweet_spot" in reason
        try:
            pipeline._resolve_pointcloud_backend("cuda", **supported)
        except RuntimeError as exc:
            assert "sweet_spot" in str(exc)
        else:
            raise AssertionError("strict CUDA accepted zero eye-normal geometry")
    finally:
        pipeline.SWEET_SPOT = original_sweet_spot

    pipeline_source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "l_splat.harnesses" not in pipeline_source
    assert 'if runtime_pointcloud_backend == "cuda":' in pipeline_source
    print(
        "backend resolution: auto=cuda, cpu=cpu, unsupported/zero-eye auto=cpu, "
        "resolved CUDA frame errors=re-raised"
    )


def test_repeated_cuda_build_releases_live_allocations() -> None:
    """Verify repeated pointcloud/model builds do not retain per-frame tensors.

    Inputs: eight identical anchored synthetic frames after one allocator
    warm-up. Outputs: retained allocated/reserved memory samples. Formula:
    delete each frame model/result, collect Python garbage, synchronize CUDA,
    then require live allocations and steady-state reservation to remain within
    small fixed tolerances. Failure modes: a retained model/tensor reference or
    unbounded caching growth fails the test.
    """
    rgb, depth, anchored = _inputs()
    common = {
        "sh_degree": pipeline.SH_DEGREE,
        "img_width": WIDTH_PX,
        "content_w": pipeline.CONTENT_W_M,
        "overlap_factor": pipeline.GAUSSIAN_OVERLAP_FACTOR,
        "target_opacity": pipeline.GAUSSIAN_OPACITY_SIGMOID,
        "edge_aware_repair_enabled": False,
        "representation_contract_prototype_enabled": False,
        "surface_patch_level_prototype_enabled": False,
        "gaussian_representation": "isotropic",
    }

    warm_tensors, _warm_metadata = _gpu_pointcloud(rgb, depth, anchored)
    with redirect_stdout(StringIO()):
        warm_model = pipeline.create_gaussian_model(
            warm_tensors[0], warm_tensors[1],
            is_super_sampled=warm_tensors[3],
            panel_owner=warm_tensors[4],
            is_bg_point=warm_tensors[5],
            edge_aware_mask=warm_tensors[6],
            **common,
        )
    torch.cuda.synchronize()
    del warm_model, warm_tensors
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())

    allocated_samples: list[int] = []
    reserved_samples: list[int] = []
    for _index in range(8):
        tensors, _metadata = _gpu_pointcloud(rgb, depth, anchored)
        with redirect_stdout(StringIO()):
            model = pipeline.create_gaussian_model(
                tensors[0], tensors[1],
                is_super_sampled=tensors[3],
                panel_owner=tensors[4],
                is_bg_point=tensors[5],
                edge_aware_mask=tensors[6],
                **common,
            )
        torch.cuda.synchronize()
        del model, tensors
        gc.collect()
        torch.cuda.synchronize()
        allocated_samples.append(int(torch.cuda.memory_allocated()))
        reserved_samples.append(int(torch.cuda.memory_reserved()))

    mib = 1024 * 1024
    allocated_tolerance = 2 * mib
    reserved_tolerance = 8 * mib
    assert max(allocated_samples) - min(allocated_samples) <= allocated_tolerance
    assert allocated_samples[-1] <= baseline_allocated + allocated_tolerance
    steady_reserved = reserved_samples[1:]
    assert max(steady_reserved) - min(steady_reserved) <= reserved_tolerance
    print(
        "repeated CUDA build memory: "
        f"allocated_span={(max(allocated_samples)-min(allocated_samples))/mib:.3f}MiB, "
        f"final_delta={(allocated_samples[-1]-baseline_allocated)/mib:.3f}MiB, "
        f"steady_reserved_span={(max(steady_reserved)-min(steady_reserved))/mib:.3f}MiB"
    )


def main() -> None:
    """Run all production CUDA integration contract tests.

    Inputs: none. Outputs: numerical stdout and assertion status; no artifacts are
    written. Formula: sequential execution of anchored, capacity, model-boundary,
    backend-resolution, and repeated-memory checks. Failure modes: any exception
    exits non-zero.
    """
    test_anchored_cuda_pointcloud_matches_cpu()
    test_cuda_capacity_matches_cpu_skip_semantics()
    test_cuda_tensors_feed_gaussian_model_without_xyz_copy()
    test_backend_resolution_is_explicit_and_fail_closed()
    test_repeated_cuda_build_releases_live_allocations()
    print("test_gpu_pointcloud_integration: 5/5 passed")


if __name__ == "__main__":
    main()
