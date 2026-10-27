"""Contract tests and a microbenchmark for video Gaussian storage reuse."""

from __future__ import annotations

import gc
import inspect
import json
import tempfile
import statistics
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

import l_splat.core.pipeline as pipeline


PARAMETER_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)
ROUTING_NAMES = ("_panel_owner", "_is_super_sampled")


def _frame(point_count: int, offset: float) -> dict[str, torch.Tensor]:
    """Create one deterministic canonical Gaussian-construction input frame.

    Inputs are a positive point count and a scalar frame offset. Outputs are
    CUDA tensors for points, RGB, panel ownership, super-sample flags, background
    flags, and edge flags. Coordinates are smooth non-degenerate ramps; RGB is
    clamped to ``[0,1]``. CUDA absence or a non-positive count fails explicitly.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gaussian frame-reuse tests")
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    device = pipeline.DEVICE
    index = torch.arange(point_count, device=device, dtype=torch.float32)
    denominator = float(max(point_count - 1, 1))
    phase = index / denominator
    points = torch.stack([
        -0.62 + 0.48 * phase + offset,
        -0.57 + 0.41 * torch.sin(phase * 1.1) - 0.5 * offset,
        0.18 + 0.83 * phase + 0.25 * offset,
    ], dim=1).contiguous()
    colors = torch.stack([
        (0.12 + 0.61 * phase + offset).clamp(0.0, 1.0),
        (0.83 - 0.47 * phase + 0.5 * offset).clamp(0.0, 1.0),
        (0.21 + 0.33 * torch.cos(phase * 1.7) + offset).clamp(0.0, 1.0),
    ], dim=1).contiguous()
    return {
        "points": points,
        "colors": colors,
        "panel_owner": (torch.arange(point_count, device=device) % 2).to(torch.uint8),
        "is_super_sampled": (
            torch.arange(point_count, device=device) % 5 == 0),
        "is_bg_point": torch.zeros(point_count, device=device, dtype=torch.bool),
        "edge_aware_mask": torch.zeros(point_count, device=device, dtype=torch.bool),
    }


def _build(
        frame: dict[str, torch.Tensor],
        reuse_state=None,
        sh_degree: int = 1):
    """Build a deterministic model at the requested SH degree with reuse.

    Inputs are one dictionary from :func:`_frame` and an optional production
    reuse state plus an SH degree (one by default for non-empty rest features).
    The output is a production ``GaussianModel``. Formulae and validation are
    delegated unchanged to ``create_gaussian_model``; stdout is captured for
    concise evidence. Production exceptions propagate unchanged.
    """
    with redirect_stdout(StringIO()):
        return pipeline.create_gaussian_model(
            frame["points"],
            frame["colors"],
            sh_degree=sh_degree,
            img_width=64,
            content_w=pipeline.CONTENT_W_M,
            overlap_factor=pipeline.GAUSSIAN_OVERLAP_FACTOR,
            is_super_sampled=frame["is_super_sampled"],
            panel_owner=frame["panel_owner"],
            target_opacity=pipeline.GAUSSIAN_OPACITY_SIGMOID,
            is_bg_point=frame["is_bg_point"],
            edge_aware_mask=frame["edge_aware_mask"],
            edge_aware_repair_enabled=False,
            representation_contract_prototype_enabled=False,
            surface_patch_level_prototype_enabled=False,
            gaussian_representation="isotropic",
            reuse_state=reuse_state,
        )


def _assert_models_equal(expected, actual) -> None:
    """Assert exact tensor and routing equivalence between two Gaussian models.

    Inputs are a normal-rebuild oracle and a reuse candidate. There is no output.
    Every parameter and routing tensor must have identical shape, dtype, and
    values. Shape, dtype, numerical, or missing-metadata drift raises an assertion.
    """
    for name in PARAMETER_NAMES + ROUTING_NAMES:
        expected_tensor = getattr(expected, name)
        actual_tensor = getattr(actual, name)
        assert expected_tensor.shape == actual_tensor.shape, name
        assert expected_tensor.dtype == actual_tensor.dtype, name
        assert torch.equal(expected_tensor, actual_tensor), name


def test_compatible_frame_updates_in_place() -> None:
    """Verify same-sized consecutive frames reuse model and backing storage.

    Two distinct 24-point CUDA frames are used. The second reuse result must be
    bit-exact to a normal rebuild while retaining model identity and all six
    parameter plus two routing storage pointers. Rebinding, stale values, or an
    unexpected allocation/reuse counter fails the test.
    """
    state = pipeline.GaussianFrameReuseState()
    first = _frame(24, 0.0)
    second = _frame(24, 0.017)
    reused_first = _build(first, state)
    parameter_pointers = {
        name: getattr(reused_first, name).data_ptr()
        for name in PARAMETER_NAMES + ROUTING_NAMES
    }
    rebuilt_second = _build(second)
    reused_second = _build(second, state)

    assert reused_second is reused_first
    for name, pointer in parameter_pointers.items():
        assert getattr(reused_second, name).data_ptr() == pointer, name
    _assert_models_equal(rebuilt_second, reused_second)
    assert state.capacity == 24
    assert state.active_count == 24
    assert state.allocation_count == 1
    assert state.reuse_count == 1
    assert state.reallocation_count == 0
    print("compatible reuse: model_identity=True, stable_pointers=8/8, exact=True")


def test_shrink_reuses_capacity_and_growth_reallocates_safely() -> None:
    """Verify active-prefix reuse for shrink and safe capacity growth.

    The state receives 24, 13, then 31 points. The 13-point frame must retain
    every storage pointer while exposing exact 13-row active views. The 31-point
    frame must keep model identity, increase capacity, record one reallocation,
    and remain bit-exact to a normal rebuild. Stale tails are excluded by shape.
    """
    state = pipeline.GaussianFrameReuseState()
    model = _build(_frame(24, 0.0), state)
    initial_pointers = {
        name: getattr(model, name).data_ptr()
        for name in PARAMETER_NAMES + ROUTING_NAMES
    }

    smaller = _frame(13, 0.021)
    rebuilt_smaller = _build(smaller)
    reused_smaller = _build(smaller, state)
    assert reused_smaller is model
    for name, pointer in initial_pointers.items():
        assert getattr(reused_smaller, name).data_ptr() == pointer, name
        assert getattr(reused_smaller, name).shape[0] == 13, name
    _assert_models_equal(rebuilt_smaller, reused_smaller)
    assert state.capacity == 24
    assert state.active_count == 13

    larger = _frame(31, 0.034)
    rebuilt_larger = _build(larger)
    reused_larger = _build(larger, state)
    assert reused_larger is model
    _assert_models_equal(rebuilt_larger, reused_larger)
    assert state.capacity == 31
    assert state.active_count == 31
    assert state.allocation_count == 2
    assert state.reuse_count == 1
    assert state.reallocation_count == 1
    changed = sum(
        getattr(reused_larger, name).data_ptr() != pointer
        for name, pointer in initial_pointers.items()
    )
    assert changed == len(PARAMETER_NAMES + ROUTING_NAMES), changed
    print("resize reuse: shrink_stable=8/8, growth_reallocated=8/8, exact=True")


def test_default_path_remains_independent() -> None:
    """Verify omitting reuse state preserves independent model construction.

    Identical frame inputs are built twice without a state. Outputs must be
    distinct model objects, proving that the new interface is opt-in outside the
    video-loop wiring. Accidental global caching or singleton behavior fails.
    """
    frame = _frame(17, 0.0)
    first = _build(frame)
    second = _build(frame)
    assert first is not second
    print("default construction: independent_models=True")


def test_video_loop_owns_and_passes_one_reuse_state() -> None:
    """Verify the public frame API is opt-in and the video loop opts in once.

    The function signature must default ``gaussian_reuse_state`` to ``None``.
    Source-level wiring must construct one state before the video while-loop and
    pass that same name into ``process_frame``. A global default cache, missing
    pass-through, or per-iteration construction fails this architectural test.
    """
    parameter = inspect.signature(
        pipeline.process_frame).parameters["gaussian_reuse_state"]
    assert parameter.default is None
    source = inspect.getsource(pipeline.main)
    construction = "gaussian_reuse_state = GaussianFrameReuseState()"
    loop = "while processed < max_frames:"
    pass_through = "gaussian_reuse_state=gaussian_reuse_state"
    assert source.count(construction) == 1
    assert source.count(pass_through) == 1
    assert source.index(construction) < source.index(loop) < source.index(pass_through)
    print("video wiring: one_state_before_loop=True, explicit_pass_through=True")


def test_two_frame_process_path_reuses_storage() -> None:
    """Run two rendered frames through the public API with one reuse state.

    Inputs are a deterministic 64x36 RGB/depth pair, the production camera, and
    one caller-owned state. Both frames use strict CUDA pointcloud generation and
    minimal saves into a temporary directory. Outputs must have production panel
    dimensions, while the state must report one allocation followed by one reuse
    and no reallocation. Rendering, saving, plumbing, or state failures propagate.
    """
    height, width = 36, 64
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    rgb = np.stack([
        np.mod(cols * 7.0 + rows * 3.0, 256.0),
        np.mod(cols * 5.0 + rows * 11.0, 256.0),
        np.mod(cols * 13.0 + rows * 2.0, 256.0),
    ], axis=-1).astype(np.uint8)
    depth = (
        0.25 + 0.015 * cols + 0.01 * rows
        + 0.05 * np.sin(cols * 0.21)
    ).astype(np.float32)
    camera = pipeline.build_dual_camera()
    state = pipeline.GaussianFrameReuseState()
    with tempfile.TemporaryDirectory() as temporary:
        outputs = []
        provenance = None
        for frame_index in range(2):
            frame_dir = Path(temporary) / f"frame_{frame_index:04d}"
            frame_dir.mkdir()
            with redirect_stdout(StringIO()):
                outputs.append(pipeline.process_frame(
                    rgb,
                    None,
                    camera,
                    out_dir=str(frame_dir),
                    frame_idx=frame_index,
                    depth_map=depth,
                    edge_aware_gaussian_repair=False,
                    representation_contract_prototype=False,
                    surface_patch_level_prototype=False,
                    depth_structure_mode="off",
                    gaussian_representation="isotropic",
                    save_profile="minimal",
                    pointcloud_backend="cuda",
                    gaussian_reuse_state=state,
                ))
        provenance_path = (
            Path(temporary) / "frame_0001" / "parameter_provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_shape = (
        3,
        int(pipeline.PANEL_PIXEL_H),
        int(pipeline.PANEL_PIXEL_W),
    )
    for output in outputs:
        assert tuple(output["render_left"].shape) == expected_shape
        assert tuple(output["render_right"].shape) == expected_shape
    assert torch.equal(outputs[0]["render_left"], outputs[1]["render_left"])
    assert torch.equal(outputs[0]["render_right"], outputs[1]["render_right"])
    assert state.allocation_count == 1
    assert state.reuse_count == 1
    assert state.reallocation_count == 0
    reuse_metadata = provenance["gaussian_representation"]["frame_reuse"]
    assert reuse_metadata["enabled"] is True
    assert reuse_metadata["last_action"] == "reuse"
    assert reuse_metadata["reuse_count"] == 1
    print("two-frame public path: rendered=2, allocated=1, reused=1")


def benchmark_rebuild_vs_reuse() -> None:
    """Report repeated construction latency and peak live-allocation evidence.

    Ten pre-created distinct 131072-point CUDA frames avoid timing input
    allocation. Each path builds all frames with a CUDA synchronization around
    every measurement. Outputs are median milliseconds and peak-live-memory
    deltas; no speed assertion is made because allocator and device noise vary.
    CUDA/runtime errors propagate and fail the harness.
    """
    frames = [_frame(131072, 0.001 * index) for index in range(10)]

    def run(reuse: bool) -> tuple[float, float]:
        state = pipeline.GaussianFrameReuseState() if reuse else None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = int(torch.cuda.memory_allocated())
        samples = []
        model = None
        for frame in frames:
            torch.cuda.synchronize()
            started = time.perf_counter()
            model = _build(frame, state, sh_degree=pipeline.SH_DEGREE)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
        peak_delta_mib = (
            int(torch.cuda.max_memory_allocated()) - baseline) / (1024.0 ** 2)
        del model, state
        return statistics.median(samples[2:]), peak_delta_mib

    rebuild_ms, rebuild_peak = run(False)
    reuse_ms, reuse_peak = run(True)
    ratio = rebuild_ms / max(reuse_ms, 1.0e-9)
    print(
        "Gaussian construction benchmark: "
        f"rebuild_p50={rebuild_ms:.3f}ms, reuse_p50={reuse_ms:.3f}ms, "
        f"ratio={ratio:.3f}x, rebuild_peak={rebuild_peak:.3f}MiB, "
        f"reuse_peak={reuse_peak:.3f}MiB"
    )


def main() -> None:
    """Run five storage contracts followed by the non-gating microbenchmark."""
    test_compatible_frame_updates_in_place()
    test_shrink_reuses_capacity_and_growth_reallocates_safely()
    test_default_path_remains_independent()
    test_video_loop_owns_and_passes_one_reuse_state()
    test_two_frame_process_path_reuses_storage()
    benchmark_rebuild_vs_reuse()
    print("test_gaussian_frame_reuse: 5/5 passed + benchmark")


if __name__ == "__main__":
    main()
