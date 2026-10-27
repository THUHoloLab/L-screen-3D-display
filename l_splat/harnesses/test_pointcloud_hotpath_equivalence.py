"""Characterization gates for allocation-neutral pointcloud hot-path refactors."""

from __future__ import annotations

import contextlib
import hashlib
import io

import numpy as np

from l_splat.core import pipeline


EXPECTED = (
    ((37440, 3), "float32", "0aba7163bdd240d7d5bd34ac3fbe2f1a41c466ba8cacc9495b318bce48e00cf2"),
    ((37440, 3), "float32", "df8104c006883bd202a074c25e1591c4de24774ec77c88187c3a2d432259b8a9"),
    ((37440,), "int8", "9e9cbd23ebb684abdd4543ea12611662dc9982ac5d3b76154888f1b82fcb83ec"),
    ((37440,), "bool", "8cef3c9253228047f241f917395ffd6a49a26ac99e7d0175846b1ba50b6c4ca3"),
    ((37440,), "uint8", "04bb08fd08b30dedf6d2278441f5fb044eeaed18b5428aef8402f4ab919a8c63"),
    ((37440,), "bool", "fb8f53151b9e1b4642951a7b837426ce6b8e7c3df63122194f337d3fc639c9b0"),
    ((37440,), "bool", "fb8f53151b9e1b4642951a7b837426ce6b8e7c3df63122194f337d3fc639c9b0"),
)


def _synthetic_inputs() -> tuple[np.ndarray, np.ndarray]:
    height, width = 36, 64
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack([
        (xx * 7 + yy * 3) % 256,
        (xx * 5 + yy * 11) % 256,
        (xx * 13 + yy * 2) % 256,
    ], axis=-1).astype(np.uint8)
    depth = (
        0.7 * xx / max(width - 1, 1)
        + 0.3 * yy / max(height - 1, 1)
    ).astype(np.float32)
    return rgb, depth


def test_pointcloud_tuple_characterization() -> None:
    rgb, depth = _synthetic_inputs()
    with contextlib.redirect_stdout(io.StringIO()):
        result = pipeline.depth_to_unified_pointcloud(
            rgb,
            depth,
            pipeline.DEPTH_FWD_M,
            pipeline.DEPTH_BACK_M,
            pipeline.CONTENT_W_M,
            pipeline.PANEL_PHYS_H_M,
            return_edge_aware_mask=True,
        )
    assert len(result) == len(EXPECTED)
    for value, (shape, dtype, digest) in zip(result, EXPECTED):
        array = np.ascontiguousarray(value)
        assert array.shape == shape
        assert str(array.dtype) == dtype
        assert hashlib.sha256(array.tobytes()).hexdigest() == digest


def test_vectorized_rgb_gather_matches_channel_loop() -> None:
    rng = np.random.default_rng(20260727)
    rgb = rng.integers(0, 256, size=(37, 61, 3), dtype=np.uint8)
    rows = rng.integers(0, rgb.shape[0], size=4096, dtype=np.int32)
    cols = rng.integers(0, rgb.shape[1], size=4096, dtype=np.int32)
    rgb_float = rgb.astype(np.float32)
    reference = np.empty((rows.size, 3), dtype=np.float32)
    for channel in range(3):
        reference[:, channel] = rgb_float[rows, cols, channel]
    candidate = np.empty_like(reference)
    candidate[:] = rgb[rows, cols]
    assert np.array_equal(reference, candidate)


def main() -> None:
    tests = (
        test_pointcloud_tuple_characterization,
        test_vectorized_rgb_gather_matches_channel_loop,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"ALL {len(tests)} POINTCLOUD HOT-PATH TESTS PASSED")


if __name__ == "__main__":
    main()
