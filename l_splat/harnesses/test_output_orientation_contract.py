"""Regression tests for the patent-aligned panel-row orientation contract."""

import numpy as np

from l_splat.core.output_orientation import (
    PATENT_RAW_ROW0_EDGE,
    SCREEN_BOTTOM,
    SCREEN_TOP,
    adapt_panel_row0_edge,
)


def _asymmetric_frame():
    """Return a frame whose top and bottom rows cannot be confused."""
    return np.array(
        [
            [[10, 11, 12], [20, 21, 22]],
            [[30, 31, 32], [40, 41, 42]],
            [[50, 51, 52], [60, 61, 62]],
        ],
        dtype=np.uint8,
    )


def test_patent_raw_output_keeps_screen_bottom_at_row_zero():
    frame = _asymmetric_frame()
    raw = adapt_panel_row0_edge(frame, SCREEN_BOTTOM)

    assert PATENT_RAW_ROW0_EDGE == SCREEN_BOTTOM
    np.testing.assert_array_equal(raw, frame)
    assert raw.flags.c_contiguous


def test_top_first_controller_adapter_applies_exactly_one_vertical_flip():
    frame = _asymmetric_frame()
    upright = adapt_panel_row0_edge(frame, SCREEN_TOP)

    np.testing.assert_array_equal(upright[0], frame[-1])
    np.testing.assert_array_equal(upright[-1], frame[0])
    assert upright.flags.c_contiguous


def test_unknown_row_zero_edge_fails_closed():
    frame = _asymmetric_frame()
    try:
        adapt_panel_row0_edge(frame, "viewer-dependent")
    except ValueError as exc:
        assert "target_row0_edge" in str(exc)
    else:
        raise AssertionError("Unknown row-zero edge must raise ValueError")


if __name__ == "__main__":
    test_patent_raw_output_keeps_screen_bottom_at_row_zero()
    test_top_first_controller_adapter_applies_exactly_one_vertical_flip()
    test_unknown_row_zero_edge_fails_closed()
    print("[PASS] Patent-aligned output orientation contract")
