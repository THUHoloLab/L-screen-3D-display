"""Panel-row orientation adapters for patent-aligned L-screen outputs.

The raster contract uses row zero for the physical screen bottom.  That raw
orientation is preserved in canonical panel files.  A controller or human-view
preview that expects row zero at the screen top must request one explicit
vertical adaptation here.
"""

from __future__ import annotations

import numpy as np


SCREEN_BOTTOM = "screen-bottom"
SCREEN_TOP = "screen-top"
PATENT_RAW_ROW0_EDGE = SCREEN_BOTTOM


def adapt_panel_row0_edge(image, target_row0_edge):
    """Return ``image`` with row zero mapped to the requested physical edge.

    Args:
        image: Array-like image with row and column as its first two axes.
        target_row0_edge: ``"screen-bottom"`` for the patent raw-raster
            convention or ``"screen-top"`` for a conventional upright image.

    The input is always interpreted as patent raw-raster order.  The returned
    array is C-contiguous and the input is never modified.
    """
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError(f"image must have at least two dimensions, got {array.shape}")
    if target_row0_edge == SCREEN_BOTTOM:
        return np.ascontiguousarray(array)
    if target_row0_edge == SCREEN_TOP:
        return np.ascontiguousarray(array[::-1, ...])
    raise ValueError(
        "target_row0_edge must be 'screen-bottom' or 'screen-top', "
        f"got {target_row0_edge!r}"
    )
