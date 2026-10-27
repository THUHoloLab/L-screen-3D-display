"""Two-frame optical-flow-guided depth fusion.

The module is deliberately independent of DAV3 and the renderer.  Backward
flow maps current-frame pixels into the previous frame, confidence bounds how
much aligned history is accepted, and invalid samples fall back exactly to the
current raw depth.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


_GRID_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_MAX_CACHED_GRIDS = 8


def _grid_cache_key(reference: torch.Tensor) -> tuple:
    device_index = reference.device.index
    return (
        reference.device.type,
        -1 if device_index is None else int(device_index),
        reference.dtype,
        int(reference.shape[-2]),
        int(reference.shape[-1]),
    )


def _base_pixel_grid(reference: torch.Tensor) -> torch.Tensor:
    """Return cached ``(1,H,W,2)`` pixel coordinates on reference's device."""
    key = _grid_cache_key(reference)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        _GRID_CACHE.move_to_end(key)
        return cached

    height, width = reference.shape[-2:]
    y_coord, x_coord = torch.meshgrid(
        torch.arange(height, device=reference.device, dtype=reference.dtype),
        torch.arange(width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    grid = torch.stack((x_coord, y_coord), dim=-1).unsqueeze(0)
    _GRID_CACHE[key] = grid
    if len(_GRID_CACHE) > _MAX_CACHED_GRIDS:
        _GRID_CACHE.popitem(last=False)
    return grid


def _normalise_pixel_grid(
    pixel_grid: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    x_coord = pixel_grid[..., 0]
    y_coord = pixel_grid[..., 1]
    if width > 1:
        x_normalised = 2.0 * x_coord / float(width - 1) - 1.0
    else:
        x_normalised = torch.zeros_like(x_coord)
    if height > 1:
        y_normalised = 2.0 * y_coord / float(height - 1) - 1.0
    else:
        y_normalised = torch.zeros_like(y_coord)
    return torch.stack((x_normalised, y_normalised), dim=-1)


def _validate_inputs(
    current_depth: torch.Tensor,
    previous_depth: torch.Tensor,
    backward_flow: torch.Tensor,
    confidence: torch.Tensor,
) -> Tuple[int, int]:
    if current_depth.ndim != 2:
        raise ValueError(
            f"current_depth must have shape (H,W), got {tuple(current_depth.shape)}"
        )
    if previous_depth.shape != current_depth.shape:
        raise ValueError(
            "previous_depth must match current_depth, got "
            f"{tuple(previous_depth.shape)} and {tuple(current_depth.shape)}"
        )
    height, width = current_depth.shape
    if backward_flow.shape != (2, height, width):
        raise ValueError(
            "backward_flow must have shape (2,H,W), got "
            f"{tuple(backward_flow.shape)} for depth {(height, width)}"
        )
    if confidence.shape != (height, width):
        raise ValueError(
            f"confidence must have shape (H,W), got {tuple(confidence.shape)}"
        )
    tensors = (previous_depth, backward_flow, confidence)
    if any(tensor.device != current_depth.device for tensor in tensors):
        raise ValueError("all fusion tensors must be on the same device")
    if any(tensor.dtype != current_depth.dtype for tensor in tensors):
        raise ValueError("all fusion tensors must have the same dtype")
    return height, width


def warp_previous_depth(
    previous_depth: torch.Tensor,
    backward_flow: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp previous depth into the current frame.

    Parameters
    ----------
    previous_depth
        ``(H,W)`` previous raw depth.
    backward_flow
        ``(2,H,W)`` current-to-previous displacement in source pixels.

    Returns
    -------
    warped_depth, valid_mask
        Two ``(H,W)`` tensors. ``valid_mask`` rejects out-of-bounds and
        non-finite flow/depth samples.
    """
    if previous_depth.ndim != 2:
        raise ValueError(
            f"previous_depth must have shape (H,W), got {previous_depth.shape}"
        )
    height, width = previous_depth.shape
    if backward_flow.shape != (2, height, width):
        raise ValueError(
            "backward_flow must have shape (2,H,W), got "
            f"{tuple(backward_flow.shape)}"
        )
    if backward_flow.device != previous_depth.device:
        raise ValueError("previous_depth and backward_flow must share a device")
    if backward_flow.dtype != previous_depth.dtype:
        raise ValueError("previous_depth and backward_flow must share a dtype")

    base_grid = _base_pixel_grid(backward_flow)
    sample_pixels = base_grid + backward_flow.permute(1, 2, 0).unsqueeze(0)
    x_coord = sample_pixels[..., 0]
    y_coord = sample_pixels[..., 1]
    finite_flow = torch.isfinite(backward_flow).all(dim=0, keepdim=True)
    valid = (
        finite_flow
        & (x_coord >= 0.0)
        & (x_coord <= float(width - 1))
        & (y_coord >= 0.0)
        & (y_coord <= float(height - 1))
    )

    normalised_grid = _normalise_pixel_grid(sample_pixels, height, width)
    warped = F.grid_sample(
        previous_depth.unsqueeze(0).unsqueeze(0),
        normalised_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0]
    valid = valid[0] & torch.isfinite(warped)
    return warped, valid


def _fuse_depth_with_stats(
    current_depth: torch.Tensor,
    previous_depth: torch.Tensor,
    backward_flow: torch.Tensor,
    confidence: torch.Tensor,
    *,
    history_weight: float,
) -> torch.Tensor:
    """Internal single-warp fusion returning bounded confidence and validity."""
    _validate_inputs(current_depth, previous_depth, backward_flow, confidence)
    history_weight = float(history_weight)
    if not 0.0 <= history_weight <= 1.0:
        raise ValueError(
            f"history_weight must be in [0,1], got {history_weight}"
        )

    warped_previous, valid = warp_previous_depth(previous_depth, backward_flow)
    bounded_confidence = torch.nan_to_num(
        confidence, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp_(0.0, 1.0)
    valid = (
        valid
        & torch.isfinite(current_depth)
        & torch.isfinite(bounded_confidence)
    )
    effective_weight = bounded_confidence.mul(history_weight)
    effective_weight = torch.where(
        valid, effective_weight, torch.zeros_like(effective_weight)
    )
    output = torch.lerp(current_depth, warped_previous, effective_weight)
    return output, bounded_confidence, valid


def fuse_depth_with_confidence(
    current_depth: torch.Tensor,
    previous_depth: torch.Tensor,
    backward_flow: torch.Tensor,
    confidence: torch.Tensor,
    *,
    history_weight: float,
) -> torch.Tensor:
    """Fuse current raw depth with one aligned raw predecessor."""
    output, _, _ = _fuse_depth_with_stats(
        current_depth,
        previous_depth,
        backward_flow,
        confidence,
        history_weight=history_weight,
    )
    return output


class FlowGuidedDepthFusion:
    """Stateful depth fusion with an exact two-raw-frame support window."""

    def __init__(self, history_weight: float = 0.5):
        history_weight = float(history_weight)
        if not 0.0 <= history_weight <= 1.0:
            raise ValueError(
                f"history_weight must be in [0,1], got {history_weight}"
            )
        self._history_weight = history_weight
        self._previous_raw_depth: Optional[torch.Tensor] = None
        self._last_stats: Dict[str, float] = {
            "history_weight": history_weight,
            "mean_confidence": 0.0,
            "fallback_fraction": 1.0,
            "temporal_support_frames": 2.0,
        }

    def reset(self) -> None:
        self._previous_raw_depth = None
        self._last_stats = {
            "history_weight": self._history_weight,
            "mean_confidence": 0.0,
            "fallback_fraction": 1.0,
            "temporal_support_frames": 2.0,
        }

    def update(
        self,
        current_depth: torch.Tensor,
        *,
        backward_flow: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current_depth.ndim != 2:
            raise ValueError(
                f"current_depth must have shape (H,W), got {current_depth.shape}"
            )

        previous = self._previous_raw_depth
        can_fuse = (
            previous is not None
            and backward_flow is not None
            and confidence is not None
            and previous.shape == current_depth.shape
        )
        if can_fuse:
            output, bounded, valid = _fuse_depth_with_stats(
                current_depth,
                previous,
                backward_flow,
                confidence,
                history_weight=self._history_weight,
            )
            effective = bounded * valid.to(bounded.dtype)
            self._last_stats = {
                "history_weight": self._history_weight,
                "mean_confidence": float(bounded.mean().item()),
                "fallback_fraction": float((effective <= 0.0).float().mean().item()),
                "temporal_support_frames": 2.0,
            }
        else:
            output = current_depth
            self._last_stats = {
                "history_weight": self._history_weight,
                "mean_confidence": 0.0,
                "fallback_fraction": 1.0,
                "temporal_support_frames": 2.0,
            }

        # Store raw current depth, never the fused output. This makes influence
        # finite: frame t can affect only output t and output t+1.
        self._previous_raw_depth = current_depth.detach().clone()
        return output

    @property
    def history_weight(self) -> float:
        return self._history_weight

    @property
    def previous_raw_depth(self) -> Optional[torch.Tensor]:
        return self._previous_raw_depth

    @property
    def last_stats(self) -> Dict[str, float]:
        return dict(self._last_stats)
