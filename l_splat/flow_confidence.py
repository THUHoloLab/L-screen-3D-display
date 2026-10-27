"""RAFT forward-backward confidence for video temporal stabilisation.

The estimator keeps optical flow and confidence on the selected Torch device
so the previous raw depth can be warped and fused before the one necessary
device-to-host transfer into the NumPy pointcloud pipeline.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FlowPair:
    """Bidirectional full-resolution flow and current-domain confidence."""

    forward: torch.Tensor
    backward: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    consistency_error: torch.Tensor
    working_resolution: Tuple[int, int]
    scaled_resolution: Tuple[int, int]
    timings_ms: Dict[str, float]


class FlowConfidenceEstimator:
    """TorchVision RAFT forward-backward consistency estimator."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        fb_threshold: float = 0.5,
        max_side: int = 1024,
        num_flow_updates: int = 12,
        model: Optional[torch.nn.Module] = None,
    ):
        model_size = str(model_size).strip().lower()
        if model_size not in ("small", "large"):
            raise ValueError(f"model_size must be 'small' or 'large', got {model_size!r}")
        if float(fb_threshold) <= 0.0:
            raise ValueError("fb_threshold must be > 0")
        if int(max_side) < 8:
            raise ValueError("max_side must be >= 8")
        if int(num_flow_updates) < 1:
            raise ValueError("num_flow_updates must be >= 1")

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self._device = requested_device
        self._model_size = model_size
        self._fb_threshold = float(fb_threshold)
        self._max_side = int(max_side)
        self._num_flow_updates = int(num_flow_updates)
        self._grid_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._sequence_previous_tensor: Optional[torch.Tensor] = None
        self._sequence_cache_key: Optional[tuple] = None

        if model is None:
            from torchvision.models.optical_flow import (
                Raft_Large_Weights,
                Raft_Small_Weights,
                raft_large,
                raft_small,
            )

            if model_size == "large":
                weights = Raft_Large_Weights.C_T_SKHT_V2
                model = raft_large(weights=weights)
            else:
                weights = Raft_Small_Weights.C_T_V2
                model = raft_small(weights=weights)

        self._model = model.to(self._device).eval()
        print(
            f"  [FlowConfidence] RAFT-{model_size} loaded on {self._device}, "
            f"max_side={self._max_side}, updates={self._num_flow_updates}, "
            f"fb_scale={self._fb_threshold:.2f}px"
        )

    @staticmethod
    def _weights_for_size(model_size: str):
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
        )

        if str(model_size).strip().lower() == "large":
            return Raft_Large_Weights.C_T_SKHT_V2
        return Raft_Small_Weights.C_T_V2

    @classmethod
    def checkpoint_path(cls, model_size: str = "small") -> str:
        weights = cls._weights_for_size(model_size)
        filename = os.path.basename(urlparse(weights.url).path)
        return os.path.join(torch.hub.get_dir(), "checkpoints", filename)

    @classmethod
    def checkpoint_is_cached(cls, model_size: str = "small") -> bool:
        return os.path.isfile(cls.checkpoint_path(model_size))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def max_side(self) -> int:
        return self._max_side

    def _sync(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _img_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert RGB uint8 HWC to contiguous NCHW float in ``[-1,1]``."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image)!r}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (H,W,3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"image must have dtype uint8, got {image.dtype}")
        contiguous = np.ascontiguousarray(image)
        tensor = torch.from_numpy(contiguous).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=torch.float32)
        tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return tensor.contiguous()

    @staticmethod
    def _scaled_size(height: int, width: int, max_side: int) -> Tuple[int, int]:
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    @staticmethod
    def _pad_to_multiple_of_eight(tensor: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        height, width = tensor.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h == 0 and pad_w == 0:
            return tensor, 0, 0
        mode = "reflect"
        if height <= pad_h or width <= pad_w:
            mode = "replicate"
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode), pad_h, pad_w

    def _prepare_pair(
        self,
        frame_previous: np.ndarray,
        frame_current: np.ndarray,
        *,
        reuse_previous: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        if frame_previous.shape != frame_current.shape:
            raise ValueError(
                "flow frames must have identical shapes, got "
                f"{frame_previous.shape} and {frame_current.shape}"
            )
        height, width = frame_previous.shape[:2]
        scaled_h, scaled_w = self._scaled_size(height, width, self._max_side)
        current = self._img_to_tensor(frame_current)
        cache_key = (
            tuple(frame_previous.shape),
            frame_previous.dtype.str,
            scaled_h,
            scaled_w,
            self._device.type,
            self._device.index,
        )
        if (
            reuse_previous
            and self._sequence_previous_tensor is not None
            and self._sequence_cache_key == cache_key
        ):
            previous = self._sequence_previous_tensor
        else:
            previous = self._img_to_tensor(frame_previous)
        if (scaled_h, scaled_w) != (height, width):
            if not (
                reuse_previous
                and self._sequence_previous_tensor is previous
            ):
                previous = F.interpolate(
                    previous, size=(scaled_h, scaled_w), mode="bilinear",
                    align_corners=False,
                )
            current = F.interpolate(
                current, size=(scaled_h, scaled_w), mode="bilinear",
                align_corners=False,
            )
        if not (
            reuse_previous
            and self._sequence_previous_tensor is previous
        ):
            previous, pad_h, pad_w = self._pad_to_multiple_of_eight(previous)
        else:
            pad_h = previous.shape[-2] - scaled_h
            pad_w = previous.shape[-1] - scaled_w
        current, _, _ = self._pad_to_multiple_of_eight(current)
        working_size = (scaled_h + pad_h, scaled_w + pad_w)
        if reuse_previous:
            self._sequence_previous_tensor = current
            self._sequence_cache_key = cache_key
        else:
            self.reset_sequence()
        return previous, current, (scaled_h, scaled_w), working_size

    def reset_sequence(self) -> None:
        """Drop the optional sequential-frame preprocessing cache."""
        self._sequence_previous_tensor = None
        self._sequence_cache_key = None

    @staticmethod
    def _resize_flow(
        flow: torch.Tensor,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        source_h, source_w = source_size
        target_h, target_w = target_size
        if source_size == target_size:
            return flow
        resized = F.interpolate(
            flow, size=target_size, mode="bilinear", align_corners=False
        )
        resized[:, 0].mul_(float(target_w) / float(source_w))
        resized[:, 1].mul_(float(target_h) / float(source_h))
        return resized

    def _base_pixel_grid(self, reference: torch.Tensor) -> torch.Tensor:
        height, width = reference.shape[-2:]
        device_index = reference.device.index
        key = (
            reference.device.type,
            -1 if device_index is None else int(device_index),
            reference.dtype,
            int(height),
            int(width),
        )
        cached = self._grid_cache.get(key)
        if cached is not None:
            self._grid_cache.move_to_end(key)
            return cached
        y_coord, x_coord = torch.meshgrid(
            torch.arange(height, device=reference.device, dtype=reference.dtype),
            torch.arange(width, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        grid = torch.stack((x_coord, y_coord), dim=-1).unsqueeze(0)
        self._grid_cache[key] = grid
        if len(self._grid_cache) > 8:
            self._grid_cache.popitem(last=False)
        return grid

    def _warp_with_valid(
        self, tensor_to_warp: torch.Tensor, flow_for_warping: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = flow_for_warping.shape
        base = self._base_pixel_grid(flow_for_warping)
        sample_pixels = base + flow_for_warping.permute(0, 2, 3, 1)
        x_coord = sample_pixels[..., 0]
        y_coord = sample_pixels[..., 1]
        valid = (
            torch.isfinite(flow_for_warping).all(dim=1)
            & (x_coord >= 0.0)
            & (x_coord <= float(width - 1))
            & (y_coord >= 0.0)
            & (y_coord <= float(height - 1))
        )
        if width > 1:
            grid_x = 2.0 * x_coord / float(width - 1) - 1.0
        else:
            grid_x = torch.zeros_like(x_coord)
        if height > 1:
            grid_y = 2.0 * y_coord / float(height - 1) - 1.0
        else:
            grid_y = torch.zeros_like(y_coord)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        warped = F.grid_sample(
            tensor_to_warp,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        valid &= torch.isfinite(warped).all(dim=1)
        return warped, valid

    def _current_domain_confidence(
        self, forward: torch.Tensor, backward: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forward_at_previous, valid = self._warp_with_valid(forward, backward)
        error = torch.abs(backward + forward_at_previous).sum(dim=1)
        finite = torch.isfinite(error)
        valid &= finite
        scale = max(self._fb_threshold, torch.finfo(error.dtype).eps)
        confidence = torch.exp(-torch.nan_to_num(error, nan=float("inf")) / scale)
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        return confidence.clamp_(0.0, 1.0), valid, error

    def estimate_pair(
        self,
        frame_previous: np.ndarray,
        frame_current: np.ndarray,
        *,
        max_side: Optional[int] = None,
        num_flow_updates: Optional[int] = None,
        reuse_previous: bool = False,
    ) -> FlowPair:
        """Estimate full-resolution bidirectional flow and current confidence."""
        original_max_side = self._max_side
        if max_side is not None:
            if int(max_side) < 8:
                raise ValueError("max_side must be >= 8")
            self._max_side = int(max_side)
        updates = self._num_flow_updates
        if num_flow_updates is not None:
            updates = int(num_flow_updates)
            if updates < 1:
                raise ValueError("num_flow_updates must be >= 1")

        total_start = time.perf_counter()
        preprocess_start = total_start
        try:
            previous, current, scaled_size, working_size = self._prepare_pair(
                frame_previous,
                frame_current,
                reuse_previous=reuse_previous,
            )
        finally:
            self._max_side = original_max_side
        self._sync()
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        raft_start = time.perf_counter()
        with torch.inference_mode():
            forward_work = self._model(
                previous, current, num_flow_updates=updates
            )[-1]
            backward_work = self._model(
                current, previous, num_flow_updates=updates
            )[-1]
        self._sync()
        raft_ms = (time.perf_counter() - raft_start) * 1000.0

        post_start = time.perf_counter()
        scaled_h, scaled_w = scaled_size
        forward_work = forward_work[..., :scaled_h, :scaled_w]
        backward_work = backward_work[..., :scaled_h, :scaled_w]
        target_size = tuple(int(v) for v in frame_previous.shape[:2])
        forward = self._resize_flow(forward_work, scaled_size, target_size)
        backward = self._resize_flow(backward_work, scaled_size, target_size)
        confidence, valid, error = self._current_domain_confidence(
            forward, backward
        )
        self._sync()
        postprocess_ms = (time.perf_counter() - post_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return FlowPair(
            forward=forward,
            backward=backward,
            confidence=confidence,
            valid=valid,
            consistency_error=error,
            working_resolution=working_size,
            scaled_resolution=scaled_size,
            timings_ms={
                "preprocess_ms": preprocess_ms,
                "raft_bidirectional_ms": raft_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": total_ms,
            },
        )

    def compute_mask(
        self, frame_t: np.ndarray, frame_t_plus_1: np.ndarray
    ) -> np.ndarray:
        """Compatibility API returning the legacy binary NumPy mask."""
        pair = self.estimate_pair(frame_t, frame_t_plus_1)
        mask = (
            (pair.consistency_error < self._fb_threshold) & pair.valid
        ).to(torch.float32)
        return mask[0].cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _fb_consistency(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor) -> torch.Tensor:
        """Legacy previous-domain forward-backward error helper."""
        flow_bwd_warped = FlowConfidenceEstimator._warp_flow(flow_bwd, flow_fwd)
        return torch.abs(flow_fwd + flow_bwd_warped).sum(dim=1)

    @staticmethod
    def _warp_flow(
        flow_to_warp: torch.Tensor, flow_for_warping: torch.Tensor
    ) -> torch.Tensor:
        """Compatibility helper used by older isolated harnesses."""
        _, _, height, width = flow_for_warping.shape
        y_coord, x_coord = torch.meshgrid(
            torch.arange(
                height,
                device=flow_for_warping.device,
                dtype=flow_for_warping.dtype,
            ),
            torch.arange(
                width,
                device=flow_for_warping.device,
                dtype=flow_for_warping.dtype,
            ),
            indexing="ij",
        )
        x_sample = x_coord + flow_for_warping[:, 0]
        y_sample = y_coord + flow_for_warping[:, 1]
        if width > 1:
            grid_x = 2.0 * x_sample / float(width - 1) - 1.0
        else:
            grid_x = torch.zeros_like(x_sample)
        if height > 1:
            grid_y = 2.0 * y_sample / float(height - 1) - 1.0
        else:
            grid_y = torch.zeros_like(y_sample)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(
            flow_to_warp,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
