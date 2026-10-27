"""
Task16C depth-structure refinement for the DAV3 -> 3DGS path.

This module intentionally does not touch DAV3 weights.  It treats DAV3 output
as an input signal, preserves RGB/depth hard edges, and returns explicit masks
that downstream Gaussian representation code can use without inventing seam
paint or cross-panel support strips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class DepthStructureResult:
    """Structured DAV3 post-process result for Task16C."""

    depth_raw: np.ndarray
    depth_refined: np.ndarray
    depth_norm: np.ndarray
    rgb_edge_mask: np.ndarray
    depth_edge_mask: np.ndarray
    hard_edge_mask: np.ndarray
    confidence_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _finite_float32(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(out)
    if finite.all():
        return out
    if finite.any():
        fill = float(np.median(out[finite]))
    else:
        fill = 0.0
    out = out.copy()
    out[~finite] = fill
    return out


def normalize_depth_percentile(
    depth: np.ndarray,
    mask: np.ndarray | None = None,
    p_lo: float = 3.0,
    p_hi: float = 97.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize depth to [0, 1] with robust percentile bounds."""
    d = _finite_float32(depth)
    valid = np.isfinite(d)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    samples = d[valid]
    if samples.size < 16:
        samples = d[np.isfinite(d)]
    if samples.size == 0:
        norm = np.zeros_like(d, dtype=np.float32)
        return norm, {
            "p_lo": float(p_lo),
            "p_hi": float(p_hi),
            "lo": 0.0,
            "hi": 1.0,
            "sample_count": 0,
        }
    lo = float(np.percentile(samples, p_lo))
    hi = float(np.percentile(samples, p_hi))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return norm, {
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "lo": lo,
        "hi": hi,
        "sample_count": int(samples.size),
    }


def compute_rgb_edge_mask(
    rgb: np.ndarray,
    canny_low: int = 32,
    canny_high: int = 96,
    sobel_percentile: float = 86.0,
    dilate_px: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return RGB hard-edge candidates from Sobel and Canny evidence."""
    rgb_u8 = np.asarray(rgb)
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    finite = grad[np.isfinite(grad)]
    thresh = float(np.percentile(finite, sobel_percentile)) if finite.size else 1.0
    thresh = max(thresh, 0.025)
    sobel_mask = grad >= thresh
    canny_mask = cv2.Canny(gray, int(canny_low), int(canny_high)) > 0
    mask = sobel_mask | canny_mask
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask.astype(bool), {
        "sobel_percentile": float(sobel_percentile),
        "sobel_threshold": thresh,
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
        "dilate_px": int(dilate_px),
        "edge_fraction": float(mask.mean()),
    }


def compute_depth_edge_mask(
    depth_norm: np.ndarray,
    percentile: float = 88.0,
    min_threshold: float = 0.018,
    dilate_px: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return depth discontinuity candidates in normalized-depth space."""
    d = _finite_float32(depth_norm)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    finite = grad[np.isfinite(grad)]
    thresh = float(np.percentile(finite, percentile)) if finite.size else min_threshold
    thresh = max(thresh, float(min_threshold))
    mask = grad >= thresh
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask.astype(bool), {
        "percentile": float(percentile),
        "threshold": thresh,
        "min_threshold": float(min_threshold),
        "dilate_px": int(dilate_px),
        "edge_fraction": float(mask.mean()),
    }


def joint_bilateral_or_fallback(
    rgb: np.ndarray,
    depth: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 0.025,
    sigma_space: float = 4.0,
) -> np.ndarray:
    """Apply OpenCV ximgproc JBF when available, else bilateral filtering."""
    guide = np.asarray(rgb, dtype=np.float32) * (1.0 / 255.0)
    d = _finite_float32(depth)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter"):
        return cv2.ximgproc.jointBilateralFilter(
            guide,
            d,
            d=int(diameter),
            sigmaColor=float(sigma_color),
            sigmaSpace=float(sigma_space),
        ).astype(np.float32)
    return cv2.bilateralFilter(
        d,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    ).astype(np.float32)


def refine_depth_structure(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    mode: str = "edge-preserve",
    fg_mask: np.ndarray | None = None,
    jbf_diameter: int = 7,
    jbf_sigma_color: float = 0.025,
    jbf_sigma_space: float = 4.0,
) -> DepthStructureResult:
    """
    Refine DAV3 depth with RGB/depth edge masks and robust normalization.

    ``mode="off"`` returns sanitized depth plus masks, preserving legacy behavior
    for callers that only need debug metadata. ``mode="edge-preserve"`` uses a
    guided filter for smooth regions while keeping raw depth at hard edges.
    """
    if mode not in {"off", "edge-preserve"}:
        raise ValueError(f"Unknown depth structure mode: {mode}")
    raw = _finite_float32(depth_raw)
    fg = None if fg_mask is None else np.asarray(fg_mask, dtype=bool)
    raw_norm, raw_norm_meta = normalize_depth_percentile(raw, mask=fg)

    rgb_edge, rgb_meta = compute_rgb_edge_mask(rgb)
    depth_edge, depth_meta = compute_depth_edge_mask(raw_norm)
    hard_edge = rgb_edge | depth_edge

    if mode == "edge-preserve":
        smooth = joint_bilateral_or_fallback(
            rgb,
            raw,
            diameter=jbf_diameter,
            sigma_color=jbf_sigma_color,
            sigma_space=jbf_sigma_space,
        )
        preserve = hard_edge
        if fg is not None:
            preserve = preserve | _mask_boundary(fg)
        refined = np.where(preserve, raw, smooth).astype(np.float32)
    else:
        smooth = raw
        refined = raw

    depth_norm, norm_meta = normalize_depth_percentile(refined, mask=fg)
    depth_edge_refined, refined_edge_meta = compute_depth_edge_mask(depth_norm)
    hard_edge = rgb_edge | depth_edge_refined
    if fg is not None:
        hard_edge = hard_edge | _mask_boundary(fg)

    confidence = (~hard_edge).astype(np.float32)
    if fg is not None:
        confidence = np.where(fg, confidence, confidence * 0.75).astype(np.float32)

    metadata = {
        "schema": "task16_depth_structure_result_v1",
        "mode": mode,
        "raw_normalization": raw_norm_meta,
        "normalization": norm_meta,
        "rgb_edges": rgb_meta,
        "depth_edges_raw": depth_meta,
        "depth_edges_refined": refined_edge_meta,
        "jbf": {
            "diameter": int(jbf_diameter),
            "sigma_color": float(jbf_sigma_color),
            "sigma_space": float(jbf_sigma_space),
            "used_ximgproc": bool(
                hasattr(cv2, "ximgproc")
                and hasattr(cv2.ximgproc, "jointBilateralFilter")
            ),
        },
        "hard_edge_fraction": float(hard_edge.mean()),
        "confidence_mean": float(confidence.mean()),
        "foreground_mask_used": bool(fg is not None),
    }
    return DepthStructureResult(
        depth_raw=raw,
        depth_refined=refined,
        depth_norm=depth_norm,
        rgb_edge_mask=rgb_edge.astype(bool),
        depth_edge_mask=depth_edge_refined.astype(bool),
        hard_edge_mask=hard_edge.astype(bool),
        confidence_mask=confidence.astype(np.float32),
        metadata=metadata,
    )


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(m, kernel)
    eroded = cv2.erode(m, kernel)
    return (dilated != eroded)


def depth_debug_images(result: DepthStructureResult) -> dict[str, np.ndarray]:
    """Build uint8 RGB/BW debug images for local provenance."""
    raw_norm, _ = normalize_depth_percentile(result.depth_raw)
    refined_norm, _ = normalize_depth_percentile(result.depth_refined)
    return {
        "task16_depth_raw_norm.png": (raw_norm * 255.0).astype(np.uint8),
        "task16_depth_refined_norm.png": (refined_norm * 255.0).astype(np.uint8),
        "task16_depth_norm.png": (result.depth_norm * 255.0).astype(np.uint8),
        "task16_rgb_edge_mask.png": (result.rgb_edge_mask.astype(np.uint8) * 255),
        "task16_depth_edge_mask.png": (result.depth_edge_mask.astype(np.uint8) * 255),
        "task16_hard_edge_mask.png": (result.hard_edge_mask.astype(np.uint8) * 255),
        "task16_confidence_mask.png": np.clip(
            result.confidence_mask * 255.0, 0, 255).astype(np.uint8),
    }
