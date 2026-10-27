"""
test_fold_frequency.py -- Spatial frequency preservation analysis at the fold.

Counts black-white transitions row-by-row in synthetic checkerboard input
and rendered output. Computes Transition Inflation Ratio (TIR) near the
fold region vs. panel interior. Diagnoses whether the L-screen pipeline
destroys or creates spatial frequencies at the 90-degree fold junction.

STRICTLY READ-ONLY. No production code modified.

Usage:
    conda activate env_5080
    PYTHONPATH=. python l_splat/harnesses/test_fold_frequency.py

Outputs (in output/forensics/fold_frequency/):
    fold_frequency_report.md   -- structured diagnosis
    transitions_vs_row.png     -- per-row transition count plot
    tir_vs_row.png             -- Transition Inflation Ratio plot
    transition_metrics.csv     -- per-row data
    frequency_metrics.json     -- machine-readable summary
    s0_input_checkerboard.png  -- synthetic input
    s6_render_left.png         -- left panel render
    s6_render_right.png        -- right panel render
    s6_render_stitched.png     -- stitched render
"""

import csv
import json
import os
import sys
import time

import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GS_DIR = os.path.join(_PROJ_ROOT, "gaussian-splatting-main")
sys.path.insert(0, GS_DIR)
sys.path.insert(0, os.path.join(GS_DIR, "submodules", "diff-gaussian-rasterization"))
sys.path.insert(0, os.path.join(GS_DIR, "submodules", "simple-knn"))

# Mock simple_knn._C
_simple_knn = type("simple_knn", (), {})()
_simple_knn._C = type("simple_knn._C", (), {})()
_simple_knn._C.distCUDA2 = lambda x: __import__("torch").zeros(0)
__import__("sys").modules["simple_knn"] = _simple_knn
__import__("sys").modules["simple_knn._C"] = _simple_knn._C

import torch
import cv2
from arguments import GroupParams

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from l_splat.core.pipeline import (
    depth_to_unified_pointcloud,
    create_gaussian_model,
    build_dual_camera,
    render_dual,
    DEVICE,
    BG_COLOR,
    PANEL_PIXEL_W,
    PANEL_PIXEL_H,
    PANEL_PHYS_W_M,
    PANEL_PHYS_H_M,
    CONTENT_W_M,
    DEPTH_FWD_M,
    DEPTH_BACK_M,
    GAUSSIAN_OVERLAP_FACTOR,
    GAUSSIAN_OPACITY_SIGMOID,
)

OUT_DIR = os.path.join(_PROJ_ROOT, "output", "forensics", "fold_frequency")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Analysis constants ---
TRANSITION_THRESHOLD = 30       # min intensity jump for a real transition
FOLD_PROXIMITY_MM = 0.05        # |X| < 50mm = fold-adjacent
TIR_LOSS_THRESHOLD = 0.5        # TIR below this = severe edge loss
TIR_INFLATION_THRESHOLD = 2.0   # TIR above this = severe spurious edges


# =============================================================================
# Synthetic Input Generators
# =============================================================================

def _make_checkerboard(H, W, squares=16):
    y_idx = np.arange(H)[:, None]
    x_idx = np.arange(W)[None, :]
    pattern = ((y_idx // (H // squares)) + (x_idx // (W // squares))) % 2
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[pattern == 0] = [255, 255, 255]
    rgb[pattern == 1] = [0, 0, 0]
    return rgb


def _make_constant_depth(H, W, value=0.5):
    return np.full((H, W), value, dtype=np.float32)


def _make_ramp_depth(H, W):
    return np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))


# =============================================================================
# Visualization Helpers
# =============================================================================

def _tensor_to_uint8(t):
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _save_rgb(img_u8, path):
    cv2.imwrite(path, cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))


# =============================================================================
# Transition Analysis
# =============================================================================

def to_gray(img):
    """Convert (H,W,3) uint8 [0,255] to (H,W) float64 grayscale."""
    if img.ndim == 3:
        return (img[:, :, 0].astype(np.float64) * 0.299 +
                img[:, :, 1].astype(np.float64) * 0.587 +
                img[:, :, 2].astype(np.float64) * 0.114)
    return img.astype(np.float64)


def count_transitions_per_row(img_gray):
    """Count black-white transitions per row via binary crossing.

    Binarizes the row at its midpoint value, then counts how many times
    the binary signal flips from 0→1 or 1→0 along the row.  This is
    robust for synthetic checkerboards where edges are exactly 1px.

    Args:
        img_gray: (H, W) float64 grayscale image.

    Returns:
        transitions: (H,) int32 array, transition count per row.
    """
    H, W = img_gray.shape
    transitions = np.zeros(H, dtype=np.int32)

    for y in range(H):
        row = img_gray[y]
        mid = (row.min() + row.max()) / 2.0
        if row.max() - row.min() < 20:
            # Nearly uniform row — no transitions
            transitions[y] = 0
            continue
        binary = (row > mid).astype(np.int8)
        flips = np.diff(binary) != 0
        transitions[y] = np.sum(flips)

    return transitions


def resample_transitions(trans, target_height):
    """Resample transition counts from source height to target height.

    Uses nearest-neighbor to preserve integer transition counts per row.
    Each source row's count is repeated for its proportional share of
    target rows, then total transitions are preserved.

    Args:
        trans: (H_src,) int32 transition counts.
        target_height: int

    Returns:
        (target_height,) float64 — resampled transition counts per row.
    """
    src_h = len(trans)
    if src_h == target_height:
        return trans.astype(np.float64)

    # Map each target row to the nearest source row
    src_indices = (np.arange(target_height) * src_h / target_height).astype(np.int32)
    src_indices = np.clip(src_indices, 0, src_h - 1)
    return trans[src_indices].astype(np.float64)


def compute_transition_ratio(out_transitions, in_transitions_resampled):
    """Compute output/input transition ratio, safe for zero input.

    Args:
        out_transitions: (H,) int32 — output transitions at render height.
        in_transitions_resampled: (H,) float64 — input transitions resampled
                                  to same height.

    Returns:
        (H,) float64 array.  Rows with 0 input transitions get NaN.
    """
    in_safe = in_transitions_resampled.copy()
    in_safe[in_safe < 0.5] = np.nan  # treat <0.5 as effectively zero
    return out_transitions.astype(np.float64) / in_safe


# =============================================================================
# Region Identification
# =============================================================================

def identify_fold_rows(points_world, panel_owner, H_img, fold_threshold_m=FOLD_PROXIMITY_MM):
    """Identify which render rows correspond to fold-adjacent world-space points.

    Fold rows: any point with panel_owner==2 (blend zone) maps to this row.
    Interior rows: panel_owner 0 or 1 AND far from the fold in the relevant axis.

    For left-panel points (panel_owner==0), "far from fold" means |X| > threshold
    (X is the distance from the fold line). Y is ~0 for all left-panel points.

    For right-panel points (panel_owner==1), "far from fold" means |Y| > threshold
    (Y is the distance from the fold line). X is ~0 for all right-panel points.

    Args:
        points_world: (N, 3) world-space coordinates.
        panel_owner: (N,) int -- 0=left, 1=right, 2=fold blend.
        H_img: int -- render image height (PANEL_PIXEL_H = 936).

    Returns:
        fold_mask: (H_img,) bool -- True for fold-affected rows.
        interior_mask: (H_img,) bool -- True for interior rows.
    """
    fold_mask = np.zeros(H_img, dtype=bool)
    interior_mask = np.zeros(H_img, dtype=bool)

    content_h = PANEL_PHYS_H_M

    def world_z_to_row(z):
        """Map world Z (0 = bottom of panel, content_h = top) to render row."""
        z_clipped = np.clip(z, 0, content_h)
        row = ((1.0 - z_clipped / content_h) * H_img).astype(np.int32)
        return np.clip(row, 0, H_img - 1)

    # Fold rows: rows that contain any blend-zone (panel_owner==2) point
    fold_pts = points_world[panel_owner == 2]
    if len(fold_pts) > 0:
        fold_rows = world_z_to_row(fold_pts[:, 2])
        fold_mask[fold_rows] = True

    # Interior rows: rows with panel_owner==0 or 1 AND far from fold
    for owner, dist_axis in [(0, 0), (1, 1)]:  # 0: left (distance = |X|), 1: right (distance = |Y|)
        mask = panel_owner == owner
        if mask.sum() == 0:
            continue
        pts = points_world[mask]
        dist_from_fold = np.abs(pts[:, dist_axis])
        far = dist_from_fold > fold_threshold_m * 2  # >0.10m from fold
        if far.sum() > 0:
            far_rows = world_z_to_row(pts[far, 2])
            for r in np.unique(far_rows):
                interior_mask[r] = True

    # Ensure interior doesn't overlap fold
    interior_mask = interior_mask & ~fold_mask

    # Fallback: if no interior rows found, mark top/bottom 30% as interior
    if not interior_mask.any():
        top_third = H_img * 7 // 10  # top 30%
        bot_third = H_img * 3 // 10  # bottom 30%
        interior_mask[:bot_third] = True
        interior_mask[top_third:] = True
        interior_mask = interior_mask & ~fold_mask

    return fold_mask, interior_mask


# =============================================================================
# Visualization
# =============================================================================

def plot_transitions_vs_row(in_left_trans, in_right_trans,
                             out_left_trans, out_right_trans,
                             fold_mask, out_path):
    """4-panel figure: transition count vs row index.

    Panel 1: Input left half
    Panel 2: Input right half
    Panel 3: Left panel render
    Panel 4: Right panel render
    Fold-affected rows are shaded red in all panels.
    """
    H = len(in_left_trans)
    rows = np.arange(H)
    fold_alpha = 0.15

    fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)

    # Panel 1: Input left half
    ax = axes[0]
    ax.plot(in_left_trans, rows, 'k-', linewidth=0.8, label='Input L')
    if fold_mask.any():
        ax.fill_betweenx(rows, 0, max(in_left_trans.max(), 1) * 1.1,
                          where=fold_mask, color='red', alpha=fold_alpha, label='Fold zone')
    ax.set_xlabel('Transitions per row')
    ax.set_ylabel('Row index (y)')
    ax.set_title('Input Left Half')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=7)

    # Panel 2: Input right half
    ax = axes[1]
    ax.plot(in_right_trans, rows, 'k-', linewidth=0.8, label='Input R')
    if fold_mask.any():
        ax.fill_betweenx(rows, 0, max(in_right_trans.max(), 1) * 1.1,
                          where=fold_mask, color='red', alpha=fold_alpha, label='Fold zone')
    ax.set_xlabel('Transitions per row')
    ax.set_title('Input Right Half')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=7)

    # Panel 3: Left render
    ax = axes[2]
    ax.plot(out_left_trans, rows, 'b-', linewidth=0.8, label='Left output')
    if fold_mask.any():
        ax.fill_betweenx(rows, 0, max(out_left_trans.max(), 1) * 1.1,
                          where=fold_mask, color='red', alpha=fold_alpha, label='Fold zone')
    ax.set_xlabel('Transitions per row')
    ax.set_title('Left Panel Render')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=7)

    # Panel 4: Right render
    ax = axes[3]
    ax.plot(out_right_trans, rows, 'r-', linewidth=0.8, label='Right output')
    if fold_mask.any():
        ax.fill_betweenx(rows, 0, max(out_right_trans.max(), 1) * 1.1,
                          where=fold_mask, color='red', alpha=fold_alpha, label='Fold zone')
    ax.set_xlabel('Transitions per row')
    ax.set_title('Right Panel Render')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=7)

    fig.suptitle('Row-by-Row Black-White Transition Count (per-panel comparison)', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_tir_vs_row(tir_left, tir_right, fold_mask, out_path):
    """Transition Inflation Ratio per row with fold zone shaded."""
    H = len(tir_left)
    rows = np.arange(H)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    for ax, tir, title, color in [
        (axes[0], tir_left, 'TIR Left Panel', 'blue'),
        (axes[1], tir_right, 'TIR Right Panel', 'red'),
    ]:
        valid = ~np.isnan(tir)
        tir_clipped = np.clip(tir, 0, 5)  # cap extreme outliers for visualization

        ax.plot(tir_clipped[valid], rows[valid], '-', color=color, linewidth=0.8)
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1.0, label='TIR=1 (preserved)')
        ax.axvline(x=TIR_LOSS_THRESHOLD, color='orange', linestyle=':', linewidth=1.0,
                    label=f'TIR={TIR_LOSS_THRESHOLD} (loss)')
        ax.axvline(x=TIR_INFLATION_THRESHOLD, color='purple', linestyle=':', linewidth=1.0,
                    label=f'TIR={TIR_INFLATION_THRESHOLD} (inflation)')

        if fold_mask.any():
            ax.fill_betweenx(rows, 0, 5, where=fold_mask,
                              color='red', alpha=0.12, label='Fold zone')

        ax.set_xlabel('Transition Inflation Ratio (output/input)')
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=6)

    fig.suptitle('Transition Inflation Ratio (TIR) by Row', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_region_metrics(tir, fold_mask, interior_mask):
    """Compute TIR statistics for a single panel's TIR array.

    Args:
        tir: (H,) float64 -- TIR per row (may contain NaN).
        fold_mask: (H,) bool
        interior_mask: (H,) bool

    Returns:
        dict with keys: TIR_fold_mean, TIR_fold_median, TIR_interior_mean,
        TIR_interior_median, fold_excess_pct, loss_count, inflation_count,
        n_fold_rows, n_interior_rows.
    """
    tir_fold = tir[fold_mask]
    tir_fold_valid = tir_fold[~np.isnan(tir_fold)]
    tir_int = tir[interior_mask]
    tir_int_valid = tir_int[~np.isnan(tir_int)]

    tir_fold_mean = float(np.mean(tir_fold_valid)) if len(tir_fold_valid) > 0 else np.nan
    tir_fold_median = float(np.median(tir_fold_valid)) if len(tir_fold_valid) > 0 else np.nan
    tir_int_mean = float(np.mean(tir_int_valid)) if len(tir_int_valid) > 0 else np.nan
    tir_int_median = float(np.median(tir_int_valid)) if len(tir_int_valid) > 0 else np.nan

    # Fold excess: how much worse the fold is vs interior
    if not np.isnan(tir_int_mean) and tir_int_mean > 0:
        fold_excess_pct = round((tir_fold_mean - tir_int_mean) / tir_int_mean * 100, 1)
    else:
        fold_excess_pct = np.nan

    loss_count = int(np.sum(tir_fold_valid < TIR_LOSS_THRESHOLD))
    inflation_count = int(np.sum(tir_fold_valid > TIR_INFLATION_THRESHOLD))
    n_fold_rows = int(np.sum(fold_mask))
    n_interior_rows = int(np.sum(interior_mask))

    return {
        "TIR_fold_mean": round(tir_fold_mean, 3) if not np.isnan(tir_fold_mean) else "N/A",
        "TIR_fold_median": round(tir_fold_median, 3) if not np.isnan(tir_fold_median) else "N/A",
        "TIR_interior_mean": round(tir_int_mean, 3) if not np.isnan(tir_int_mean) else "N/A",
        "TIR_interior_median": round(tir_int_median, 3) if not np.isnan(tir_int_median) else "N/A",
        "fold_excess_pct": fold_excess_pct if not np.isnan(fold_excess_pct) else "N/A",
        "loss_count": loss_count,
        "inflation_count": inflation_count,
        "n_fold_rows": n_fold_rows,
        "n_interior_rows": n_interior_rows,
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_frequency_report(metrics_left, metrics_right, config, out_dir):
    """Write fold_frequency_report.md with structured diagnosis."""
    path = os.path.join(out_dir, "fold_frequency_report.md")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def diagnosis(m):
        """Generate a one-line diagnosis for a panel's metrics."""
        parts = []
        tir_fold = m.get("TIR_fold_mean", "N/A")
        tir_int = m.get("TIR_interior_mean", "N/A")

        try:
            tir_f = float(tir_fold)
        except (ValueError, TypeError):
            tir_f = None

        try:
            tir_i = float(tir_int)
        except (ValueError, TypeError):
            tir_i = None

        if tir_f is not None:
            if tir_f > 2.0:
                parts.append("SEVERE inflation")
            elif tir_f > 1.3:
                parts.append("MODERATE inflation")
            elif tir_f > 1.1:
                parts.append("MILD inflation")
            elif tir_f > 0.9:
                parts.append("Frequency PRESERVED")
            elif tir_f > 0.6:
                parts.append("MILD loss")
            elif tir_f > 0.3:
                parts.append("MODERATE loss")
            else:
                parts.append("SEVERE loss")

        if tir_i is not None and tir_f is not None:
            diff = tir_f - tir_i
            if abs(diff) > 0.3:
                parts.append(f"(fold-interior delta={diff:+.2f})")

        loss = m.get("loss_count", 0)
        infl = m.get("inflation_count", 0)
        if loss > 0:
            parts.append(f"{loss} rows with severe loss")
        if infl > 0:
            parts.append(f"{infl} rows with severe inflation")

        return "; ".join(parts) if parts else "No data"

    content = f"""# Fold Frequency Analysis Report

**Generated by** `test_fold_frequency.py`
**Resolution**: {config['resolution']}x{config['resolution']}
**Squares**: {config['squares']}x{config['squares']}
**Depth**: {config['depth_type']}
**Transition threshold**: {TRANSITION_THRESHOLD}
**Fold proximity**: {FOLD_PROXIMITY_MM * 1000:.0f} mm

---

## Summary

| Metric | Left Panel | Right Panel |
|--------|-----------|-------------|
| TIR fold (mean) | {fmt(metrics_left['TIR_fold_mean'])} | {fmt(metrics_right['TIR_fold_mean'])} |
| TIR fold (median) | {fmt(metrics_left['TIR_fold_median'])} | {fmt(metrics_right['TIR_fold_median'])} |
| TIR interior (mean) | {fmt(metrics_left['TIR_interior_mean'])} | {fmt(metrics_right['TIR_interior_mean'])} |
| TIR interior (median) | {fmt(metrics_left['TIR_interior_median'])} | {fmt(metrics_right['TIR_interior_median'])} |
| Fold excess % | {fmt(metrics_left['fold_excess_pct'])} | {fmt(metrics_right['fold_excess_pct'])} |
| Loss rows (TIR<{TIR_LOSS_THRESHOLD}) | {metrics_left['loss_count']} | {metrics_right['loss_count']} |
| Inflation rows (TIR>{TIR_INFLATION_THRESHOLD}) | {metrics_left['inflation_count']} | {metrics_right['inflation_count']} |
| Fold rows total | {metrics_left['n_fold_rows']} | {metrics_right['n_fold_rows']} |
| Interior rows total | {metrics_left['n_interior_rows']} | {metrics_right['n_interior_rows']} |

---

## Diagnosis

### Left Panel

**{diagnosis(metrics_left)}**

### Right Panel

**{diagnosis(metrics_right)}**

---

## Interpretation Guide

- **TIR ~ 1.0**: Spatial frequency perfectly preserved (ideal).
- **TIR > 1.0**: Transition *inflation* -- pipeline introduces spurious edges.
  Near fold: likely from geometry distortion, interpolation artifacts, or
  overlapping Gaussian projections creating false boundaries.
- **TIR < 1.0**: Transition *loss* -- pipeline blurs/destroys real edges.
  Near fold: likely from Gaussian overlap smoothing, insufficient
  scale resolution, or blend-zone averaging.
- **Fold excess > 0**: The fold region is quantifiably worse than interior
  panels at preserving spatial frequency.
- **Fold excess ~ 0**: Fold and panels perform similarly (good).
- **Fold excess < 0**: Fold performs *better* than panels (unexpected -- may
  indicate interior has its own problems).

---

## Output Files

| File | Description |
|------|-------------|
| `transitions_vs_row.png` | Per-row transition count (input + left + right) |
| `tir_vs_row.png` | TIR by row with fold zone shaded |
| `transition_metrics.csv` | Full per-row data |
| `frequency_metrics.json` | Machine-readable summary |
| `s6_render_left.png` | Left panel render (832x936) |
| `s6_render_right.png` | Right panel render (832x936) |

---

*Generated by test_fold_frequency.py -- spatial frequency preservation only.*
*No gradient, blur, or spectral metrics were computed.*
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Wrote {path}")
    return path


def write_transition_csv(tir_left, tir_right,
                          in_left_trans, in_right_trans,
                          out_left_trans, out_right_trans,
                          fold_mask, out_dir):
    """Write per-row transition data to CSV."""
    path = os.path.join(out_dir, "transition_metrics.csv")
    H = len(in_left_trans)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_idx", "in_left_trans", "in_right_trans",
                     "out_left_trans", "out_right_trans",
                     "TIR_left", "TIR_right", "is_fold"])
        for y in range(H):
            w.writerow([
                y,
                int(in_left_trans[y]),
                int(in_right_trans[y]),
                int(out_left_trans[y]),
                int(out_right_trans[y]),
                round(float(tir_left[y]), 3) if not np.isnan(tir_left[y]) else "NaN",
                round(float(tir_right[y]), 3) if not np.isnan(tir_right[y]) else "NaN",
                int(fold_mask[y]),
            ])
    print(f"  Wrote {path}")


# =============================================================================
# Test Runner
# =============================================================================

def run_test(test_name, resolution, squares, depth_type, out_dir):
    """Run a single test configuration through the pipeline and analyze.

    Args:
        test_name: str label for this test.
        resolution: int -- square image resolution.
        squares: int -- checkerboard squares per dimension.
        depth_type: "constant" or "ramp".
        out_dir: str -- output directory.

    Returns:
        metrics dict with left/right TIR stats.
    """
    print(f"\n{'='*64}")
    print(f"  {test_name}")
    print(f"  Resolution={resolution}, Squares={squares}, Depth={depth_type}")
    print(f"{'='*64}")

    # --- Generate test inputs ---
    print(f"  Generating {resolution}x{resolution} checkerboard ({squares}x{squares})...")
    rgb = _make_checkerboard(resolution, resolution, squares)
    depth_norm = (_make_constant_depth(resolution, resolution, 0.5) if depth_type == "constant"
                  else _make_ramp_depth(resolution, resolution))

    # --- Build camera ---
    camera = build_dual_camera()

    # --- Run pipeline ---
    print(f"  Running depth_to_unified_pointcloud...")
    t0 = time.time()
    content_h = PANEL_PHYS_H_M
    points_world, colors, x_sign, is_super, panel_owner, is_bg = \
        depth_to_unified_pointcloud(
            rgb, np.full((resolution, resolution), 0.5, dtype=np.float32),
            DEPTH_FWD_M, DEPTH_BACK_M,
            CONTENT_W_M, content_h,
            depth_norm=depth_norm,
            depth_power=1.0,
            fg_mask=None)
    n_pts = points_world.shape[0]
    print(f"  Done in {time.time() - t0:.2f}s -- {n_pts} points")

    print(f"  Creating Gaussian model...")
    t0 = time.time()
    model = create_gaussian_model(
        points_world, colors, sh_degree=0,
        img_width=resolution, content_w=CONTENT_W_M,
        overlap_factor=GAUSSIAN_OVERLAP_FACTOR,
        is_super_sampled=is_super,
        panel_owner=panel_owner,
        target_opacity=GAUSSIAN_OPACITY_SIGMOID,
        is_bg_point=is_bg)
    print(f"  Done in {time.time() - t0:.2f}s")

    print(f"  Rendering dual frustums...")
    t0 = time.time()
    out = render_dual(camera, model, BG_COLOR)
    print(f"  Done in {time.time() - t0:.2f}s")

    # --- Save renders ---
    render_left = _tensor_to_uint8(out["render_left"])
    render_right = _tensor_to_uint8(out["render_right"])
    _save_rgb(render_left, os.path.join(out_dir, "s6_render_left.png"))
    _save_rgb(render_right, os.path.join(out_dir, "s6_render_right.png"))
    stitched = np.hstack([render_left, render_right])
    _save_rgb(stitched, os.path.join(out_dir, "s6_render_stitched.png"))
    _save_rgb(rgb, os.path.join(out_dir, "s0_input_checkerboard.png"))

    # =====================================================================
    # Frequency Analysis
    # =====================================================================
    print(f"\n  --- Frequency Analysis ---")

    # Convert to grayscale
    input_gray = to_gray(rgb)
    left_gray = to_gray(render_left)
    right_gray = to_gray(render_right)

    # Split input into left/right halves (the pipeline splits at W/2)
    W_in = input_gray.shape[1]
    input_left_half = input_gray[:, :W_in // 2]
    input_right_half = input_gray[:, W_in // 2:]

    # Count transitions per row at native resolutions
    print(f"  Counting transitions...")
    in_left_trans = count_transitions_per_row(input_left_half)
    in_right_trans = count_transitions_per_row(input_right_half)
    out_left_trans = count_transitions_per_row(left_gray)
    out_right_trans = count_transitions_per_row(right_gray)

    render_h = out_left_trans.shape[0]
    in_left_resampled = resample_transitions(in_left_trans, render_h)
    in_right_resampled = resample_transitions(in_right_trans, render_h)

    expected_per_panel = (squares // 2) - 1  # N squares = N-1 internal boundaries
    in_left_mean = float(np.mean(in_left_trans))
    in_right_mean = float(np.mean(in_right_trans))
    out_left_mean = float(np.mean(out_left_trans))
    out_right_mean = float(np.mean(out_right_trans))

    print(f"    Input left half:       mean={in_left_mean:.1f} trans/row (expected ~{expected_per_panel})")
    print(f"    Input right half:      mean={in_right_mean:.1f} trans/row (expected ~{expected_per_panel})")
    print(f"    Left output ({render_h}px):  mean={out_left_mean:.1f} trans/row")
    print(f"    Right output ({render_h}px): mean={out_right_mean:.1f} trans/row")

    # Identify fold region rows (at render height)
    fold_mask, interior_mask = identify_fold_rows(points_world, panel_owner, render_h)

    # Compute TIR per panel against its corresponding input half
    tir_left = compute_transition_ratio(out_left_trans, in_left_resampled)
    tir_right = compute_transition_ratio(out_right_trans, in_right_resampled)

    # Compute region metrics
    metrics_left = compute_region_metrics(tir_left, fold_mask, interior_mask)
    metrics_right = compute_region_metrics(tir_right, fold_mask, interior_mask)

    print(f"\n    Left  TIR: fold_mean={metrics_left['TIR_fold_mean']}, "
          f"interior_mean={metrics_left['TIR_interior_mean']}, "
          f"excess={metrics_left['fold_excess_pct']}%")
    print(f"    Right TIR: fold_mean={metrics_right['TIR_fold_mean']}, "
          f"interior_mean={metrics_right['TIR_interior_mean']}, "
          f"excess={metrics_right['fold_excess_pct']}%")
    print(f"    Fold rows: {metrics_left['n_fold_rows']}, Interior rows: {metrics_left['n_interior_rows']}")

    # --- Visualizations ---
    print(f"\n  --- Generating plots ---")
    plot_transitions_vs_row(in_left_resampled, in_right_resampled,
                             out_left_trans, out_right_trans,
                             fold_mask,
                             os.path.join(out_dir, "transitions_vs_row.png"))
    plot_tir_vs_row(tir_left, tir_right, fold_mask,
                     os.path.join(out_dir, "tir_vs_row.png"))

    # --- CSV ---
    write_transition_csv(tir_left, tir_right,
                          in_left_resampled, in_right_resampled,
                          out_left_trans, out_right_trans,
                          fold_mask, out_dir)

    # --- Report ---
    config = {"resolution": resolution, "squares": squares,
              "depth_type": depth_type}
    generate_frequency_report(metrics_left, metrics_right, config, out_dir)

    # --- JSON ---
    json_path = os.path.join(out_dir, "frequency_metrics.json")
    summary = {
        "test_name": test_name,
        "config": config,
        "input_left_mean_transitions": round(in_left_mean, 1),
        "input_right_mean_transitions": round(in_right_mean, 1),
        "expected_per_panel": expected_per_panel,
        "left": {"output_mean_transitions": round(out_left_mean, 1),
                  **metrics_left},
        "right": {"output_mean_transitions": round(out_right_mean, 1),
                   **metrics_right},
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {json_path}")

    return summary


# =============================================================================
# Orchestrator
# =============================================================================

def harness():
    print("=" * 64)
    print("  FOLD FREQUENCY ANALYSIS HARNESS")
    print("  Spatial frequency preservation at the fold")
    print("  Read-only diagnostic -- no production code modified")
    print("=" * 64)
    print(f"  Device: {DEVICE}")
    print(f"  Panel:  {PANEL_PHYS_W_M}x{PANEL_PHYS_H_M} m "
          f"({PANEL_PIXEL_W}x{PANEL_PIXEL_H} px)")
    print(f"  Transition threshold: {TRANSITION_THRESHOLD}")
    print(f"  Fold proximity: {FOLD_PROXIMITY_MM * 1000:.0f} mm")
    print(f"  Output: {OUT_DIR}")

    all_results = []

    # Test 1: Low resolution baseline
    out1 = os.path.join(OUT_DIR, "test_01_cb_low_res")
    os.makedirs(out1, exist_ok=True)
    r1 = run_test("test_01_cb_low_res", 128, 8, "constant", out1)
    all_results.append(r1)

    # Test 2: Native resolution
    out2 = os.path.join(OUT_DIR, "test_02_cb_native")
    os.makedirs(out2, exist_ok=True)
    r2 = run_test("test_02_cb_native", 720, 16, "constant", out2)
    all_results.append(r2)

    # Test 3: Gradient depth
    out3 = os.path.join(OUT_DIR, "test_03_cb_gradient")
    os.makedirs(out3, exist_ok=True)
    r3 = run_test("test_03_cb_gradient", 720, 16, "ramp", out3)
    all_results.append(r3)

    # Cross-test summary
    print(f"\n{'='*64}")
    print(f"  CROSS-TEST SUMMARY")
    print(f"{'='*64}")
    print(f"  {'Test':<28s} {'TIR_fold_L':>10s} {'TIR_fold_R':>10s} {'TIR_int_L':>10s} {'TIR_int_R':>10s} {'Excess_L':>9s} {'Excess_R':>9s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*9}")
    for r in all_results:
        name = r["test_name"]
        l = r["left"]
        ri = r["right"]
        print(f"  {name:<28s} {str(l['TIR_fold_mean']):>10s} {str(ri['TIR_fold_mean']):>10s} "
              f"{str(l['TIR_interior_mean']):>10s} {str(ri['TIR_interior_mean']):>10s} "
              f"{str(l['fold_excess_pct']):>9s} {str(ri['fold_excess_pct']):>9s}")

    # Write cross-test manifest
    manifest_path = os.path.join(OUT_DIR, "all_frequency_metrics.json")
    with open(manifest_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Cross-test manifest: {manifest_path}")

    print(f"\n{'='*64}")
    print(f"  FOLD FREQUENCY ANALYSIS COMPLETE")
    print(f"  All outputs in: {OUT_DIR}/")
    print(f"  Report: {OUT_DIR}/fold_frequency_report.md")
    print(f"{'='*64}")


if __name__ == "__main__":
    harness()
