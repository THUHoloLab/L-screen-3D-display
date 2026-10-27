# L-DAGS: Automatic Anamorphic Content Generation for L-Shaped Displays

**Jiajun Peng, Huiyu Qian, Qiushuang Lian, and Liangcai Cao**  
Department of Precision Instrument, Tsinghua University, Beijing, China

L-DAGS (display-aware Gaussian splatting for L-shaped displays) converts ordinary monocular RGB videos into anamorphic content for two perpendicular display panels at a specified viewing position. Monocular depth guides the construction of frame-wise Gaussians in display coordinates, and two off-axis cameras sharing the same observer position project them onto the panels.

The workflow requires neither manual scene modeling nor acquisition-camera intrinsics. It constructs a display-conditioned 2.5D representation without scene-specific Gaussian training. The command-line implementation accepts both individual RGB images and videos.

## Method

1. **Estimate and stabilize depth.** Depth Anything 3 (DA3) estimates monocular depth. For video, the full configuration uses RAFT-guided fusion of the current and preceding raw depth maps, followed by RGB-guided joint bilateral filtering and temporal percentile anchoring.
2. **Place content in display coordinates.** Source pixels are mapped to the calibrated panel geometry, with depth controlling bounded displacement relative to the panels. This displacement is intended for display content generation; it is not recovered metric scene depth.
3. **Construct Gaussians.** Initialize Gaussian positions, colors, scales, and opacities from the depth-derived representation, including adaptive sampling where required.
4. **Render both panels.** A modified CUDA Gaussian rasterizer uses two asymmetric viewing frusta with a shared observer. The evaluated configuration packs the panel images seam-exclusively in the native dual-panel layout.

The paper's full configuration, **B3/A4**, uses **12 RAFT updates** and two consecutive raw depth maps. It does not recursively reuse fused depth as the historical input. **WARP2/CONF2** in the focused diagnostic experiments use two RAFT updates and are separate configurations. Optional guard-band blending and output-color propagation are not part of the formal benchmark.

## Results reported in the manuscript

The following values are reproduced from Table 1 of the manuscript. They are sequence means over **eight independent synthetic sequences** evaluated against panel-atlas image references.

| Metric | B1: 2D warp | B2/A1: Depth point | B3/A4: Full Gaussian |
| --- | ---: | ---: | ---: |
| PSNR (dB, higher is better) | 27.00 | 14.02 | 19.95 |
| SSIM (higher is better) | 0.923 | 0.397 | 0.856 |
| LPIPS (lower is better) | 0.037 | 0.330 | 0.074 |
| Seam luma MAE (lower is better) | 0.017 | 0.077 | 0.033 |
| Hole ratio (%, lower is better) | 0.00 | 11.48 | 2.18 |
| Temporal luma MAE (lower is better) | 0.0048 | 0.0369 | 0.0064 |

On **ten real-video clips**, B3/A4 reduced the hole ratio from **12.03% to 0.61%** and flow-aligned temporal luma MAE from **0.0340 to 0.0081**, relative to depth-point rendering. The real clips had no target-view image references, so reference-image PSNR, SSIM, and LPIPS were evaluated only on the synthetic set.

These comparisons establish gains over depth-point rendering. B1 achieves the best synthetic reference-image scores; the atlas measures image agreement, not independently recovered 3D geometry or perceived depth. Temporal metrics use each method's own valid correspondences. Physical-display observations remain qualitative, and validation is limited to one calibrated display configuration.

Reported cached-input timings exclude depth inference, flow inference, cache construction, and metric evaluation. They do not establish end-to-end real-time throughput.

## Repository structure

| Path | Purpose |
| --- | --- |
| `l_splat/core/` | Main pipeline, GPU point-cloud construction, dual-frustum cameras, filtering, and output orientation |
| `l_splat/depth/` | DA3 loading, depth anchoring, video depth streaming, and structure refinement |
| `l_splat/geometry/` | Display/frustum geometry and virtual-background utilities |
| `l_splat/temporal/`, `l_splat/flow_confidence.py` | Optical-flow confidence and adjacent-frame depth fusion |
| `l_splat/video_io.py` | Video input/output utilities |
| `l_splat/harnesses/` | Geometry, depth, temporal, GPU point-cloud, and pipeline checks |
| `gaussian-splatting-main/` | 3DGS inference modules, the modified CUDA rasterizer, and required GLM headers |
| `Depth-Anything-3-main/` | DA3 Python API dependencies and model configurations |
| `setup.py`, `config/envir.yaml` | Package definition and historical environment reference |

This repository contains the core inference implementation. Paper-specific benchmark scripts, datasets, cached inputs, figures, and historical experiment outputs are not included in this trimmed version. Running the examples below is therefore not a complete reproduction of the manuscript's tables.

## Requirements and installation

The development environment uses **Windows, Python 3.10, and an NVIDIA RTX 5080 Laptop GPU**. A compatible PyTorch/CUDA installation and a C++/CUDA compiler toolchain are required to build the rasterizer.

`config/envir.yaml` is a historical environment reference, not a verified clean-install specification for the manuscript environment. In particular, it specifies CUDA 12.1, whereas the manuscript experiments used CUDA 12.8. A fresh installation has not been validated as part of this documentation update.

Clone the repository:

```powershell
git clone https://github.com/Jrn-0313/L-DAGS.git
cd L-DAGS
```

From the repository root, activate your configured environment and install the local packages. The authors' existing environment is named `env_5080`:

```powershell
conda activate env_5080
python -m pip install -e .
python -m pip install -e Depth-Anything-3-main
python -m pip install --no-build-isolation -e gaussian-splatting-main/submodules/diff-gaussian-rasterization
```

Use the **modified rasterizer included in this repository**. The `simple-knn` training extension is not required by the current inference entry point, which uses a compatibility stub and does not execute its KNN training operations.

DA3 weights are loaded from Hugging Face or the local cache and are not stored in Git. The bundled DA3 code provides its Python API; its standalone CLI and web interface are not included.

## Usage

Inspect the available arguments:

```powershell
python -m l_splat.core.pipeline --help
```

Process a single image:

```powershell
python -m l_splat.core.pipeline --image C:/data/input.jpg --out-dir output/image
```

Process a video with RAFT-guided temporal fusion and seam-boundary support disabled:

```powershell
python -m l_splat.core.pipeline --video C:/data/input.mp4 --max-frames 100 --flow-confidence on --seam-boundary-support-mode off --out-dir output/video
```

Replace the input paths with your own files. These commands run inference; reproducing the reported benchmarks also requires the corresponding evaluation inputs, configurations, and protocols.

| Option | Behavior |
| --- | --- |
| `--max-frames N` | Number of video frames to process; the default is 5, and 0 processes all available frames |
| `--frame-stride N` | Process every Nth input frame; the default is 1 |
| `--flow-confidence on` | Require RAFT; initialization failures are reported instead of silently selecting EMA |
| `--flow-confidence auto` | Default mode: use cached RAFT weights when available, otherwise fall back to legacy EMA |
| `--flow-confidence off` | Use legacy EMA depth fusion |
| `--flow-history-weight` | Maximum aligned previous-raw-depth weight; the default is 0.5 |
| `--seam-boundary-support-mode off` | Disable additional seam-support samples; this is the default |
| `--save-diagnostics` | Save the extended diagnostic outputs in video mode |
| `--out-dir` | Output directory; the default is `output/` |

### Display calibration

The manuscript configuration uses two panels at **90 degrees**, each **1.28 m wide and 1.60 m high**, with **936 x 1176 pixels per panel**. The observer is 3.00 m horizontally from the fold on the angular bisector, at 0.40 m above the panel center. Depth-derived display displacement spans -0.15 m to +0.15 m.

The default CLI height is an eye height of 170 cm above the floor, with the panel bottom 50 cm above the floor. For the default 160 cm panel height, this gives the paper's 40 cm height relative to the panel center. Use `--sweet-spot-height-mode world-z` to specify height directly in the centered display coordinate system; height arguments remain in centimeters.

Panel dimensions, raster size, and observer settings can be inspected with `--help` and in [`l_splat/core/pipeline.py`](l_splat/core/pipeline.py). Set these to match the physical installation before generating playback content.

### Outputs

Single-image results are written directly to the selected output directory. Video results are written to per-frame subdirectories named `frame_XXXX/`. Use `render_left.png` and `render_right.png` for panel playback. `render_raw_combined.png` is a diagnostic concatenation, and `parameter_provenance.json` records the configuration and output orientation. Combined or blended previews are separate from the individual panel outputs. The video command produces frame outputs rather than automatically assembling a playback MP4.

## Core checks

From the repository root, keep `PYTHONPATH` pointed at the current directory:

```powershell
$env:PYTHONPATH='.'
python l_splat/harnesses/test_frustum.py
python l_splat/harnesses/test_canonical_runtime_contract.py
python l_splat/harnesses/test_output_orientation_contract.py
python l_splat/harnesses/test_flow_guided_depth_fusion.py
python l_splat/harnesses/test_video_flow_streamer.py
python l_splat/harnesses/test_gpu_pointcloud_integration.py
```

These scripts check specific implementation properties. They do not by themselves reproduce the paper's benchmark results or validate perceptual performance on a physical display.

## Third-party code and licenses

This project includes components from Depth Anything 3, 3D Gaussian Splatting, the differentiable Gaussian rasterizer, and GLM. Their license files are retained:

- [Depth Anything 3 license](Depth-Anything-3-main/LICENSE)
- [3D Gaussian Splatting license](gaussian-splatting-main/LICENSE.md)
- [CUDA rasterizer license](gaussian-splatting-main/submodules/diff-gaussian-rasterization/LICENSE.md)
- [GLM license](gaussian-splatting-main/submodules/diff-gaussian-rasterization/third_party/glm/copying.txt)

Follow each component's license terms when using or redistributing it.
