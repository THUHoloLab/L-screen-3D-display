# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

---

## ⚠️ 治理合规要求（最高优先级 — 所有智能体必须严格遵循）

本项目在 `.claude/governance/` 下设有**强制性**治理体系。所有智能体、所有任务、所有代码修改**必须**严格遵循以下流程，无一例外：

### 工作前（Pre-Task）
1. **变更分类**：按照 `CHANGE_CONTROL.md` 将变更分类为 PATCH / FEATURE / REFACTOR / ARCHITECTURE_CHANGE。未分类的变更**不得执行**。
2. **阶段检查**：查阅 `PHASE_GOVERNANCE.md`，确认当前阶段（**Phase 0：投影验证**）允许修改的目标文件。超出阶段白名单的文件修改**需要明确批准**。
3. **研究对齐声明**：按照 Rule E-28，每项编码任务开始前必须撰写 Research Alignment Statement，明确关联到 RQ-1 至 RQ-6 中的研究问题。
4. **冻结组件检查**：确认目标文件或参数不在冻结清单中。冻结项修改**必须人工 UNLOCK**。

### 执行中（During Task）
5. **35条强制规则**：遵守 `AGENT_CONSTITUTION.md` 的全部规则（硬停止、范围约束、证据要求），不得绕过。
6. **成功标准对齐**：所有修改必须服务于 SC-01 至 SC-11 成功标准（见 `PROJECT_CONSTITUTION.md`）。
7. **NOS 红线**：NOS-01 至 NOS-20 为**永久超出范围**（见 `PROJECT_CONSTITUTION.md` §Part IV），绝对不得实现。

### 工作后（Post-Task）
8. **证据归档**：变更完成后，记录变更分类、研究对齐声明、受影响文件，确保可追溯。

### 冻结组件（修改需人工 UNLOCK）
| 组件 | 说明 |
|------|------|
| `geometry/frustum.py` | 视锥体几何计算 |
| 折叠角度 | 固定 90° |
| 甜点位置 | (3.0/√2, 3.0/√2, 1.70) ≈ (2.12, 2.12, 1.70)，距离3.0m |
| 反投影公式 | 盒子投影 + 45度平面融合 |
| Alpha 合成公式 | 能量守恒 smoothstep |
| DAV3/RAFT 权重 | 预训练模型权重 |

**违反以上任何一条规则，必须在执行前获得用户明确批准。**

### 治理体系文档（`.claude/governance/`）
| 文档 | 内容 |
|------|------|
| `AGENT_CONSTITUTION.md` | 35条强制性规则（硬停止、范围约束、证据要求） |
| `PROJECT_CONSTITUTION.md` | 任务目标、RQ-1至RQ-6、成功标准 SC-01至SC-11、冻结假设、非目标项 |
| `PHASE_GOVERNANCE.md` | **当前阶段：Phase 0（投影验证）** — 各阶段允许/禁止的文件清单 + 非生产工作模式 |
| `CHANGE_CONTROL.md` | 变更分类流程：PATCH / FEATURE / REFACTOR / ARCHITECTURE_CHANGE |
| `RESEARCH_ALIGNMENT.md` | 研究对齐声明模板与要求（Rule E-28） |
| `WAIVER_PROTOCOL.md` | UNLOCK/豁免协议的正式流程 |
| `CODE_REVIEW_PROTOCOL.md` | 代码审查协议与证据要求 |
| `AGENT_BEHAVIOR_SCORECARD.md` | 智能体行为评分卡 |
| `README_GOVERNANCE.md` | 治理体系总览与快速入门 |

### Governance v2 非生产工作模式

除生产实现模式外，Phase Governance v2 新增了三种非生产模式，允许审计和诊断工作在不推进实现阶段的前提下进行：

- **Audit Mode（审计模式）**：只读调查、架构分析、取证报告。可写入 `docs/audits/` 和 `docs/session_records/`。禁止修改源代码。
- **Diagnostic Mode（诊断模式）**：创建和运行纯诊断性 harness（`l_splat/harnesses/`），对比生产输出与独立参考。禁止修改生产代码、冻结几何、渲染代码、CUDA代码、DAV3/RAFT代码或高斯初始化代码。
- **Commit Mode（提交模式）**：证据和审查门禁。提交前需完成 `CODE_REVIEW_PROTOCOL.md` 和 `AGENT_BEHAVIOR_SCORECARD.md`。

---

## 项目：L-Splat

针对L型90度外折裸眼3D LED显示屏的实时单目转变形3DGS引擎。
单RGB相机 -> DAV3深度 -> 3D高斯泼溅 -> 双非对称视锥体CUDA光栅化 -> 2D Alpha混合输出。

Git 仓库已初始化（3 次提交，分支 `main`）。执行破坏性编辑前先 `git status` 确认状态。

## 环境与命令

```bash
# 激活conda环境（所有工作必需）
conda activate env_5080
# Python路径：/c/Users/34893/anaconda3/envs/env_5080/python
# GPU：NVIDIA GeForce RTX 5080 Laptop (16 GB)

# 单图流水线（端到端测试，约60秒）
cd C:\Users\34893\Desktop\Gaussian_modification
PYTHONPATH=. python l_splat/core/pipeline.py --image ~/Desktop/hypercar.jpg

# 单图，不进行前/背景分割
PYTHONPATH=. python l_splat/core/pipeline.py --image ~/Desktop/hypercar.jpg --no-segment

# 视频流水线
PYTHONPATH=. python l_splat/core/pipeline.py --video ~/Videos/test.mp4 --ema-alpha 0.3 --max-frames 100 --frame-stride 1

# 镜像标志（水平翻转左/右视锥体）
PYTHONPATH=. python l_splat/core/pipeline.py --image ~/Desktop/hypercar.jpg --mirror-left --mirror-right

# 测试套件回归（流水线变更后运行）
PYTHONPATH=. python l_splat/harnesses/test_frustum.py                  # 几何验证
PYTHONPATH=. python l_splat/harnesses/test_blending.py                 # Alpha混合 + 重叠
PYTHONPATH=. python l_splat/harnesses/test_box_projection.py           # 盒子投影几何
PYTHONPATH=. python l_splat/harnesses/test_zero_copy.py                # RealtimeGaussianPool
PYTHONPATH=. python l_splat/harnesses/test_depth_anchor.py             # EMA时序锚定
PYTHONPATH=. python l_splat/harnesses/test_depth_loader.py             # DAV3加载 + 20轮基准
PYTHONPATH=. python l_splat/harnesses/test_depth_filter.py             # cv2 C++ JBF正确性与性能
PYTHONPATH=. python l_splat/harnesses/test_depth_streamer.py           # VideoDepthStreamer
PYTHONPATH=. python l_splat/harnesses/test_single_image_pipeline.py    # 合成端到端
PYTHONPATH=. python l_splat/harnesses/test_dot_projection.py           # 点投影往返
PYTHONPATH=. python l_splat/harnesses/test_diagnostic_projection.py    # 45度平面几何诊断
PYTHONPATH=. python l_splat/harnesses/test_extrinsics.py               # 视图矩阵验证
PYTHONPATH=. python l_splat/harnesses/test_video_extractor.py          # FastVideoExtractor
PYTHONPATH=. python l_splat/harnesses/test_video_flow_pipeline.py      # FlowConfidence流水线
PYTHONPATH=. python l_splat/harnesses/test_asymmetric_projection_chain.py  # 完整投影链
PYTHONPATH=. python l_splat/harnesses/test_content_placement.py        # 内容放置验证
PYTHONPATH=. python l_splat/harnesses/test_corrected_placement.py      # 修正放置回归
PYTHONPATH=. python l_splat/harnesses/test_world_placement_audit.py    # 世界放置审计
PYTHONPATH=. python l_splat/harnesses/test_virtual_box.py              # 虚拟盒子背景

# 参数验证测试
PYTHONPATH=. python l_splat/harnesses/harness_3d_scale_shrink.py       # 远场缩放收缩
PYTHONPATH=. python l_splat/harnesses/harness_opacity_fade.py          # sigmoid不透明度衰减
PYTHONPATH=. python l_splat/harnesses/harness_scale_comparison.py      # 缩放模型比较
PYTHONPATH=. python l_splat/harnesses/harness_grazing_filter.py        # 动态低通滤波
PYTHONPATH=. python l_splat/harnesses/harness_mirror_module.py         # FLIP_X镜像矩阵
PYTHONPATH=. python l_splat/harnesses/harness_sh_clamp.py              # SH阶数边界测试

# 棋盘格取证（只读逐阶段诊断，识别损坏来源）
PYTHONPATH=. python l_splat/harnesses/checkerboard_forensics.py

# 折叠区域取证（只读诊断，测量90度折叠缝处的畸变）
PYTHONPATH=. python l_splat/harnesses/fold_forensics.py

# 折叠区域频率分析（只读诊断，逐行黑白跳变计数 + 跳变膨胀比TIR）
PYTHONPATH=. python l_splat/harnesses/test_fold_frequency.py

# 质量指标（在流水线运行后对输出进行评分）
PYTHONPATH=. python l_splat/harnesses/quality_metrics.py [output_dir]

# 参考投影器（纯诊断几何投影器，非生产代码）
PYTHONPATH=. python l_splat/harnesses/reference_projector.py

# 折叠缝诊断系列（Task10A-10J，只读诊断harness）
PYTHONPATH=. python l_splat/harnesses/task10a_seam_strategy_harness.py          # 零重叠缝策略证据
PYTHONPATH=. python l_splat/harnesses/task10b_shared_seam_ray_harness.py        # 共享缝射线边界一致性
PYTHONPATH=. python l_splat/harnesses/task10c_seam_gaussian_footprint_harness.py # 缝局部高斯足迹行为
PYTHONPATH=. python l_splat/harnesses/task10d_rendered_representation_attribution.py # 渲染表示归因
PYTHONPATH=. python l_splat/harnesses/task10e_seam_local_boundary_support_repair.py  # 边界支撑候选评估
PYTHONPATH=. python l_splat/harnesses/task10h_boundary_support_rollback_review.py    # 边界支撑回滚审查
PYTHONPATH=. python l_splat/harnesses/task10i_b_production_equivalent_diagnostic_replacement.py  # 生产等效诊断替换
PYTHONPATH=. python l_splat/harnesses/task10i_c_flat_depth_feature_provenance.py  # 平面深度特征溯源
PYTHONPATH=. python l_splat/harnesses/task10i_e_edge_aware_gaussian_footprint_density_repair.py  # 边缘感知修复
PYTHONPATH=. python l_splat/harnesses/task10i_g_synthetic_known_geometry_cube_diagnostic.py  # 合成已知几何诊断
PYTHONPATH=. python l_splat/harnesses/task10i_i_aggressive_representation_contract_repair_prototype.py  # 激进表示契约原型
PYTHONPATH=. python l_splat/harnesses/task10j_c_renderer_coverage_alpha_diagnostic.py  # 渲染器覆盖/Alpha诊断
PYTHONPATH=. python l_splat/harnesses/task12_surface_patch_level_representation_prototype.py  # 面片级表示原型

# 模块自测
PYTHONPATH=. python l_splat/core/dual_frustum.py    # 双视锥体自测
```

## 架构

### 流水线阶段（在 `l_splat/core/pipeline.py` 中）

`process_frame()` 函数执行7个阶段：

1. **DAV3深度估计**（`l_splat/depth/loader.py`）—— DINOv2-ViT-S骨干网络，process_res=756，FP16
2. **cv2 C++ JBF滤波**（`l_splat/core/depth_filter.py`）—— RGB引导的边缘保持深度平滑
3. **反投影**（`depth_to_unified_pointcloud()`）—— 盒子投影（图像在W/2处分割，每半幅映射到面板平行平面）。同时处理稀疏区域的超采样。
4. **高斯初始化**（`create_gaussian_model()`）—— 统一像素间距缩放 + 自适应max_scale上限 + 深度感知裁剪 + 远场收缩 + 不透明度衰减。在模型上存储`_is_super_sampled`、`_panel_owner`、`_panel_{left,right}_mask`。
5. **双视锥体相机**（`build_dual_camera()` -> `l_splat/core/dual_frustum.py`）—— 每面板非对称离轴投影
6. **CUDA光栅化**（`gaussian-splatting-main/submodules/diff-gaussian-rasterization/`）—— 修改版3DGS，采用`use_asymmetric=True`（FLT_MAX裁剪，跳过NDC剔除，invdepth输出）
7. **2D Alpha混合**（`soft_blend_seam()`）—— 线性alpha渐变折叠缝（`np.linspace`，非smoothstep）

集成到`process_frame()`中的其他阶段：
- **1.5：前/背景分割**（`l_splat/segmentation/sam3_wrapper.py`）—— SAM3文本提示分割或中心高斯启发式回退方案。使用`--no-segment`禁用。
- **背景墙壁生成**（`l_splat/geometry/virtual_box.py`）—— `map_background_to_panels()`在内容平面后方生成背景墙壁高斯

### 坐标系统

- **右手系，Z轴朝上**。折叠线 = Z轴（X=0, Y=0）。
- **左面板**：Y=0平面，X ∈ [-1.28, 0] 米
- **右面板**：X=0平面，Y ∈ [-1.28, 0] 米
- **甜点**：(3.0/√2, 3.0/√2, 1.70) —— +X+Y象限的45度角平分线，距离折叠线3.0m
- **左视锥体**朝向(0,-1,0)方向；**右视锥体**朝向(-1,0,0)方向
- **视图矩阵**：`flip_y @ look_at_offscreen()`，det=-1（候选方案H）

注意：`docs/ARCHITECTURE_SPEC.md`采用Y-up约定。这仅是命名差异 —— 物理几何完全相同。代码使用Z-up。阅读规范文档时，在脑中映射：规范Y -> 代码Z，规范Z -> 代码X/Y。

### 面板遮罩（`_panel_owner`）与渲染分割

- `0` = 仅左面板（输入图像左半部），`1` = 仅右面板（输入图像右半部）
- 基于**输入图像列**分配，而非世界坐标。每个高斯仅在对应其输入图像半部的一个视锥体中渲染。
- 不再使用`panel_owner=2`（折叠融合区）。smoothstep插值提供几何C0连续性，无需双视锥体渲染。
- 回退方案（当没有`_panel_owner`时）：基于世界X坐标的分割（`xyz[:,0] < 0.001`）

### 统一缩放模型

`base_scale = pixel_pitch * GAUSSIAN_OVERLAP_FACTOR`，其中`pixel_pitch = content_w / img_width`。
受**自适应上限**约束：`max(3.2mm, pixel_pitch * GAUSSIAN_OVERLAP_FACTOR * 0.8)`。
对于720p输入（pixel_pitch=1.78mm），固定上限3.2mm生效。对于128p输入（pixel_pitch=10mm），自适应上限放宽至8.0mm以确保覆盖。
超采样点使用`effective_pixel_pitch = pixel_pitch / SUPER_SAMPLE_SIZE`作为缩放下限。

**混合区超采样排除**：自适应超采样掩码排除折叠混合区像素（`|uu - split_u| <= blend_px`）。
折叠处90度转折在相邻像素（不同投影平面）之间产生人为夸大的3D间距，触发超采样，
产生16倍同色副本高斯，导致过渡膨胀。排除后，在128p输入下，超采样像素从~87%降低至~47%。

### 核心模块

| 模块 | 用途 |
|--------|---------|
| `l_splat/core/pipeline.py` | 主流水线：`process_frame()`、`create_gaussian_model()`、`depth_to_unified_pointcloud()`、`build_dual_camera()`、`render_dual()`、`soft_blend_seam()`、`save_results()` |
| `l_splat/core/dual_frustum.py` | `DualFrustumCamera` —— 两个非对称视锥体共享一个观察者。管理V、P矩阵和面板遮罩。 |
| `l_splat/core/gaussian_pool.py` | `RealtimeGaussianPool` —— 视频零拷贝GPU池（已构建，**尚未集成到流水线中**） |
| `l_splat/core/depth_filter.py` | `cv2_jbf()` —— OpenCV C++ Joint Bilateral Filter封装 |
| `l_splat/depth/loader.py` | DAV3模型加载与推理（`load_dav3`、`predict_depth_fast`、`predict_depth_with_jbf`） |
| `l_splat/depth/anchor.py` | `DepthAnchorStream` —— 用于时序稳定性的EMA平滑百分位数深度归一化 |
| `l_splat/depth/streamer.py` | `VideoDepthStreamer` —— 带EMA深度平滑的多线程视频I/O |
| `l_splat/geometry/frustum.py` | 纯numpy几何：`look_at_offscreen`、`compute_frustum_bounds`、`build_asymmetric_proj_matrix`、`build_l_screen_corners` |
| `l_splat/geometry/virtual_box.py` | 虚拟空间盒子 + 用于内容后方变形背景墙壁的逆感知映射 |
| `l_splat/segmentation/sam3_wrapper.py` | SAM3文本提示前/背景分割，带中心高斯回退方案 |
| `l_splat/video_io.py` | `VideoIO` —— 视频帧提取 + MP4合成 |
| `l_splat/flow_confidence.py` | `FlowConfidenceEstimator` —— RAFT前后向光流一致性（**尚未集成到流水线中**） |
| `l_splat/harnesses/quality_metrics.py` | 7维度数值评分卡（覆盖率、填充度、锐度、双峰性、矩形度、边缘、平衡性） |
| `l_splat/harnesses/checkerboard_forensics.py` | 只读逐阶段损坏分析，使用合成输入，绕过DAV3/SAM3/时序模块 |
| `l_splat/harnesses/fold_forensics.py` | 只读折叠区域诊断：细胞压缩、线偏差、畸变热图、拓扑评分 |
| `l_splat/harnesses/test_fold_frequency.py` | 只读空间频率保留分析：逐行黑白跳变计数、跳变膨胀比(TIR)、折叠vs内部对比 |
| `l_splat/harnesses/reference_projector.py` | 纯诊断几何参考投影器（非生产代码），用于对比验证生产投影输出 |
| `l_splat/harnesses/task10a-e_*.py` | Task10A-10E 系列：折叠缝策略、共享射线、高斯足迹、渲染归因、边界支撑评估 |
| `l_splat/harnesses/task10h_*.py` | Task10H：边界支撑回滚/窄消融审查 |
| `l_splat/harnesses/task10i_*.py` | Task10I 系列：生产等效诊断替换、平面深度特征溯源、边缘感知修复、合成几何诊断、激进表示契约原型 |
| `l_splat/harnesses/task10j_c_*.py` | Task10J-C：渲染器覆盖/Alpha 诊断深化 |
| `l_splat/harnesses/task12_*.py` | Task12：表面/面片级表示原型 |

### 当前活跃参数（2026-06-30，更新 2026-06-30）

```python
# pipeline.py -- 配置常量（Task07A 硬件配置）
PANEL_PIXEL_W = 936; PANEL_PIXEL_H = 1176  # 每面板渲染分辨率
PANEL_PHYS_W_M = 1.28; PANEL_PHYS_H_M = 1.60  # 物理尺寸（高宽比 1.60/1.28 = 1.25）
SWEET_SPOT = np.array([3.0/math.sqrt(2), 3.0/math.sqrt(2), 1.70])  # 距离折叠线3.0m，Z=1.70
CONTENT_W_M = PANEL_PHYS_W_M  # = 1.28m
# content_h = PANEL_PHYS_H_M = 1.60m（在process_frame()中固定，填充整个面板高度）
DEPTH_FWD_M  = +0.15; DEPTH_BACK_M = -0.15  # 对称+/-0.15m，总计0.30m范围
OVERLAP_WORLD_M = 0.0; BLEND_PIXELS_TARGET = 50  # Task07B：规范原始面板使用物理折叠，无重叠视锥体

# create_gaussian_model() 默认值
GAUSSIAN_OVERLAP_FACTOR = 1.0  # 从2.0降至1.0，将边界过渡区从~6px缩至~2px
GAUSSIAN_OPACITY_SIGMOID = 0.999  # 目标不透明度（从0.99提升以补偿更小的缩放）
depth_clamp_min=2.0; depth_clamp_range=8.0
far_shrink_gamma=2.0; far_shrink_max=0.0  # 已禁用
shrink_floor_ratio=0.5; max_scale_mm=3.2  # 自适应：max(3.2mm, pixel_pitch * OVERLAP_FACTOR * 0.8)，见统一缩放模型
fade_k=12.0; fade_threshold=0.10

# 折叠区域感知参数（当前为死代码，panel_owner==2不出现）
FOLD_SCALE_BOOST = 1.1       # panel_owner==2点的缩放提升（当前未触发）
FADE_THRESHOLD_M = 0.010     # 过渡区（10mm），|X|内高斯在两个视锥体渲染（仅panel_owner=2回退时使用）

# depth_to_unified_pointcloud() 默认值
depth_power=1.0  # 线性映射
blend_px = max(int(0.015 * W_img), 8)  # 图像宽度的1.5%，最小8px

# process_frame() JBF默认值
jbf_ksize=9; jbf_sigma_s=5.0; jbf_sigma_r=0.03

# 超采样（720p因MAX_GAUSSIANS限制跳过；混合区像素被排除）
# 超采样颜色使用最近邻插值（保留二值边界，消除灰色中间高斯）
SUPER_SAMPLE_SIZE=4; LOCAL_SPACING_THRESHOLD=2.0; MAX_GAUSSIANS=5_000_000

# Task10E 折叠边界支撑（默认禁用，Task10H 将其隔离出生产路径）
SEAM_BOUNDARY_SUPPORT_ENABLED = False
SEAM_BOUNDARY_SUPPORT_BAND_PX = 16

# Task10I-E 边缘感知高斯足迹/密度修复（默认禁用）
EDGE_AWARE_GAUSSIAN_REPAIR_ENABLED = False
EDGE_AWARE_SCALE_MULTIPLIER = 1.45

# Task10I-I 激进表示契约修复原型（默认禁用）
REPRESENTATION_CONTRACT_PROTOTYPE_ENABLED = False
# 8类表示分类：UNKNOWN, FACE_INTERIOR, TEXTURE_ONLY_EDGE, DEPTH_DISCONTINUITY_EDGE,
#   SILHOUETTE_EDGE, FOLD_NEAR_STRUCTURAL_EDGE, BACKGROUND_CONTROL, AMBIGUOUS_BOUNDARY

# Task12 表面/面片级表示原型（默认禁用）
SURFACE_PATCH_LEVEL_PROTOTYPE_ENABLED = False
SURFACE_PATCH_PROTO_TILE_SOURCE_PX = 14
```

### 新增 CLI 参数（2026-06-30）

```bash
# 硬件配置覆盖
--panel-width-m 1.28 --panel-height-m 1.60
--left-panel-px-w 936 --right-panel-px-w 936 --panel-px-h 1176
--seam-width-m 0.0 --seam-px 50
--sweet-spot-distance-cm 300 --sweet-spot-height-cm 170

# 功能开关（可通过 --enable-* / --disable-* 控制）
--enable-edge-aware-gaussian-repair / --disable-edge-aware-gaussian-repair
--enable-representation-contract-prototype / --disable-representation-contract-prototype
--enable-surface-patch-prototype / --disable-surface-patch-prototype
```

### HDD（测试驱动开发）模式

`l_splat/harnesses/test_*.py`中的每个测试都是包含独立断言的独立脚本。
测试不得依赖pipeline.py内部实现 —— 它们为了独立性而重复核心数学逻辑。
新功能在修改生产代码之前必须先有测试。

### CUDA修改：`use_asymmetric`

`gaussian-splatting-main/submodules/`中修改后的`diff-gaussian-rasterization`将`use_asymmetric=True`传播至整个CUDA栈：
- `computeCov2D`：limx=FLT_MAX（绕过对称FOV裁剪）
- `preprocessCUDA`：跳过NDC剔除（离轴点可能位于对称NDC之外）
- `renderCUDA`：启用invdepth输出
- `backward.cu`：x_grad_mul=1.0（对裁剪点的梯度不阻断）
- 伪tanfov：`tan_fovx = 1/A_x`，其中`A_x = 2n/(r-l)`来自非对称投影矩阵

### 关守系统（GateGuard）

本项目启用了GateGuard，要求在文件编辑和bash命令之前进行事实陈述。当被阻止时，需说明：
1. 导入/依赖目标文件的文件
2. 受影响的公开函数/类
3. 是否读取/写入数据文件
4. 用户的当前指令原文

## 当前状态（2026-06-30）

### 已知问题
- **单像素噪声（深度退化）**：恒定深度下所有高斯z_view相同，CUDA排序对等键值保留输入顺序（扫描线交替黑白）。
  alpha混合非交换律（opacity=0.999），产生极少量的像素级变化。720p下TIR=1.078/1.09，接近1.0。
- **折叠代码死路径**：`FOLD_SCALE_BOOST=1.1`和`FADE_THRESHOLD_M=0.010`仅在`panel_owner==2`时触发。当前`panel_owner`仅赋值0或1，因此这些代码路径从未执行。
- **折叠缝覆盖间隙**：原始面板（OVERLAP_WORLD_M=0.0）在折叠缝处存在可见间隙。Task10系列正在诊断和修复中。

### 近期完成
- **Governance v2**：新增 Audit Mode、Diagnostic Mode、Commit Mode 三种非生产工作模式
- **Task07A 硬件配置更新**：面板分辨率 936×1176，物理高度1.60m，甜点距离3.0m（Z=1.70）
- **Task07B 重叠归零**：OVERLAP_WORLD_M=0.0，规范原始面板使用物理折叠
- **参考投影器**：`l_splat/harnesses/reference_projector.py` — 纯诊断几何投影器，用于对比验证
- **Task10A-10J 折叠缝诊断系列**：10+个诊断harness覆盖缝策略、共享射线、高斯足迹、渲染归因、边界支撑、边缘感知修复、表示契约原型
- **Task12 表面面片原型**：面片级表示路径（默认禁用）
- **可配置 CLI**：新增 `--panel-width-m`、`--panel-height-m`、`--sweet-spot-distance-cm`、`--sweet-spot-height-cm`、`--seam-width-m` 等参数

### 待处理
- **P1**：将`RealtimeGaussianPool`集成到视频流水线（零拷贝，避免每帧重建）
- **P2**：折叠缝覆盖修复（Task10系列诊断完成后评估方案）
- **P3**：FlowConfidenceEstimator用于视频的时序滤波
- 通用缩放/SH调优

### 未来探索
- `sam3-main/`和`4DGaussians-master/`是探索目标。未经事先讨论不得实施。

## 核心文档

- `docs/ARCHITECTURE_SPEC.md` —— 理想化的10阶段架构规范（Y-up约定，与Z-up代码不同）
- `docs/ARCHITECTURE_AUDIT.md` —— 全面的差距分析：规范vs实际代码
- `docs/world_placement_audit.md` —— `depth_to_unified_pointcloud()`的深度数值分析
- `docs/l_screen_geometry_audit.md` —— L屏幕几何的详细分析
- `ISSUES.md` —— 9个观察到的视觉质量问题

## 归档

`archive/` —— 废弃代码（3D折叠增密、早期冒烟测试）。不被任何活跃代码导入。
`gaussian-splatting-main/` —— 修改版3DGS分支（CUDA内核 + 场景工具）。
`Depth-Anything-3-main/` —— DAV3本地源码（可编辑安装）。
`4DGaussians-master/`、`sam3-main/` —— 未来集成目标，尚未链接。
