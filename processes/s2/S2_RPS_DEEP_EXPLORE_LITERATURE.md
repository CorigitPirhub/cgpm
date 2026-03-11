# S2 Deep Explore Literature Brief

日期：`2026-03-10`
协议背景：`S2 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
目标：为 `97_global_map_anchoring -> completeness recovery` 提供近三年顶会方法启发。

## 1. 核心论文

### 1) MorpheuS: Neural Dynamic 360° Surface Reconstruction from Monocular RGB-D Video
- 年份 / 会议：`CVPR 2024`
- 链接：
  - `https://openaccess.thecvf.com/content/CVPR2024/html/Wang_MorpheuS_Neural_Dynamic_360deg_Surface_Reconstruction_from_Monocular_RGB-D_Video_CVPR_2024_paper.html`
  - `https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_MorpheuS_Neural_Dynamic_360deg_Surface_Reconstruction_from_Monocular_RGB-D_Video_CVPR_2024_paper.pdf`
- 核心思想：
  - 用隐式表面表示动态场景；
  - 用扩散先验补足单目 RGB-D 下未观测区域；
  - 强调“360° surface reconstruction”而不是只拟合当前可见面。
- 可迁移点：
  - 我们当前 `TB` 簇可视为“高置信种子”；
  - 后续恢复 `Comp-R` 时不应做无约束扩张，而应做“seed-conditioned hidden surface completion”。

### 2) DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes
- 年份 / 会议：`CVPR 2025`
- 链接：
  - `https://cvpr.thecvf.com/virtual/2025/poster/33099`
  - `https://openaccess.thecvf.com/content/CVPR2025/papers/Peng_DeSiRe-GS_4D_Street_Gaussians_for_Static-Dynamic_Decomposition_and_Surface_Reconstruction_CVPR_2025_paper.pdf`
- 核心思想：
  - 先做 static/dynamic decomposition；
  - 用 temporal cross-view consistency 和几何正则抑制“floating in air”；
  - 目标不是只分离动态物体，而是把静态表面重建得物理合理。
- 可迁移点：
  - 我们的 `97` 已经具备 static/dynamic 分离能力；
  - 下一步应借鉴其 temporal cross-view consistency，把 `balloon2/balloon` 的弱支撑簇转成跨视角一致簇，而不是继续单帧局部 patch。

### 3) Decompositional Neural Scene Reconstruction with Generative Diffusion Prior
- 年份 / 会议：`CVPR 2025`
- 链接：
  - `https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf`
  - `https://arxiv.org/abs/2503.14830`
- 核心思想：
  - 将场景分解为背景/物体；
  - 用生成式扩散先验在稀疏观测下补出更平滑、更完整的背景；
  - 关键不是盲目 hallucination，而是 decomposition 后的 conditional completion。
- 可迁移点：
  - 我们现在已经有 `96/97` 的 purified clusters；
  - 因此后续恢复 `Comp-R` 时应做 `purified-cluster-conditioned completion`，而不是从全量 Noise 重新搜索。

### 4) Scene4U: Hierarchical Layered 3D Scene Reconstruction from Single Panoramic Image for Your Immerse Exploration
- 年份 / 会议：`CVPR 2025`
- 链接：
  - `https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Scene4U_Hierarchical_Layered_3D_Scene_Reconstruction_from_Single_Panoramic_Image_CVPR_2025_paper.html`
  - `https://www.openaccess.thecvf.com/content/CVPR2025/papers/Huang_Scene4U_Hierarchical_Layered_3D_Scene_Reconstruction_from_Single_Panoramic_Image_CVPR_2025_paper.pdf`
- 核心思想：
  - 先做 layered decomposition；
  - 再用 layered repair module 基于 diffusion + depth 修补遮挡区域；
  - 说明“先分层、后修补”比直接整体重建更稳。
- 可迁移点：
  - 非常适合我们当前框架：
  - `97` 先得到高精度 cluster layer；
  - 下一步可在 2D 视角内仅对该 layer 后方空洞做 inpainting，再反投影回 3D。

### 5) PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes
- 年份 / 会议：`CVPR 2025`
- 链接：
  - `https://openaccess.thecvf.com/content/CVPR2025/papers/Tan_PlanarSplatting_Accurate_Planar_Surface_Reconstruction_in_3_Minutes_CVPR_2025_paper.pdf`
  - `https://arxiv.org/abs/2412.03451`
- 核心思想：
  - 把 3D planes 作为主优化对象；
  - 用平面深度/法向约束显式拟合表面；
  - 对结构化场景比点级优化更稳定。
- 可迁移点：
  - 直接支撑了本轮 `96_support_cluster_model_fitting` 的 plane-band 策略；
  - 也指向下一步：不是恢复随机点，而是恢复 plane patches / structured surface elements。

## 2. 对当前 Rear-Point Cluster 框架的可迁移结论

### A. 当前最适配的方向
- `DeSiRe-GS` + `PlanarSplatting`
- 原因：
  - 我们已经解决了 dynamic/static disentanglement；
  - 也已经验证 plane-band fitting 能有效打破 `tb_noise_correlation` 死锁；
  - 因此下一步最自然的是“跨视角一致性 + 结构化平面恢复”。

### B. 次优但有潜力的方向
- `Scene4U`
- 原因：
  - 我们当前掉的是 `Comp-R`，本质是局部结构被过度清洗后留下空洞；
  - `2D layered repair -> back-project` 很适合做受控空洞补全。

### C. 暂不建议直接重型引入的方向
- `MorpheuS` / `Decompositional Neural Scene Reconstruction with Generative Diffusion Prior`
- 原因：
  - 这些方法强调生成式隐式补全，研究价值高，但对当前 `S2` 的工程落地成本偏高；
  - 更适合作为后续 `S2` 末段或独立研究支线，而不是当前最快补 `Comp-R` 的路径。

## 3. 对本轮实验设计的直接启发

### 已落地
- `96_support_cluster_model_fitting`
  - 对应 `PlanarSplatting` 风格的结构化平面拟合；
  - 结果：成功把 `tb_noise_correlation` 从 `0.991` 压到 `-0.756`。
- `97_global_map_anchoring`
  - 对应 `DeSiRe-GS` 风格的结构一致性与静态锚定；
  - 结果：相关性进一步压到 `-0.786`。

### 下一轮建议
- 不再继续“随机点扩张”；
- 应做 `purified cluster completion`，候选实现按优先级排序：
  1. `plane patch completion with temporal consistency`
  2. `layered 2D inpainting + back-projection`
  3. `seed-conditioned geodesic diffusion on structured planes`

## 4. 结论

- 文献结论与本地实验一致：
  - 先分离，再修补；
  - 先结构化，再补全；
  - 先保住几何一致性，再追求完整性。
- 对当前项目最关键的启示不是“继续更强清洗”，而是：
  - 以 `96/97` 过滤后的高置信 cluster 作为唯一修补种子，
  - 做结构化 completeness recovery，
  - 在保持 `tb_noise_correlation < 0.9` 的前提下，把 Bonn `Comp-R` 拉回 `70%+`。
