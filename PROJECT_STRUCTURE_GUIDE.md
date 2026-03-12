# PROJECT_STRUCTURE_GUIDE

## 1. 结构合理性确认

基于当前仓库状态，现有分层是合理的，原因如下：

- **主线集中**：稳定建图实现集中在 `egf_dhmap3d/core/`、`egf_dhmap3d/modules/`、`egf_dhmap3d/data/`、`egf_dhmap3d/eval/`。
- **实验隔离**：阶段性脚本与试验性方法集中在 `experiments/<stage>/`，未接纳方法不会混入主线。
- **结果主导**：阶段结果直接进入 `output/<stage>/`，比额外维护独立历史目录更直接。
- **细节下沉**：详细实验结果、中间文件、临时文件集中放到 `output/tmp/`，避免污染阶段总览目录。
- **外部依赖隔离**：第三方代码保留在 `third_party/`。

结论：后续统一按 `egf_dhmap3d/ + experiments/ + output/ + output/tmp/` 组织文件是合理且必要的。

## 2. 划分原则

- **主线逻辑**：放在 `egf_dhmap3d/` 与稳定入口 `scripts/`。
- **实验代码**：放在 `experiments/<stage>/`。
- **结果记录**：放在 `output/<stage>/`。
- **细节文件**：放在 `output/tmp/`。

## 3. 目录职责

- `egf_dhmap3d/`
  - 主线 3D 建图实现。
  - `core/`、`modules/`、`data/`、`eval/` 属于稳定主线。
  - 只有确认被主线接纳的方法，才允许进入该目录。

- `scripts/`
  - 仅保留主线入口与稳定工具。
  - 典型文件：
    - `scripts/run_benchmark.py`
    - `scripts/run_benchmark_bonn.py`
    - `scripts/run_egf_3d_tum.py`
    - `scripts/run_tsdf_baseline.py`
    - `scripts/run_simple_removal_baseline.py`
    - `scripts/update_summary_tables.py`

- `experiments/`
  - 阶段实验脚本与实验性方法隔离区。
  - 规则：
    - `experiments/s1/`、`experiments/s2/`：阶段实验脚本
    - `experiments/p10/`、`experiments/p11/` 等：专项/论文阶段脚本与方法包
    - 试验性方法先留在 `experiments/<stage>/`，只有被明确接纳后才可迁入主线

- 根目录理论文档
  - `SURVEY.md`：跨方案调研总报告。
  - `DESIGN_A.md`、`DESIGN_B.md`、`DESIGN_C.md`：方案级数学原型与实验计划书。
  - 这类文档属于研究设计资产，不属于 `output/` 实验结果目录。

- `output/`
  - 规范化输出目录。
  - 约定：
    - `output/<stage>/OVERVIEW.md`：阶段总览
    - `output/<stage>/<attempt>.csv`：该尝试的结果汇总
    - `output/<stage>/<attempt>.md`：该尝试的结论说明与后续计划
    - `output/tmp/`：详细实验结果、中间文件、临时文件、缓存、日志
    - `output/<special>/`：重要基线或需长期保留的大型结果目录
    - `output/design_a/`、`output/design_b/`、`output/design_c/`：对应设计方案未来的实验结果目录
  - 当前专项约定：
    - `output/s2_stage/`：`S2` 的 special baseline root
    - 用于存放需要被其他 `S2` 脚本复用的完整阶段运行目录，而不是单个汇总文件
    - 其中通常包含：
      - `bonn_slam/slam/...`、`tum_oracle/oracle/...`
      - `tables/reconstruction_metrics.csv`
      - `tables/dynamic_metrics.csv`
      - `summary.json`
      - `surface_points.ply`、`rear_surface_points.ply`
      - `trajectory.npy`
    - 典型用途：
      - 作为 `control_root`、`donor_root`、`reference_root`
      - 保存 `111`、`116`、`125` 一类需要被后续 runner 继续消费的完整基线结果

- `archives/`
  - 历史冻结归档目录。
  - 仅保留需要长期保存的冻结快照或任务书归档。

- `assets/`
  - 静态图表与展示资源目录。
  - 不再用于堆积临时实验产物。

- `third_party/`
  - 外部依赖或第三方子模块。
  - 保持独立，不与主线/实验代码混放。

## 4. 后续开发规范

- 每个阶段尝试必须在 `output/<stage>/` 下保留两个文件：一个结果汇总 `.csv`，一个结论/未来计划 `.md`。
- 阶段总览必须写在 `output/<stage>/OVERVIEW.md`。
- 新实验脚本必须放在 `experiments/<stage>/`。
- 新实验性方法必须先放在 `experiments/<stage>/`，不得直接写入 `egf_dhmap3d/`。
- 新的详细结果、缓存、日志、中间文件必须放在 `output/tmp/`。
- 重要基线或需长期保留的大型结果可以在 `output/` 下单独开目录存放。
- 理论设计文档应放在根目录，不应放入 `output/`。
- 不再使用 `process/` 目录，也不得再将阶段结果长期堆积到 `processes/` 或根目录单一大文件。

## 5. 验证要求

- 所有迁移后的实验脚本必须至少通过导入验证。
- 主线入口脚本必须通过 `--help` 或等价 smoke test。
- 若涉及目录或路径调整，必须同步更新：
  - `README.md`
  - `TASK_LOCAL_TOPTIER.md`
  - 对应的 `output/<stage>/OVERVIEW.md`
