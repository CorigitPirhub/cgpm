# S1 实验链充分性评估

## 当前判断
结论：**只有 3–4 条 baseline 的对比，不足以直接支撑最终顶刊/顶会水准。**

## 原因
- `TSDF` 只能给出 classical floor；
- `DynaSLAM` 代表 classic dynamic RGB-D，但年代偏早；
- `RoDyn-SLAM` 虽然更近，但仍不足以单独代表 2024/2025 dynamic dense 全谱系；
- 若缺少 representation / neural dense 与 2025 recent dynamic dense / 4DGS 线，审稿时很容易被质疑 baseline matrix 不充分。

## S1 是否还能完成
可以，但条件是：
- `RB-Core` 必须完成本地闭环；
- 核心数据集必须完成 protocol check；
- `RB-S1+` 必须冻结，明确：
  - `NICE-SLAM` 作为 representation/neural dense 线；
  - `4DGS-SLAM` 作为 2025 recent dynamic dense 线。

## 对后续阶段的含义
- `S1` 通过并不等于“最终投稿实验链已充分”；
- `S1` 通过只意味着：
  - 主命题收敛完成；
  - baseline floor 建立完成；
  - 后续 `S2-S5` 可在统一口径上继续扩展 baseline matrix。

## 说明修正（2026-03-09）
- `S1` 的“实验链充分性判断”仍然成立；
- `S2` 暴露出的 current-code downstream drift，不会推翻 `S1` 已完成 baseline floor / protocol closure 的事实；
- 但它说明：`S1` 交接给 `S2` 的 active direction，后续仍需在 current-code canonical 上重新建立有效性。
