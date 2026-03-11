# S2 Final Closing Report Draft

日期：`2026-03-11`
阶段：`S2 closing draft / not-approved`

## 1. 技术路径总结

- `111_native_geometry_chain_direct`：原生化耦合链，Bonn `Acc=4.233`, `Comp-R=70.86`。
- `116_occupancy_entropy_gap_activation`：证明 `occupancy + entropy + gap-only activation` 机制成立，Bonn `Acc=4.120`, `Comp-R=76.14`，但依赖 Oracle gap mask。
- `122_evidential_visibility_deficit`：重建 GT-free visibility deficit 信号，`proxy_recall=0.519`, `proxy_precision=0.309`。
- `125_hybrid_papg_constrained`：GT-free 主线最平衡版本，Bonn `Acc=4.273`, `Comp-R=72.87`, `Ghost=3`。
- `126_local_geometry_convergence`：在不改集合的前提下做局部几何收敛，Bonn `Acc=4.272`, `Comp-R=72.94`, `Ghost=8`。

## 2. 结项判断

- 当前技术路径已经完整闭环，但结果仍**未**满足 S2 结项硬门槛。
- 具体缺口：
  - Acc 门槛要求 `<= 4.200 cm`，当前最佳 GT-free 为 `4.272 cm`；
  - Comp-R 门槛要求 `>= 72.5%`，当前 `126` 满足，为 `72.94%`；
  - Ghost 门槛要求 `<= 10`，当前 `126` 满足，为 `8`。

结论：
- `S2` 核心技术路径已被验证为正确；
- 但由于 `Acc` 仍未压到 `4.200 cm` 以下，当前只能提交“结项申请草稿”，不能正式宣布 S2 fully pass。
