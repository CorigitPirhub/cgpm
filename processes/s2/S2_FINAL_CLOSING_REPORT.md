# S2 Final Closing Report

日期：`2026-03-11`
阶段：`S2 final closing / not-pass`

## 1. 完整技术路径

- `111`：原生主线基线，Bonn `Acc=4.233`, `Comp-R=70.86`。
- `116`：Oracle 上界，Bonn `Acc=4.120`, `Comp-R=76.14`。
- `122`：GT-free visibility deficit，Bonn `Acc=4.391`, `Comp-R=74.10`, `proxy_recall=0.519`。
- `125`：GT-free 主线平衡版，Bonn `Acc=4.273`, `Comp-R=72.87`, `Ghost=3`。
- `126`：局部几何收敛，Bonn `Acc=4.272`, `Comp-R=72.94`, `Ghost=8`。
- `129`：局部配准偏差建模，Bonn `Acc=4.280`, `Comp-R=72.29`, `Ghost=4`。

## 2. 最终结论

- S2 Final Boss 门槛：
  - `Acc <= 4.200 cm`
  - `Comp-R >= 72.5%`
  - `Ghost <= 10`

- 当前最佳 GT-free 结果仍为 `126`：Acc=`4.272`, Comp-R=`72.94`, Ghost=`8`。

判定：
- `Comp-R` 达标（以 `126` 为准）；
- `Ghost` 达标；
- `Acc` 未达标。

因此：
- `S2` 核心技术路径已完整验证；
- 但 GT-free 路线仍停在约 `4.25~4.27 cm` 的性能天花板附近；
- **S2 不通过，禁止进入 S3。**

## 3. 后续研究方向

- 若继续提升 Acc，需要引入更强的 SLAM 前端几何约束，或更换当前场景补全范式；
- 再继续对当前 S2 主线做局部补丁，预期收益已极低。
