# S2 Comp-R Gap Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`
基线：`111_native_geometry_chain_direct`

## 1. Bonn all3 缺口概览

- 家族均值 `Comp-R = 70.86%`，距离 `98%` 仍差 `27.14 pts`
- 缺失参考点总数约 `2623`
- rear completion 总点数仅 `90`，对缺失区域的覆盖比仅 `3.43%`
- `104 -> 111` 的 `Comp-R` 增益约 `0.000 pts`，几乎为零

## 2. 序列级缺口画像

| sequence | comp_r_5cm | missing_refs | weak_support_missing | high_occlusion_missing | depth_hole_missing | rear_points |
|---|---:|---:|---:|---:|---:|---:|
| rgbd_bonn_balloon2 | 83.57% | 493 | 75.1% | 39.4% | 22.1% | 21 |
| rgbd_bonn_balloon | 70.23% | 893 | 76.1% | 16.0% | 21.3% | 6 |
| rgbd_bonn_crowd2 | 58.77% | 1237 | 72.4% | 29.2% | 24.7% | 63 |

## 3. crowd2 专项诊断

- 最差序列是 `rgbd_bonn_crowd2`，`Comp-R=58.77%`
- 缺失点中，`72.4%` 只得到 `<=2` 次稳定观测，说明当前管线不会把弱证据累积成稳定表面。
- 缺失点中，`29.2%` 呈现遮挡主导模式，说明动态前景干扰仍是重灾区。
- `missing_never_in_view_ratio = 0.0%`，说明在当前 5 帧参考集内，缺失区域并不是纯粹的传感器盲区；它们大多至少进入过视锥。
- `rear_points = 63`，相对于 `1237` 个缺失参考点仍然太稀疏，无法改变整体 completeness。

## 4. 对当前 Rear Completion 的结论

- 当前 rear completion 不是全局缺口修复器，而是局部 TB 恢复器。
- 它能恢复一小部分有 donor/support plane 的后景点，但不能把大量弱观测区域转成稳定表面。
- 从 `104 -> 111` 几乎零 `Comp-R` 增益可以确认：当前 completion 没有真正触达 completeness 主缺口。

## 5. 物理特征判断

- 当前缺失地图的主特征不是“完全没进视野”。
- 主特征是“进过视野，但观测次数低、且经常被前景截断”。
- 因此下一轮不应优先做更激进的平面补洞，而应优先做：
  1. 位姿受约束的多视图弱证据累积
  2. 动态边界附近的观测保真与重投影一致性
  3. 让 one/two-view 弱支持也能形成可提交的背景假设

## 6. 结论

- `Comp-R` 主缺口已锁定为：`weak support accumulation failure under dynamic occlusion`。
- 若不先解决弱观测累积与 pose consistency，继续加局部 rear completion 点数也无法把 `70.86%` 推近 `98%`。
