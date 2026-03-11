# S2 upstream geometry analysis

日期：`2026-03-11`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_UPSTREAM_GEOMETRY_COMPARE.csv`

## 1. 上游参数是否有响应
- `99` 参考行：Bonn `Acc=4.346 cm`, `Comp-R=70.08%`, `TB=39`。
- `depth_bias=-1 cm`：Bonn `Acc=4.238 cm`, `Comp-R=70.86%`，平面距离 `0.0980 -> 0.0746`。
- `depth_bias=+1 cm`：Bonn `Acc=4.238 cm`, `Comp-R=70.81%`，平面距离 `0.0980 -> 0.0983`。

## 2. 结论
- `depth_bias` 确实让 Bonn `Acc` 有响应，且负偏置会明显降低平面误差，说明存在可测的系统性前侧偏置。
- 但这种响应仍然远不足以把 `Acc` 压到 `3.10 cm`；同时真实 upstream rerun 没有保住 downstream rear branch（`TB=0`），说明当前可执行上游前驱与 `99` 下游 completion 链之间存在结构断裂。
- 因此当前瓶颈更像是“上游几何偏置 + 下游 rear completion 失联”的组合问题，而不是单纯传感器噪声极限。
- `S2` 仍未通过，且绝对不能进入 `S3`。
