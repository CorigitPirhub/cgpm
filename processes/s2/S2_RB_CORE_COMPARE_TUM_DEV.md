# S2 RB-Core 对比表（TUM dev quick）

协议：`TUM / oracle / rgbd_dataset_freiburg3_walking_xyz / frames=5 / stride=3 / seed=7`

| method | acc_cm | comp_r_5cm | notes |
|---|---:|---:|---|
| `egf_s2_anchor_ultralite_noroute` | 2.724 | 98.90 | Current best S2 iterate candidate. |
| `tsdf` | 0.374 | 0.93 | Same-protocol quick baseline from `00_no_synthesis_rps`. |
| `dynaslam` | n/a | 0.07 | Same-protocol local adapter result is extremely weak on this quick subset. |
| `rodyn_slam` | partial | partial | Current same-protocol rerun was blocked by checkpoint-path mismatch in ad-hoc compare script; see S1 core protocol check for runnable status. |

结论：
- 在当前 `TUM dev quick` 上，`egf_s2_anchor_ultralite_noroute` 仍未被 `RB-Core` 中已可对齐的 baseline 压制；
- 但由于 `rodyn_slam` 的本次 ad-hoc quick compare 未完成同口径补齐，这张表只能视为 **partial RB-Core compare**，不能支撑 `S2` 通过。
