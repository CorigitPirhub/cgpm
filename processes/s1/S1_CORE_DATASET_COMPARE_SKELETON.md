# S1 核心数据集 compare 骨架

| family | method_line | tum_walking_xyz | tum_walking_static | tum_walking_halfsphere | bonn_balloon | bonn_balloon2 | bonn_crowd2 | static_sanity | current_status | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | EGF | canonical | canonical | canonical | canonical | canonical | canonical | canonical | available | Current active mainline and canonical tables already exist. |
| classical | TSDF | canonical | canonical | canonical | canonical | canonical | canonical | canonical | available | Bottom-line baseline already in local benchmark. |
| classic_dynamic | DynaSLAM | local_smoke | local_smoke | local_smoke | not_in_current_local_runner | not_in_current_local_runner | not_in_current_local_runner | not_in_current_local_runner | partial | Current local runner is TUM-oriented; Bonn remains future extension. |
| recent_dynamic_dense | RoDyn-SLAM | local_smoke_eval | local_smoke_eval | local_smoke_eval | local_smoke_eval | local_smoke_eval | local_smoke_eval | local_smoke_eval | available | Core-dataset protocol checks completed in S1. |
| representation_neural | NICE-SLAM | real_external_partial | not_yet | not_yet | not_yet | not_yet | not_yet | not_yet | partial | Existing real external chain is currently strongest on TUM walking_xyz only. |
| recent_2025_dynamic_dense | 4DGS-SLAM | planned | planned | planned | planned | planned | planned | planned | frozen_target | 2025 baseline frozen into RB-S1+; local integration moves into subsequent stage extension. |
