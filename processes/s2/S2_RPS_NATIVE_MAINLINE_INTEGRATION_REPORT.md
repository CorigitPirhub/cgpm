# S2 native mainline integration report

日期：`2026-03-11`
目标：验证原生标准管道与 `108/109` 的一致性。

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_tb | target_acc_cm | abs_diff_acc_cm | decision |
|---|---:|---:|---:|---:|---:|---|
| 111_native_geometry_chain_direct | 4.233 | 70.86 | 39 | 4.233 | 0.000 | match |
| 112_native_geometry_chain_projected | 4.235 | 70.82 | 31 | 4.233 | 0.003 | match |

重构落点：
- `egf_dhmap3d/P10_method/geometry_chain.py`
- `egf_dhmap3d/core/config.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

结论：
- 两个原生变体都满足 `abs_diff_acc_cm < 0.01`；
- `111_native_geometry_chain_direct` 在最新重跑中更稳，`Comp-R/TB` 优于 `112`；
- `scripts/run_s2_rps_geometry_chain_coupling.py` 现仅保留为诊断脚本，不再作为主入口。
