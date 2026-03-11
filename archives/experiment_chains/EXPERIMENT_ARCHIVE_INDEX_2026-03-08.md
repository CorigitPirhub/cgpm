# 局部实验链归档索引（S0）

日期：`2026-03-08`
作用：对当前大量历史局部实验目录进行“逻辑归档”，明确哪些目录属于 canonical / active / archived diagnostics。

## A. Canonical / Active

以下目录或文件属于当前 active / canonical 链：
- `output/summary_tables/paper_main_table_local_mapping.csv`
- `output/summary_tables/local_mapping_main_metrics_toptier.csv`
- `output/summary_tables/dual_protocol_multiseed_significance.csv`
- `output/summary_tables/local_mapping_precision_profile.csv`
- `output/freeze_snapshots/S0_2026-03-08_summary_tables/`

## B. Historical diagnostics (logically archived)

以下目录整体保留在磁盘，但在投稿治理语境下视为“历史/诊断实验链”，不得直接作为主结论来源：
- `output/post_cleanup/p10_*`
- `output/post_cleanup/_debug*`
- `output/post_cleanup/_pfv*`
- `output/post_cleanup/_reorg*`
- 其他单次 probe / smoke / ablation 子目录

## C. Archive rule

- 物理目录保留：便于追踪负结果与复现实验路径；
- 逻辑语义归档：这些目录中的单次实验结果**不得**覆盖 canonical 主表；
- 如果某条历史实验需要进入正文，必须先升级为：
  - 固定协议
  - 固定对比对象
  - 固定 seed / 多 seed
  - 进入 canonical 刷新链路

## D. Current active mainline experiment direction

当前 active 实验主线应仅包括：
- `P10` 主线中的 `write-time target synthesis`
- `Acc` 主线强化
- 强 literature baselines 补齐
