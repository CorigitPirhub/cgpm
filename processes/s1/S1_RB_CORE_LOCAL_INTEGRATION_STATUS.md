# S1 RB-Core 本地接入状态表

| baseline_family | source_state | protocol_aligned | smoke_status | notes |
|---|---|---|---|---|
| TSDF | `native_local` | `yes` | `runnable` | Local native baseline already integrated in benchmark framework. |
| DynaSLAM family | `native_local` | `yes` | `runnable` | Confirmed local TUM smoke/dev/lockbox runs succeeded via `scripts/external/run_dynaslam_tum_runner.py` + `scripts/adapters/run_dynaslam_adapter.py`. |
| RoDyn-SLAM family | `upstream_local_patched` | `yes` | `runnable` | Upstream repo at `third_party/rodyn-slam` is runnable in `cgpm` after local compatibility patches; core-dataset protocol checks are recorded in `processes/s1/S1_RODYN_CORE_PROTOCOL_CHECK.md`. |

补充说明：
- `StaticFusion` 已从当前正式 `RB-Core` 中归档移除，不再作为 `S1` blocking baseline；
- 当前 `RB-Core` 的 recent dynamic dense 替代线固定为：`RoDyn-SLAM`（preferred）/ `NID-SLAM`（fallback，当前未启用）。
