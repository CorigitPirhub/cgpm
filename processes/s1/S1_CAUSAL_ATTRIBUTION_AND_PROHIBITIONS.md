# S1 因果归因与禁止事项清单

## A. 归因成立条件
任何阶段若要宣称“收益来自某方法/模块设计”，必须同时满足：
- 固定 protocol 不变
- 固定 seed 集不变
- 固定 frames / stride 不变
- 固定 canonical 刷新链不变
- 对照组只改变该模块或其直接依赖项
- 有至少一个 controlled compare 表

## B. 绝对禁止
- 混用 `oracle / slam` 形成单一主结论
- 在 `slam` 中引入 GT delta 或未来信息
- 使用非 canonical 主表或历史 probe 直接覆盖主结论
- 横向拼接不同 frame/stride/seed/protocol 的“最好结果”
- 多模块同时漂移后宣称单模块有效
- 用最终锁箱集反复调参
- 基于测试集结果反向改阈值并再次作为最终结论

## C. 当前 S1 判定原则
- 若 `RB-Core` 不能本地运行或口径不能对齐，则 `S1` 不得判定完成
- 若 active candidate 不能在开发协议上形成初步净优势，则 `S1` 不得进入 `S2`
