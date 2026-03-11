# S1 第一轮方法筛选记录

## 候选方向

### Candidate A
- 名称：`downstream tri-map/export continuation`
- 目标：继续在 delayed/export 末端提升指标
- 当前判断：`abandon`
- 原因：近期多轮验证已显示该线边际收益衰减，主要产生极小 mixed change。

### Candidate B
- 名称：`delayed-branch persistent bank / readout continuation`
- 目标：通过 delayed readout / bank 增加 delayed 分支信息量
- 当前判断：`abandon`
- 原因：`dedicated banked readout` 与 `persistent bank accumulation` 在 focused probe 上几乎等价，增益不足。

### Candidate C
- 名称：`delayed-branch write-time target synthesis`
- 目标：让 delayed branch 在写入期获得 delayed-specific geometry hypothesis
- 当前判断：`accept`
- 原因：当前证据最强，且它针对的是 delayed 主线最上游、最可能仍有信息增益的瓶颈。

## S1 唯一 active candidate
- `delayed-branch write-time target synthesis`

## 原因总结
- 当前 downstream export-side trick 基本已验证到边际收益衰减区；
- 若要继续沿 delayed 主线推进，最有希望的新信息增益来自 write-time target，而不是继续读写/导出末端修补。
