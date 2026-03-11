# S1 主命题摘要（3句版）

我们关注动态场景中的局部建图核心矛盾：静态几何精度（`Acc`）与动态污染抑制（`ghost`）在传统后置删点框架中天然耦合。
本项目主张在状态层而非提取层进行解耦：通过 `evidence-gradient + dual-state` 建立静态/动态分离状态，并用 `delayed branch` 隔离冲突几何样本。
最终研究目标是让 delayed branch 在写入期就拥有 delayed-specific 几何假设（`write-time target synthesis`），从而同时推进 `Acc / Comp-R / ghost suppression`。
