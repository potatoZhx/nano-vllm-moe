# 35. 撤回压缩权重路线并锁定精确权重约束

日期：2026-08-22

## 一句话总结

按用户约束，后续优化禁止 Q8/Q4/INT8/FP8 等权重压缩或量化；仓库未加入任何压缩权重
运行时代码、配置或 preset，隔离分析脚本已删除，当前 F16 single-weight 最优路径不变。

## 边界

上一轮只在 `/tmp` 下用正式 KTransformers 扩展做了只读可行性微基准，尚未修改 Nano 或
KTransformers 运行时。用户明确“不能压缩权重”后立即终止，没有进入实现和端到端 TPOT
门禁，也没有产生需要回滚的仓库代码。

从本提交起，所有可保留候选必须满足：

- expert 权重保持现有未压缩 F16/BF16 表示，不新增量化误差；
- 当前 canonical 路径继续使用 F16 single-weight；
- CPUInfer 总线程固定 16，双 NUMA `2 x 8`，其它资源继续与 baseline 对齐；
- operator 优化仅允许等价的调度、分块、访存、软件预取、临时缓冲复用或融合；
- GPU cache 容量不能通过压缩权重扩大。

## 后续顺序

剩余高价值方向调整为：

1. 在 F16 同源 llamafile kernel 内减少中间 buffer 写回、改善 NUMA-local 分块和软件预取；
2. 减少 CPU exposed routes，但 admission 必须预测 next reuse / tail saved，不能简单混合
   LFU/LRU；
3. 在现有动态 K 框架中补足可观测的 K2 边际成本或 alpha2 信息；
4. profile 证明主路径有收益后，再做不改变权重格式的 GPU 小核/控制面融合。

当前推荐仍为：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse
```

公平 t16 已测最低点仍为 **52.566035 ms/token**。
