# 27. 短期 ghost cache 保护

日期：2026-08-21

> **公平性复测（2026-08-22）：** 保留 preset 的 canonical 名称改为
> `k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8`。统一 16 total threads / 双 NUMA
> 2 x 8 后，b1 **54.627249** → ghost8 **53.336804 ms/token**，改善 **2.36%**。下文
> 2026-08-21 数值是旧 t32 资源下的历史门禁；完整复测见
> [`20260822_30_uniform_t16_fairness_revalidation.md`](20260822_30_uniform_t16_fairness_revalidation.md)。

## 一句话总结

对“换出后 8 个 model step 内又被传回”的 expert 提供 8-step 保护，同日一请求
TPOT 从 57.294 降到 54.393 ms/token（-5.06%），因此保留独立 `ghost8` preset。

## 动机与参数选择

当前 b1 instrumented profile 有 2531 次 source-tracked publication，约传输
22.24 GiB。232 次“未消费即换出”中，176 次（75.86%）后来又被传回，重传间隔中位数
只有 11 model steps。这说明单轮保护结束后，LRU/LFU 仍会反复淘汰短期会再次需要的对象。

在不运行新端到端候选的前提下，对已有 b1 lifecycle trace 做了 shadow sweep：

| ghost window / protect TTL | ghost hit | 改变 victim 的机会 | 占已有 victim 决策 |
|:---:|---:|---:|---:|
| 4 / 4 | - | 23 | 0.9% |
| **8 / 8** | **350** | **59** | **2.3%** |
| 16 / 16 | - | 129 | 5.1% |

8/8 是保守点：能够覆盖明确的短期重传，但只影响约 2.3% 的 victim 决策，避免 16/16
对当前 LRU/LFU 策略施加过强干预。350 个 ghost hit 按来源分为 verify 271、phase1 44、
draft 35；shadow 中被保护 victim 的来源为 verify 43、phase1 14、draft 2。

## 实现

新增 opt-in 配置：

```text
predictive_ghost_window_steps=8
predictive_ghost_protect_steps=8
```

Predictive runtime 在一次成功 publication 后记录它替换的 `(layer, expert)`；若该 expert
在 window 内重新 publication，则将它放入短期 hot 集合。victim selector 把仍在 TTL 内且
当前 resident 的 expert 与既有 round/rankguard 保护合并；当所有可用 slot 都受保护时，
继续使用原来的 LRU/LFU safety valve，保证预取不会停滞。

状态上限由模型的 `(layer, expert)` 数量约束；选择 victim 时只遍历该层现有 active slots，
不扫描 lifecycle metadata。两个参数默认为 0，所以此前全部 preset 的行为保持不变。

保留的独立 preset：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8
```

原 `..._b1` 是关闭 ghost 的直接 fallback；budget2、active14 与 full-context-safe 动态
preset 也都未覆盖。

## 同日一请求门禁

共同口径：MMLU-Pro validation sample 0、seed 20260719、temperature 0.6、107-token
prompt、固定生成 512 token、single-weight llamafile F16、动态 K1/K2、profile 全关。

| 配置 | TPOT | decode tok/s | decode steps | 平均 step wall | 校验 |
|:---|---:|---:|---:|---:|:---:|
| b1 baseline | 57.293907 ms | 17.454 | 267 | 109.652 ms | 512/512 |
| b1 + ghost8 | **54.392560 ms** | **18.385** | 263 | **105.683 ms** | 512/512 |

相对同日 baseline：TPOT **-5.06%**，decode throughput **+5.33%**，平均 step wall
**-3.62%**。结果目录：

```text
results/tpot_phase1_b1_baseline_20260821/
results/tpot_phase1_b1_ghost8_20260821/
```

两次运行的 output digest 不同；在随机采样下，round 数和 route 轨迹也随输出分叉。因此
5.06% 是符合既定规则的真实一请求端到端正收益门禁，不是 ghost 策略的隔离因果估计。
候选的 54.393 ms/token 也没有刷新跨日期绝对最低点：历史 b1 单点仍为
53.725789 ms/token。相比跨日期绝对数值，同日直接 A/B 更适合决定是否保留本候选。

## 验证

```text
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_predictive_prefetch.py \
  tests/test_eval_tpot_config.py \
  tests/test_config_predictive_prefetch.py
```

覆盖 ghost hit/TTL、TTL 到期恢复原 victim、默认关闭时行为不变，以及独立 preset 和
benchmark metadata 传递。端到端请求通过 fixed-length/output validation。

## 后续

下一项按预期收益排序转向 transfer/publish/LUT 批量化：当前请求已有 2531 次、约
22.24 GiB publication，控制面存在大量 event query、Python commit 和小 GPU LUT 更新。
ghost window/TTL 16 暂不继续扫描；它影响约 5.1% victim，风险显著高于 8/8，且当前
更需要机制不同的增益来源。
