# Verify Miss Policy Validation

| policy | ratio | out | max draft | accept | cache hit | output tok/s | decode tok/s | draft ms | verify ms | graph replays | prefetch submit/done/used | verify-layer submit/done/used | cache-fill promoted/cpu/evicted | text |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|:---|:---|
| cpu | 0.25 | 128 | 3 | 0.6119 | 0.7103 | 2.949 | 3.480 | 24.533 | 742.240 | 134/134 | 4644/4644/12349 | 4205/4205/10549 | 0/0/0 | ok |
| cpu | 0.50 | 128 | 5 | 0.8803 | 0.9092 | 8.125 | 10.624 | 21.008 | 396.438 | 117/117 | 1804/1804/5516 | 1282/1282/3342 | 0/0/0 | ok |
| cpu | 0.75 | 128 | 7 | 0.8934 | 0.9712 | 13.764 | 17.641 | 21.412 | 254.229 | 122/122 | 1036/1036/2632 | 648/648/1318 | 0/0/0 | ok |
| cpu | 0.25 | 512 | 3 | 0.6369 | 0.7508 | 3.750 | 3.904 | 24.403 | 669.606 | 526/526 | 18252/18252/44471 | 16381/16381/37918 | 0/0/0 | ok |
| cpu | 0.50 | 512 | 5 | 0.9292 | 0.9405 | 10.225 | 10.917 | 21.986 | 402.458 | 452/452 | 6412/6412/14175 | 4122/4122/7272 | 0/0/0 | ok |
| cpu | 0.75 | 512 | 7 | 0.9446 | 0.9897 | 20.292 | 22.489 | 20.786 | 187.727 | 469/469 | 2177/2177/4376 | 719/719/1324 | 0/0/0 | ok |
| cache_fill | 0.25 | 128 | 3 | 0.6434 | 0.9865 | 3.724 | 4.307 | 21.960 | 608.630 | 129/129 | 1900/1900/6221 | 1484/1484/4209 | 11143/0/11143 | ok |
| cache_fill | 0.50 | 128 | 5 | 0.8583 | 0.9852 | 9.149 | 11.755 | 21.573 | 342.562 | 120/120 | 1349/1349/3053 | 926/926/1888 | 1584/0/1584 | ok |
| cache_fill | 0.75 | 128 | 7 | 0.9244 | 0.9894 | 14.693 | 20.880 | 20.617 | 213.411 | 119/119 | 863/863/1758 | 458/458/996 | 256/0/256 | ok |
| cache_fill | 0.25 | 512 | 3 | 0.7532 | 0.9962 | 5.315 | 5.622 | 21.501 | 512.776 | 470/470 | 3454/3454/15188 | 1474/1474/4147 | 40817/0/40817 | ok |
| cache_fill | 0.50 | 512 | 5 | 0.9048 | 0.9961 | 13.826 | 15.214 | 21.837 | 250.058 | 462/462 | 2745/2744/7508 | 927/927/1794 | 4715/0/4715 | ok |
| cache_fill | 0.75 | 512 | 7 | 0.9716 | 0.9971 | 23.350 | 25.551 | 20.174 | 160.262 | 458/458 | 1503/1503/3203 | 475/475/1001 | 524/0/524 | ok |

## Generated Text Samples

### cpu_ratio25_l128

```text
The optimal storage strategy balances how many experts to keep in GPU vs CPU, and how to manage memory when the current workload requires more experts than available in GPU.

In this work, we analyze how different MoE routing algorithms affect the efficiency of expert caching. We propose a novel routing strategy that minimizes the required GPU memory while maintaining low routing error. Our experiments on a large-scale language model with 60B parameters show that our method reduces the required GPU memory by 35% compared to existing routing strategies, without sacrificing model accuracy.

I need to rephrase the above text into a more concise version that preserves the main points.
```

### cpu_ratio50_l128

```text
The main question is: how to efficiently manage this expert memory and routing to minimize latency while maintaining model accuracy.

The challenge is to design a routing strategy that minimizes the average number of expert transfers per token while also minimizing the deviation from the optimal routing (the ideal case where all tokens are routed to the same experts). This leads to a multi-objective optimization problem: reduce transfer frequency and maintain routing quality.

A naive greedy approach of always keeping the most frequently used experts in memory can lead to poor performance when the input distribution changes. For example, if the input shifts to a new domain, the cached experts may no longer be optimal,
```

### cpu_ratio75_l128

```text
The main question is: how to optimally manage the expert cache to minimize the total execution time, considering both GPU memory constraints and CPU/GPU data transfer costs.

The problem requires designing an efficient expert caching strategy that optimizes for both memory usage and transfer latency. Given a fixed GPU memory budget, the goal is to choose which experts to keep in GPU memory and when to cache/evict them to minimize the total execution time. The challenge is to find the optimal eviction policy that balances the trade-off between memory usage and transfer cost.

To solve this problem, I'll outline a caching strategy and analyze its theoretical guarantees. Let's formal
```

### cpu_ratio25_l512

```text
The optimal storage strategy balances how many experts to keep in GPU vs. CPU, and how to schedule expert loading.

Our main contributions are:
1. A novel expert cache eviction policy that leverages the locality of reference in the routing distribution, significantly outperforming traditional cache eviction strategies.
2. An efficient cache-aware expert scheduling algorithm that minimizes CPU-GPU data transfers while respecting the parallelism constraints of the GPU.
3. Comprehensive empirical evaluation across diverse tasks, demonstrating the effectiveness of our techniques in reducing inference latency and memory usage.

For the routing distribution, we have tested both per-token and per-batch routing. Per-token routing produces more diverse expert access patterns, which benefits from our locality-aware eviction policy. Per-batch routing is more predictable and suitable for static scheduling. We find that the optimal strategy depends on the specific workload and hardware constraints.

Our results show that with our cache management techniques, MoE models can achieve up to 2.5x speedup in inference time and 40% reduction in memory usage compared to baseline methods. These improve
```

### cpu_ratio50_l512

```text
The main question is: how to efficiently manage this expert memory hierarchy to minimize total execution time.

Our focus is on the problem of expert cache eviction, an important component of the memory management system. When the total number of experts is larger than the available GPU memory, the cache must evict some experts to make room for others. The goal is to choose which experts to evict such that the total computation time is minimized.

From the perspective of the execution model, the cost of evicting an expert is the time required to transfer its weights from CPU to GPU memory when it is next accessed. This transfer time is determined by the size of the expert's weights (s), the bandwidth (b) between CPU and GPU, and the number of times the expert is accessed during the execution (k). The total cost of evicting expert e is thus:

T_e = s_e * k_e / b

Our task is to find a set of experts to evict such that the total cost T = sum(T_e) is minimized, subject to the constraint that the total memory occupied by the active experts is less than or equal to the available GPU memory.

Let's define the problem more formally:

Let E be the set of all experts, with each expert e in
```

### cpu_ratio75_l512

```text
The main question is: how to optimally manage the expert cache to minimize the total execution time, considering both GPU memory constraints and CPU transfer delays.

There is a lot of open research in efficient MoE deployment. Recent work has explored dynamic expert loading, where only the most relevant experts are cached, and pre-fetching techniques to hide transfer latency. However, these approaches often assume static input distributions. In practice, the routing distribution can change over time, necessitating adaptive caching strategies.

You are tasked with designing a caching strategy that intelligently manages the expert cache, adapting to the dynamic nature of the routing distribution. 

### Problem Statement:

Given a sequence of input tokens, each requiring a set of expert sub-networks based on their routing distribution, your task is to design a caching strategy that minimizes the total execution time of the entire sequence, considering the following constraints:

1. **Memory constraint**: The cache can hold at most $ C $ experts.
2. **Transfer cost**: Loading an expert from CPU to GPU takes $ T $ units of time.
3. **Expert reuse**: If an expert is needed again in the
```

### cache_fill_ratio25_l128

```text
The main question is: what is the optimal strategy for managing which experts are cached in GPU memory?

This problem is similar to the classic "optimal caching" problem in computer architecture, but with two key differences: 1) the access pattern is dynamic and depends on the input tokens, and 2) there is a "cost" associated with loading an expert from CPU to GPU. The goal is to choose a subset of experts to keep in GPU memory (with size limited by available VRAM) such that the total cost (in terms of transfer latency) is minimized, while maintaining acceptable inference latency.

Several approaches have been proposed for
```

### cache_fill_ratio50_l128

```text
The main question is: how to efficiently manage this expert memory hierarchy to minimize the overall inference time?

The main contributions of this paper are:

1. Proposing a novel expert memory management framework called EMERGE (Expert Memory EfFicient ROuting and Execution) that optimizes the trade-off between memory capacity and transfer latency.

2. Introducing a hybrid caching strategy that combines static and dynamic expert caching. Static caching keeps frequently used experts in GPU memory, while dynamic caching adapts to current routing patterns.

3. Designing a lightweight expert scheduler that predicts future routing patterns and preloads experts into GPU memory before they are needed.

4
```

### cache_fill_ratio75_l128

```text
The main question is: how to optimally manage the expert cache to minimize the total execution time.

This problem is a classic instance of the "optimal caching" problem, which is NP-hard. However, MoE models have specific characteristics that can be exploited to design effective heuristics. The routing probabilities are typically smooth and not too skewed, and the same expert is often reused across tokens in the same batch.

From the perspective of the cache, each expert has a "reuse distance" - the number of other experts that need to be evicted before it's needed again. In a batch of tokens, the reuse distance for
```

### cache_fill_ratio25_l512

```text
The main question is: what is the optimal strategy for managing which experts are kept in GPU memory?

To address this, we propose an expert cache-aware routing strategy that actively manages the expert cache. Our approach combines three key components:

1. **Dynamic expert selection**: Instead of always selecting the top-K experts, we employ a dynamic selection mechanism that takes into account both the routing probabilities and the current state of the expert cache. This ensures that tokens are routed to experts that are currently available in GPU memory, when possible, to minimize transfer latency.

2. **Cache-aware routing**: The routing mechanism is modified to incorporate a cache state-aware loss function during training. This loss penalizes routing decisions that would result in frequent cache evictions or unnecessary transfers, thereby encouraging the model to learn routing strategies that are more cache-friendly.

3. **Eviction policy**: A cache eviction policy is implemented to manage which experts are evicted from the GPU memory when the cache becomes full. We explore different eviction policies, such as least recently used (LRU), least frequently used (LFU), and a nove
```

### cache_fill_ratio50_l512

```text
The main question is: how to efficiently manage this expert memory hierarchy to minimize the overall inference time?

The main contributions of this paper are:

1. Proposing a new routing algorithm that improves expert load balancing through a novel entropy regularization term. This leads to more uniform expert usage, enabling better utilization of available GPU memory.

2. Introducing a novel caching strategy that leverages the hierarchical memory structure of modern accelerators. This strategy dynamically moves expert weights between CPU and GPU memory based on their predicted future usage, minimizing transfer overheads.

3. Demonstrating that these two components together yield significant improvements in inference performance on standard benchmarks while maintaining the same level of model quality as the original MoE transformer.

The results show that with the new routing and caching strategy, the model achieves 2.1x faster inference on a 30B parameter MoE transformer compared to the baseline, with no degradation in quality. This represents a substantial improvement in the efficiency of large-scale MoE models.

The approach is general and can be applied to various MoE archite
```

### cache_fill_ratio75_l512

```text
The main question is: how to optimally manage the expert cache to minimize the total execution time.

This problem is a classic instance of the "optimal caching" problem, which is NP-hard. However, MoE models have specific characteristics that can be exploited to design effective heuristics. The routing patterns are dynamic but may have temporal locality – the same experts are reused within a short time window. Also, the number of active experts per token is typically small (e.g., 2-4), and the number of total experts is large (e.g., 100-1000). 

A promising approach is to use a combination of temporal locality and popularity-based caching. We can maintain a cache of recently used experts and also track the frequency of expert usage. When cache space is limited, we can evict the least recently used (LRU) or least frequently used (LFU) experts. However, this may not be sufficient due to the dynamic nature of routing patterns.

Another approach is to use a predictive caching strategy. By analyzing the routing patterns of the current batch, we can pre-fetch the experts that are likely to be needed in the next batch. However, this requires accurate prediction and may add overhead.

To
```

