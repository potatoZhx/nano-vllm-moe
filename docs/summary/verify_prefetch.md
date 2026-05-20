目标

  实现 verify 阶段逐层 prefetch：在 verify 执行第 i 层计算时，预测并传输第 i+1 层可能需要的专家，目标是降低后续
  verify 层的专家缺失带来的计算延迟。实现时保证两点：不等待 prefetch 完成、不改变推理结果；只在估计传输可被当前层计算完全覆盖时提交传输。
  实现verify加速

  设计

  核心策略是 direct-active prefetch：

  - verify 第 i 层开始前，先提交第 i+1 层的专家预取。
  - 因为第 i 层和第 i+1 层使用不同 layer cache，传输目标不会和当前正在计算的专家冲突。
  - 预取直接写入目标层 active cache slot，不经过 staging publish 的额外拷贝。
  - 每层计算耗时用 CUDA event 采样并维护 EMA；warmup 阶段会先跑一次 verify-like forward 来初始化这些数据。
  - 可用传输窗口为 layer_i_compute_ms_ema * prefetch_verify_layer_safety_ratio。
  - 单个专家传输耗时按权重字节数和配置带宽估计：bytes / prefetch_verify_layer_transfer_bandwidth_gbps。
  - 提交数量同时受 available_ms、prefetch_verify_layer_max_budget、prefetch_step_budget、prefetch_max_inflight 限制。
  - hook 内只 query() ready event 并提交异步 copy，不同步等待，因此不会把传输暴露到 verify 关键路径。

  正确性保护：

  - direct-active 预取在 reservation 时会先把被替换 slot 的旧专家映射失效。
  - 如果传输未完成就进入目标层，该专家不会被认为 cached，verify 会走原有 CPU/GPU fallback 精确计算路径。
  - 只有 event ready 后才 commit 新专家映射。
  - 因此 late prefetch 最多损失一次 cache hit，不会读到半传输权重，也不会改变 logits。

  代码实现

  配置项新增在 nano-vllm-moe/nanovllm/config.py:48：

  - prefetch_verify_layer_enabled=True
  - prefetch_verify_layer_safety_ratio=0.8
  - prefetch_verify_layer_min_compute_ms=0.05
  - prefetch_verify_layer_transfer_bandwidth_gbps=12.0
  - prefetch_verify_layer_max_budget=2
  - 同时加入参数合法性 assert。

  active slot reservation 和 direct-active commit 在 nano-vllm-moe/nanovllm/expert/cache.py:25：

  - 新增 ActiveReservation。
  - 新增 active_slot_pending_expert，标记正在异步写入的 active slot。
  - 新增 reserve_active_slot_for_prefetch()、begin_async_put_to_active()、commit_active_prefetch()。
  - put_to_slot() / staging publish 会清理 pending 标记，避免状态残留。

  prefetch runtime 主逻辑在 nano-vllm-moe/nanovllm/expert/prefetcher.py:413：

  - 新增 submit_verify_layer_prefetch(step_id, target_layer_idx, available_ms)。
  - 从 global warm-start queue 中筛选目标层候选专家。
  - 用估计传输耗时决定是否提交。
  - 直接向目标层 active slot 发起异步 copy。
  - 新增 publish_direct_active_ready()，只发布已 ready 的 direct-active ticket。
  - staging publish 的 victim 选择也会跳过 pending active slot，避免并发覆盖。

  模型层 hook 在 nano-vllm-moe/nanovllm/models/qwen3_moe.py:652：

  - Qwen3MoeModel 增加 verify_prefetch_controller。
  - 每个 decoder layer 前调用 before_verify_layer(layer_idx)，后调用 after_verify_layer(layer_idx)。
  - Qwen3MoeForCausalLM 暴露 set_verify_prefetch_controller() 给 runner 使用。

  runner 集成在 nano-vllm-moe/nanovllm/engine/model_runner.py:725 和 nano-vllm-moe/nanovllm/engine/
  model_runner.py:1201：

  - warmup_model() 后新增 _warmup_verify_layer_timings()，按用户建议在 warmup 收集 verify 层耗时数据。
  - run_verify() 期间临时挂载 controller，结束后恢复。
  - before_verify_layer() 负责发布已完成 direct-active ticket、估算当前层可覆盖传输时间、提交下一层 prefetch、记录本
    层 timing start。
  - after_verify_layer() 记录 timing end 并更新 EMA。
  - get_profile() 会先 poll pending timing event，确保 profile 尽量包含最新采样。

  另外做了两个相关正确性修复：

  - nano-vllm-moe/nanovllm/layers/fuse_moe/heterogeneous.py:440：route buffer cache 现在检查 hidden_dim，避免只按
    numel 复用导致 shape mismatch。
  - nano-vllm-moe/nanovllm/layers/fuse_moe/cpu_backend.py:533：fused CPU backend 的相关路径统一到 F.linear + act_fn 计算形式，和 torch backend 精度对齐，符合 docs/spec_determinism_fix_20260506.md 的确定性要求。

  测试情况

  已通过本地 targeted suite：

  pytest -q tests/test_prefetch_runtime.py tests/test_prefetch_wait.py tests/test_verify_feedback.py tests/
  test_model_runner_prefetch.py tests/test_config_prefetch.py tests/test_spec_engine_prefetch.py tests/
  test_spec_engine_flow.py tests/test_prefetch_runtime_meta.py tests/test_prefetch_global_queue.py tests/
  test_prefetch_strategy.py tests/test_model_runner_spec_modes.py tests/test_llm_engine_mode_dispatch.py tests/
  test_mode_config.py tests/test_spec_engine_basic.py tests/test_expert_cache_staging.py tests/
  test_cache_strategy.py

  结果：43 passed, 3 skipped in 8.53s。

  新增/更新的单测覆盖：

  - nano-vllm-moe/tests/test_prefetch_runtime.py:88：verify layer direct-active prefetch 提交、pending 状态、ready
    后 publish。
  - nano-vllm-moe/tests/test_model_runner_prefetch.py:57：warmup verify timing 能初始化 per-layer EMA。
  - nano-vllm-moe/tests/test_config_prefetch.py:20：新增配置默认值和非法带宽校验。

  A100 上也跑过关键正确性测试：

  pytest -q tests/test_cpu_moe_correctness.py::TestCpuMoeCorrectness::test_fused_backend_matches_torch_backend
  tests/test_cpu_moe_correctness.py::TestCpuMoeCorrectness::test_torch_packed_backend_matches_torch_backend tests/
  test_prefetch_runtime.py tests/test_model_runner_prefetch.py tests/test_config_prefetch.py tests/
  test_spec_engine_prefetch.py

  结果：14 passed in 28.13s。日志在 /home/mumura/moe_spec/logs/
  salloc_fused_backend_exact_after_refpath_20260518_220332.log。

  待测试任务

  还没有跑完整端到端性能 benchmark。建议下一步重点测：

  - verify latency 对比：关闭/开启 prefetch_verify_layer_enabled，看 verify forward、expert miss、CPU fallback 时间
    变化。
  - 不同 cache ratio 下收益：尤其专家 cache 紧张时 direct-active 替换是否稳定收益。
  - 调参：prefetch_verify_layer_transfer_bandwidth_gbps、safety_ratio、max_budget 对隐藏率和命中率的影响。
  - 长序列、多 batch、多 draft token 的端到端 spec decode 正确性和吞吐。
  - profile counter 检查：verify_layer_prefetch_submit_count、publish_count、budget_stop_count、used_budget_ms 是否
    符合预期。
  - fused CPU backend 全量 correctness suite 和真实模型 logits 对齐长跑。