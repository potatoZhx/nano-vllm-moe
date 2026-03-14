我有一个加速moe端侧单卡cpu-gpu协同推理的idea，实现为/zx_data1/sparsity/on_device_sd/demo，但与nano-vllm对比发现，实现结果即使在纯gpu推理的场景下都比nano-vllm慢十倍（on_device_sd/demo/examples/simple_speculative_speed_compare.py vs nano-vllm/examples/simple_speed_compare.py)；尝试优化后没有什么改进，因此，我fork了nano-vllm(nano-vllm-moe),想在它的基础上实现我的idea，请你先仔细阅读on_device_sd/demo项目，梳理其目标、架构、实现、流程、算法等，整理为改造目标的详细md文档写到nano-vllm-moe路径下；然后仔细阅读nano-vllm-moe/nanovllm，设计基于其实现on_device_sd/demo功能的设计文档；注意demo中有些功能需要留出优化接口; 在改造中，比较困难的可能是fused moe kernel，请仔细思考如何实现；实现文档也写到nano-vllm-moe路径下

/zx_data1/sparsity/nano-vllm-moe/docs/demo_analysis.md
1）"Qwen3-30B-A3B-Base（48 层，128 routed experts / 层，top-8 路由，1 shared expert）"请确认该模型是否有shared expert，我看config中好像没有提及; 即使这个模型没有，也需要能够兼容带shared expert的模型
2）4.4 的draft-phase中expert_transfers可能在每一层route results出来之后启动，也可能在draft one step之后启动，目前倾向于前者
3）4.5中Draft 期间的传输调度的“选取 top 若干个异步传输“需要补充：需要在推理引擎启动前根据expert传输速度和一层或one step的计算时长估算传输数量
4）4.7 KV Cache 与 Draft/Verify 切换中不太准确，每一轮draft-verify之后，会完全丢弃draft的kv cache，替换为verify阶段生成的接受token的kv cache；流程上，draft结束之后就会丢弃draft生成的KV Cache，然后verify接着draft之前的kv cache推理【上一轮prefill/verify生成的最后一个token+draft期间生成的token】，然后丢弃没被接受token的kv cache，被接受的最后一个token之后的一个token就是verify这一次生成的token 


migration_design.md
1）为了支持fused moe，是不是需要每一层的expert cache数量固定且在gpu内存中连续
2）由于目前在显存较大的机器上测试，为了模拟消费级显卡，需要接受一个设置可用gpu内存的大小（G为单位）的参数
3）expert_placement的初始支持从配置文件读取，如果没有传入配置文件，则每层固定数量，随机选取
4）4.2 异构参数加载器中，routed expert也认为是静态参数载入gpu，但如果fused moe需要连续内存不方便加载，也可以考虑在expert cache中pin住routed expert
5）4.6 def speculative_step中似乎有点问题，异步预取发生在draft中，预取的是根据draft激活预测的verify的experts，不是phase4；理想中这个传输会在verify的第一层attention结束前结束（但传输还需要考虑到不能和层的执行冲突，所以如果实现复杂的话也可以先采用每层attention结束时或draft结束时sync等）；另外self.expert_cache.complete_transfers()是在做什么呢？
6）4.7 中注意上面demo_analysis.md意见4）提到的kv cache切换流程
7）5. Fused MoE Kernel 适配中，策略C每次forward都需要创建临时buffer，且G的长度可能是变化的(比如动态的话kernel是不是需要重新编译），我有点担心开销会太大；我有一个不成熟的想法：根据可容纳的experts总数在最开始就为每层固定好GPU的expert数量，并分层expert cache，每层需要替换expert时只能从这一层中选择slot替换，这样就可以实现为每层expert连续存储了，是否就可以直接使用fused moe kernel了
8）对于8.3，当前不需要考虑TP 
9）10.1中是否需要加上算子正确性的测试（比如异构moe算子和原算子精度对齐）
以上是我的一些意见，请参考意见检查、修改文档