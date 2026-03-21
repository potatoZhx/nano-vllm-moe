import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    inference_mode: str = "standard"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    dist_port: int = 2333
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_heterogeneous: bool = False
    enable_speculative: bool = False
    max_draft_tokens: int = 8
    draft_top_c: int = 2
    acceptance_strategy: str = "greedy"
    acceptance_threshold: float = 0.7
    draft_scheduler: str = "simple"
    spec_verify_eager: bool = True
    spec_enable_prefetch: bool = False
    spec_profile: bool = False
    engine_profile: bool = False
    engine_profile_cuda_sync: bool = True
    heterogeneous_slots_per_layer: int = 0
    cpu_expert_pin_memory: bool = True

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1024 <= self.dist_port <= 65535

        valid_modes = {"standard", "heter", "spec", "auto"}
        assert self.inference_mode in valid_modes, f"invalid inference_mode: {self.inference_mode}"

        # Backward-compatible inference from legacy flags.
        if self.inference_mode == "auto" or (
            self.inference_mode == "standard" and (self.enable_heterogeneous or self.enable_speculative)
        ):
            if self.enable_speculative:
                self.inference_mode = "spec"
            elif self.enable_heterogeneous:
                self.inference_mode = "heter"
            else:
                self.inference_mode = "standard"

        if self.inference_mode == "standard":
            self.enable_speculative = False
        elif self.inference_mode == "heter":
            assert self.enable_heterogeneous, "heter mode requires enable_heterogeneous=True"
            self.enable_speculative = False
        else:  # spec
            assert self.enable_heterogeneous, "spec mode requires enable_heterogeneous=True"
            self.enable_speculative = True

        assert self.max_draft_tokens >= 1
        assert self.draft_top_c >= 0

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
