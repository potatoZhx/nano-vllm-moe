import atexit
from dataclasses import fields
from collections import defaultdict
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.speculative.spec_engine import SpeculativeEngine


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.spec_engine = SpeculativeEngine(self.model_runner, self.scheduler, config)
        self.profile_enabled = bool(getattr(config, "engine_profile", False))
        self._profile = defaultdict(float)
        atexit.register(self.exit)

    def get_profile(self, reset: bool = False) -> dict:
        """Return merged engine/model/spec profile counters on rank-0 process."""
        out = {}
        if self.profile_enabled:
            out.update({
                k: (int(v) if k.endswith("_count") else float(v))
                for k, v in self._profile.items()
            })
        if hasattr(self.model_runner, "get_profile"):
            model_profile = self.model_runner.get_profile(reset=reset)
            out.update({f"model_{k}": v for k, v in model_profile.items()})
        if hasattr(self.spec_engine, "get_profile"):
            spec_profile = self.spec_engine.get_profile(reset=reset)
            out.update({f"spec_{k}": v for k, v in spec_profile.items()})

        # Single-forward granularity metrics across core phases.
        prefill_steps = float(out.get("prefill_step_count", 0.0))
        if prefill_steps > 0:
            out["prefill_forward_ms"] = float(out.get("prefill_runner_ms", 0.0) / prefill_steps)

        decode_steps = float(out.get("decode_step_count", 0.0))
        if decode_steps > 0:
            out["decode_forward_ms"] = float(out.get("decode_runner_ms", 0.0) / decode_steps)

        draft_calls = float(out.get("spec_run_draft_calls", 0.0))
        if draft_calls > 0:
            out["draft_forward_ms"] = float(out.get("spec_run_draft_infer_ms_total", 0.0) / draft_calls)

        verify_calls = float(out.get("spec_run_verify_calls", 0.0))
        if verify_calls > 0:
            out["verify_forward_ms"] = float(out.get("spec_run_verify_infer_ms_total", 0.0) / verify_calls)

        # Canonical phase2_post keys for downstream reports.
        canonical_map = {
            "route_ms": "model_route_ms",
            "plan_ms": "model_plan_ms",
            "gpu_gather_ms": "model_gpu_gather_ms",
            "gpu_compute_ms": "model_gpu_compute_ms",
            "cpu_prepare_ms": "model_cpu_prepare_ms",
            "cpu_compute_ms": "model_cpu_compute_ms",
            "cpu_to_gpu_merge_ms": "model_cpu_to_gpu_merge_ms",
            "scatter_ms": "model_scatter_ms",
            "draft_ms": "spec_draft_ms",
            "verify_ms": "spec_verify_ms",
            "spec_step_ms": "spec_spec_step_ms",
            "graph_hit_rate": "model_graph_hit_rate",
            "graph_replay_count": "model_graph_replay_count",
            "cpu_route_ratio": "model_cpu_route_ratio",
            "cpu_weight_mass_ratio": "model_cpu_weight_mass_ratio",
            "activated_expert_set_size": "model_activated_expert_set_size",
            "realized_cpu_expert_count": "model_realized_cpu_expert_count",
            "prefetch_submit_count": "model_prefetch_submit_count",
            "prefetch_completed_count": "model_prefetch_completed_count",
            "prefetch_late_count": "model_prefetch_late_count",
            "prefetch_wait_ms": "model_prefetch_wait_ms",
            "prefetch_consumed_count": "model_prefetch_consumed_count",
            "prefetch_timeout_count": "model_prefetch_timeout_count",
            "publish_count": "model_publish_count",
            "publish_ms": "model_publish_ms",
            "metadata_offload_count": "model_metadata_offload_count",
            "metadata_offload_ms": "model_metadata_offload_ms",
            "metadata_offload_bytes": "model_metadata_offload_bytes",
            "metadata_offload_enqueue_ms": "model_metadata_offload_enqueue_ms",
            "metadata_offload_transfer_wait_ms": "model_metadata_offload_transfer_wait_ms",
            "metadata_offload_collect_ms": "model_metadata_offload_collect_ms",
            "metadata_offload_observe_ms": "model_metadata_offload_observe_ms",
            "metadata_offload_prefill_count": "model_metadata_offload_prefill_count",
            "metadata_offload_prefill_ms": "model_metadata_offload_prefill_ms",
            "metadata_offload_prefill_bytes": "model_metadata_offload_prefill_bytes",
            "metadata_offload_draft_count": "model_metadata_offload_draft_count",
            "metadata_offload_draft_ms": "model_metadata_offload_draft_ms",
            "metadata_offload_draft_bytes": "model_metadata_offload_draft_bytes",
            "metadata_offload_verify_count": "model_metadata_offload_verify_count",
            "metadata_offload_verify_ms": "model_metadata_offload_verify_ms",
            "metadata_offload_verify_bytes": "model_metadata_offload_verify_bytes",
            "metadata_observe_skipped_count": "model_metadata_observe_skipped_count",
            "history_prefetch_submit_count": "model_history_prefetch_submit_count",
            "verify_history_prefetch_submit_count": "model_verify_history_prefetch_submit_count",
            "draft_live_prefetch_submit_count": "model_draft_live_prefetch_submit_count",
            "verify_ready_before_wait_count": "model_verify_ready_before_wait_count",
            "verify_ready_after_wait_count": "model_verify_ready_after_wait_count",
        }
        for alias, source_key in canonical_map.items():
            if alias not in out and source_key in out:
                out[alias] = out[source_key]
        if reset and self.profile_enabled:
            self._profile.clear()
        return out

    def exit(self):
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return

        model_runner.call("exit")
        self.model_runner = None
        for p in self.ps:
            if p.is_alive():
                p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        step_t0 = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        if self.profile_enabled:
            self._profile["step_count"] += 1
            self._profile["scheduled_seqs_total"] += len(seqs)
            if is_prefill:
                self._profile["prefill_step_count"] += 1
            elif self.config.inference_mode == "spec":
                self._profile["spec_step_count"] += 1
            else:
                self._profile["decode_step_count"] += 1

        if is_prefill:
            t0 = perf_counter()
            token_ids = self.model_runner.call("run", seqs, True)
            if self.profile_enabled:
                self._profile["prefill_runner_ms"] += (perf_counter() - t0) * 1000.0
            t1 = perf_counter()
            self.scheduler.postprocess(seqs, token_ids)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(len(seq) for seq in seqs)
            if self.profile_enabled:
                self._profile["postprocess_ms"] += (perf_counter() - t1) * 1000.0
                self._profile["step_ms"] += (perf_counter() - step_t0) * 1000.0
            return outputs, num_tokens
        elif self.config.inference_mode == "spec":
            t0 = perf_counter()
            token_ids = self.spec_engine.speculative_step(seqs)
            if self.profile_enabled:
                self._profile["spec_engine_ms"] += (perf_counter() - t0) * 1000.0
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = -len(seqs)
            if self.profile_enabled:
                self._profile["step_ms"] += (perf_counter() - step_t0) * 1000.0
            return outputs, num_tokens
        else:
            t0 = perf_counter()
            token_ids = self.model_runner.call("run", seqs, False)
            if self.profile_enabled:
                self._profile["decode_runner_ms"] += (perf_counter() - t0) * 1000.0
        t1 = perf_counter()
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        if self.profile_enabled:
            self._profile["postprocess_ms"] += (perf_counter() - t1) * 1000.0
            self._profile["step_ms"] += (perf_counter() - step_t0) * 1000.0
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
