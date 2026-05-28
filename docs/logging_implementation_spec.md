# nano-vLLM Logging Implementation Spec

> Status: proposed implementation spec  
> Scope: integrate a vLLM-grade logging system into `nano-vllm-moe` with explicit support for per-step timing logs and benchmark-friendly runtime disable switches.

---

## 1. Purpose

This document defines a concrete, implementation-oriented logging design for `nano-vllm-moe`.

The goal is not to copy vLLM logging mechanically. The goal is to adopt the parts of vLLM's model that fit this repository:

1. a central logger module and configuration contract,
2. consistent startup and runtime logs,
3. periodic operational stats,
4. low-noise debug and warning behavior,
5. environment-controlled runtime switches,
6. clear separation between logging, profiling, and benchmark measurement.

This spec is written to be directly implemented in PR-sized steps.

Relevant vLLM references:

1. `vllm/logger.py`: root logger initialization, default dictConfig, `*_once` helpers, log control utilities.  
   Source: [vllm/logger.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/logger.py)
2. `vllm/envs.py`: environment variables controlling logging behavior.  
   Source: [vllm/envs.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/envs.py)
3. `vllm/v1/metrics/loggers.py`: periodic stats logger pattern.  
   Source: [vllm/v1/metrics/loggers.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/metrics/loggers.py)
4. vLLM logging configuration docs: custom JSON `dictConfig` path and default-config control.  
   Source: [logging configuration docs](https://docs.vllm.ai/examples/others/logging_configuration.html)

---

## 2. Target Logging Goals

The logging system in `nano-vllm-moe` should satisfy the following goals.

### 2.1 Primary Goals

1. Provide a single logging subsystem for all library code under `nanovllm/`.
2. Replace ad hoc runtime `print(...)` usage inside library/runtime code with structured logging.
3. Keep script-level human output in `examples/` and `benchmarks/` where appropriate.
4. Make startup/runtime behavior explainable:
   model load mode, heterogeneous enablement, speculative enablement, tensor parallel setup, CUDA graph policy, KV cache allocation, benchmark mode.
5. Provide periodic operational stats similar in spirit to vLLM, but adapted to this repo's current architecture and available counters.
6. Provide optional per-step timing logs for:
   prefill steps,
   draft steps,
   verify steps.
7. Make benchmark runs able to disable:
   visible log emission,
   periodic stats emission,
   and most instrumentation overhead associated with logging.

### 2.2 Secondary Goals

1. Add `warning_once` / `info_once` helpers to avoid repeated noise in hot loops.
2. Keep hot-path call sites clean and readable.
3. Avoid turning the existing `get_profile()` system into the logger itself.
4. Preserve future extensibility for richer metrics and tracing without redesigning the logger contract.

### 2.3 Non-Goals

1. This spec does not replace benchmark JSON outputs.
2. This spec does not replace `engine_profile` / `spec_profile`; those remain structured counters for benchmarks and reports.
3. This spec does not introduce Prometheus or external observability backends.
4. This spec does not require distributed log aggregation across ranks beyond simple rank-aware filtering.

---

## 3. Adaptation of vLLM's Logging Model

vLLM's logging model has three strong ideas worth adopting:

1. central logger bootstrap,
2. env-controlled configuration,
3. periodic stats separate from ordinary logs.

However, `nano-vllm-moe` should not copy vLLM exactly.

### 3.1 What We Should Reuse Conceptually

1. A single `nanovllm.logger` module as the source of truth.
2. Default `logging.config.dictConfig` setup.
3. Optional external JSON logging config file.
4. `info_once` / `warning_once`.
5. Separation of ordinary log records vs periodic stats logger behavior.

### 3.2 What We Should Adapt

`nano-vllm-moe` differs from vLLM in ways that matter:

1. The codebase is smaller and less service-oriented.
2. It already has internal profile dictionaries in:
   [nanovllm/engine/llm_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/llm_engine.py:42),
   [nanovllm/engine/model_runner.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/model_runner.py:138),
   [nanovllm/engine/speculative/spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:31),
   [nanovllm/models/qwen3_moe.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/models/qwen3_moe.py:333).
3. It has more research/benchmark scripts and fewer long-running server endpoints.
4. It has a performance-sensitive speculative path where extra timing instrumentation can matter.

Therefore:

1. Logging must remain thinner than vLLM's full stack.
2. Existing profile counters remain the source of truth for benchmark reports.
3. The logger should consume a compact summary of those counters rather than duplicate all internal accounting.
4. Timing logs must be implemented with explicit fast-path guards so they do not clutter the hot path or add avoidable overhead.

---

## 4. Logging Architecture

## 4.1 High-Level Structure

Add a new module:

`nanovllm/logger.py`

This module owns:

1. logger initialization,
2. default formatters and handlers,
3. env parsing for logging knobs,
4. once-only logging helpers,
5. runtime logging state helpers,
6. periodic stats logger,
7. step timing emission helpers,
8. no-op behavior when logging is disabled.

### 4.1.1 Proposed Public API

```python
import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class LoggingConfig:
    configure_logging: bool
    logging_level: str
    logging_stream: str
    logging_prefix: str
    logging_color: bool
    logging_config_path: str | None

    stats_enabled: bool
    stats_interval_s: float

    timing_enabled: bool
    timing_level: str
    timing_include_rank: bool

    benchmark_quiet: bool
    instrumentation_enabled: bool


def get_logging_config() -> LoggingConfig: ...
def configure_nanovllm_logging() -> None: ...
def init_logger(name: str) -> logging.Logger: ...

def info_once(logger: logging.Logger, msg: str, *args, **kwargs) -> None: ...
def warning_once(logger: logging.Logger, msg: str, *args, **kwargs) -> None: ...
def debug_once(logger: logging.Logger, msg: str, *args, **kwargs) -> None: ...

def is_logging_enabled() -> bool: ...
def is_stats_logging_enabled() -> bool: ...
def is_timing_logging_enabled() -> bool: ...
def is_instrumentation_enabled() -> bool: ...

class StatsLogger:
    def __init__(self, logger: logging.Logger, interval_s: float): ...
    def maybe_log(self, snapshot: dict, now: float | None = None) -> None: ...

class NoOpStatsLogger:
    def maybe_log(self, snapshot: dict, now: float | None = None) -> None: ...

def log_step_timing(
    logger: logging.Logger,
    *,
    phase: str,
    elapsed_ms: float,
    seq_count: int | None = None,
    token_count: int | None = None,
    extra: dict | None = None,
    level: int | None = None,
) -> None: ...
```

### 4.1.2 Internal Helper API

```python
def should_emit_for_rank(rank: int | None) -> bool: ...
def parse_bool_env(name: str, default: bool) -> bool: ...
def parse_float_env(name: str, default: float) -> float: ...
def parse_level_env(name: str, default: str) -> int: ...
```

---

## 4.2 Module Boundaries

### `nanovllm/logger.py`

Responsibilities:

1. bootstrap logging,
2. expose loggers,
3. manage env-driven runtime state,
4. provide periodic stats and timing helpers,
5. keep no-op behavior centralized.

Must not:

1. depend on engine internals,
2. import heavy runtime modules,
3. compute model metrics itself.

### Runtime modules such as:

1. [nanovllm/engine/llm_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/llm_engine.py:17)
2. [nanovllm/engine/model_runner.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/model_runner.py:20)
3. [nanovllm/engine/speculative/spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:11)
4. [nanovllm/models/qwen3_moe.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/models/qwen3_moe.py:301)

Responsibilities:

1. call `init_logger(__name__)`,
2. emit startup and operational logs,
3. feed compact snapshots into the stats logger,
4. capture timing only when the logger layer says instrumentation is enabled.

Must not:

1. contain ad hoc logging config code,
2. perform expensive string formatting before checking logging state,
3. create duplicate timing policy logic.

### Example and benchmark scripts

Responsibilities:

1. keep script-specific final output formatting,
2. pass explicit knobs for benchmark quiet mode,
3. avoid manually reconfiguring loggers unless the script specifically wants custom behavior.

---

## 5. Configuration Model

The configuration model has three layers, highest priority first.

1. explicit kwargs/config fields passed by the caller,
2. environment variables,
3. library defaults.

### 5.1 Proposed `Config` Additions

Add the following fields to [nanovllm/config.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/config.py:6):

```python
logging_enabled: bool = True
logging_level: str = "INFO"
logging_prefix: str = ""

stats_logging_enabled: bool = True
stats_logging_interval: float = 10.0

timing_logging_enabled: bool = False
timing_logging_level: str = "DEBUG"

benchmark_quiet: bool = False
log_instrumentation_enabled: bool | None = None
```

Semantics:

1. `logging_enabled`
   Controls visible log emission for ordinary logger output.
2. `stats_logging_enabled`
   Controls periodic stats emission only.
3. `timing_logging_enabled`
   Controls per-step timing log emission only.
4. `benchmark_quiet`
   Convenience switch intended for benchmark scripts. It forces:
   `logging_enabled=False`,
   `stats_logging_enabled=False`,
   `timing_logging_enabled=False`,
   and `log_instrumentation_enabled=False` unless the caller explicitly overrides.
5. `log_instrumentation_enabled`
   Controls whether timing/logging-specific measurements are captured at all.
   `None` means derive automatically:
   enabled if either `stats_logging_enabled` or `timing_logging_enabled` is true,
   disabled otherwise.

### 5.2 Proposed Environment Variables

All names below are repo-specific rather than `VLLM_*`.

```text
NANOVLLM_CONFIGURE_LOGGING=1
NANOVLLM_LOGGING_LEVEL=INFO
NANOVLLM_LOGGING_STREAM=stderr
NANOVLLM_LOGGING_PREFIX=
NANOVLLM_LOGGING_COLOR=0
NANOVLLM_LOGGING_CONFIG_PATH=

NANOVLLM_LOGGING_ENABLED=1
NANOVLLM_STATS_LOGGING_ENABLED=1
NANOVLLM_LOG_STATS_INTERVAL=10

NANOVLLM_TIMING_LOGGING_ENABLED=0
NANOVLLM_TIMING_LOGGING_LEVEL=DEBUG

NANOVLLM_BENCHMARK_QUIET=0
NANOVLLM_LOG_INSTRUMENTATION_ENABLED=
```

### 5.3 Resolution Rules

Resolution order:

1. construct `Config` from kwargs,
2. apply environment defaults where the field was not explicitly set by caller,
3. derive benchmark/instrumentation behavior.

Recommended derivation:

```python
if config.benchmark_quiet:
    config.logging_enabled = False
    config.stats_logging_enabled = False
    config.timing_logging_enabled = False
    if config.log_instrumentation_enabled is None:
        config.log_instrumentation_enabled = False

if config.log_instrumentation_enabled is None:
    config.log_instrumentation_enabled = (
        config.stats_logging_enabled or config.timing_logging_enabled
    )
```

This is the key distinction required for benchmark mode:

1. visible emission can be disabled,
2. periodic stats can be disabled separately,
3. instrumentation itself can also be disabled to remove measurement overhead.

---

## 6. Default Logging Formats and Levels

## 6.1 Default Format

Default ordinary log format:

```text
LEVEL YYYY-MM-DD HH:MM:SS | nanovllm.module | message
```

Example:

```text
INFO 2026-04-20 12:14:07 | nanovllm.engine.llm_engine | initialized engine mode=spec tp=1 model=/models/Qwen3-30B-A3B
```

This is intentionally simpler than vLLM's file-and-line-heavy default.
For this repo, module name is higher signal than source line in normal operation.

### 6.1.1 Debug Format

When `logging_level=DEBUG`, append filename and line number:

```text
DEBUG 2026-04-20 12:14:07 | nanovllm.engine.model_runner | model_runner.py:245 | captured draft cuda graph bucket=8
```

### 6.1.2 Optional Prefix

If `logging_prefix` is set, prepend:

```text
[bench-a100] INFO 2026-04-20 12:14:07 | ...
```

Useful for multi-run benchmarking.

## 6.2 Log Levels by Category

`INFO`

1. startup summaries,
2. mode enablement,
3. CUDA graph capture summary,
4. periodic stats,
5. one-time warnings promoted to info when appropriate.

`WARNING`

1. configuration inconsistencies,
2. fallbacks from fast path to eager path,
3. missing optional capabilities,
4. one-time degraded-path notices.

`DEBUG`

1. per-step timing logs,
2. graph hit/miss details,
3. speculative step summaries,
4. draft/verify decision details,
5. detailed heterogeneous-path summaries.

`ERROR`

1. runtime failures with enough context for diagnosis.

---

## 7. Proposed `nanovllm/logger.py` Design

## 7.1 Bootstrap Behavior

Add to [nanovllm/__init__.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/__init__.py:1):

```python
from nanovllm.logger import configure_nanovllm_logging

configure_nanovllm_logging()
```

If import-time side effects feel too aggressive, alternatively call it from [nanovllm/engine/llm_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/llm_engine.py:19) before any runtime logger is used. The preferred approach is import-time bootstrap because it keeps logger behavior consistent across entrypoints.

Bootstrap rules:

1. If `NANOVLLM_CONFIGURE_LOGGING=0`, do nothing.
2. If `NANOVLLM_LOGGING_CONFIG_PATH` is set, load JSON `dictConfig`.
3. Otherwise install `DEFAULT_LOGGING_CONFIG`.
4. Make bootstrap idempotent.

## 7.2 `DEFAULT_LOGGING_CONFIG`

Recommended shape:

```python
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"()": "nanovllm.logger.NanoFormatter", "debug": False},
        "debug": {"()": "nanovllm.logger.NanoFormatter", "debug": True},
    },
    "handlers": {
        "nanovllm": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "standard",
            "level": "DEBUG",
        }
    },
    "loggers": {
        "nanovllm": {
            "handlers": ["nanovllm"],
            "level": "INFO",
            "propagate": False,
        }
    },
}
```

`NanoFormatter` should:

1. optionally include filename/line in debug mode,
2. optionally prepend `logging_prefix`,
3. optionally colorize levels if enabled.

## 7.3 Once-Only Helpers

Maintain an internal process-local set:

```python
_ONCE_KEYS: set[tuple[str, str]] = set()
```

Key by:

```python
(logger.name, rendered_message_template)
```

Expose:

1. `debug_once`
2. `info_once`
3. `warning_once`

Use cases:

1. warning that standard graph is skipped in spec mode,
2. warning that logging is disabled by benchmark mode,
3. warning that detailed timing is requested but instrumentation is off.

## 7.4 Stats Logger Design

`StatsLogger` should be lightweight and stateful.

```python
class StatsLogger:
    def __init__(self, logger, interval_s):
        self.logger = logger
        self.interval_s = interval_s
        self._last_log_t = 0.0

    def maybe_log(self, snapshot, now=None):
        if not is_stats_logging_enabled():
            return
        if now is None:
            now = perf_counter()
        if now - self._last_log_t < self.interval_s:
            return
        self._last_log_t = now
        self.logger.info("stats %s", format_stats_snapshot(snapshot))
```

`NoOpStatsLogger.maybe_log()` should return immediately.

The stats logger should not calculate the snapshot. The caller provides a compact dict.

## 7.5 Timing Helper Design

To keep hot-path code clean, timing emission should use a tiny helper:

```python
def log_step_timing(logger, *, phase, elapsed_ms, seq_count=None, token_count=None, extra=None, level=None):
    if not is_timing_logging_enabled():
        return
    payload = [f"phase={phase}", f"elapsed_ms={elapsed_ms:.3f}"]
    if seq_count is not None:
        payload.append(f"seqs={seq_count}")
    if token_count is not None:
        payload.append(f"tokens={token_count}")
    if extra:
        payload.extend(f"{k}={v}" for k, v in extra.items())
    logger.log(level or logging.DEBUG, "step_timing " + " ".join(payload))
```

This creates a single stable log family:

```text
DEBUG ... | nanovllm.engine.llm_engine | step_timing phase=prefill elapsed_ms=12.481 seqs=8 tokens=1024 mode=heter
DEBUG ... | nanovllm.engine.speculative.spec_engine | step_timing phase=draft elapsed_ms=1.932 seqs=4 draft_iter=2
DEBUG ... | nanovllm.engine.speculative.spec_engine | step_timing phase=verify elapsed_ms=4.201 seqs=4 verify_tokens=20
```

---

## 8. Logging Categories and Exact Integration Points

## 8.1 Startup Logs

Startup logs are emitted once per engine/model-runner construction. These are `INFO` logs.

### Integration Point A: `LLMEngine.__init__`

File: [nanovllm/engine/llm_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/llm_engine.py:19)

Add:

```python
from nanovllm.logger import init_logger, create_stats_logger

logger = init_logger(__name__)
```

Emit after config creation:

1. inference mode,
2. tensor parallel size,
3. model path,
4. heterogeneous/spec flags,
5. benchmark quiet state,
6. logging/timing/stats enablement.

Example:

```text
INFO ... | nanovllm.engine.llm_engine | initialized engine mode=spec tp=1 heter=true spec=true benchmark_quiet=false logging=true stats=true timing=false
```

### Integration Point B: `ModelRunner.__init__`

File: [nanovllm/engine/model_runner.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/model_runner.py:26)

Emit:

1. rank/world size,
2. NCCL initialization,
3. default dtype/device transition,
4. heterogeneous loader enabled,
5. draft scheduler enabled,
6. eager vs graph mode,
7. warmup completion,
8. KV cache sizing summary.

Example:

```text
INFO ... | nanovllm.engine.model_runner | model runner initialized rank=0 world_size=1 enforce_eager=false mode=spec
INFO ... | nanovllm.engine.model_runner | heterogeneous mode enabled cpu_exec=true parallel_exec=true draft_scheduler=simple
INFO ... | nanovllm.engine.model_runner | kv cache allocated blocks=1536 block_size=256 gpu_mem_util=0.90
```

### Integration Point C: speculative engine constructor

File: [nanovllm/engine/speculative/spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:18)

Emit:

1. max draft tokens,
2. acceptance strategy,
3. acceptance threshold,
4. timing logging enabled state if spec path timing is on.

---

## 8.2 Operational Logs

Operational logs are occasional `INFO` or `WARNING` logs for significant runtime events.

### Integration Point D: CUDA graph capture

File: [nanovllm/engine/model_runner.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/model_runner.py:66)

Emit:

1. standard graph capture skipped in spec mode,
2. draft graph capture started/completed,
3. graph bucket summary,
4. eager fallback if graph template unavailable.

Use `info_once` or `warning_once` for repetitive fallbacks.

### Integration Point E: fallback behavior

Potential sites:

1. `_run_model_eager`
2. `_can_use_standard_cudagraph`
3. `_can_use_draft_cudagraph`
4. speculative temperature fallback in [spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:61)

Examples:

```text
WARNING ... | nanovllm.engine.speculative.spec_engine | speculative path fell back to standard decode because temperature > 0
WARNING ... | nanovllm.engine.model_runner | draft cuda graph miss bs=6 max_bucket=4 falling back to eager
```

These should be `warning_once` in loops.

---

## 8.3 Debug Logs

Debug logs are high detail and should be disabled by default.

### Suggested Debug Content

1. graph policy choice per decode step,
2. per-step speculative summary:
   `draft_steps`,
   `accepted_tokens`,
   `verify_trace_tokens_total`,
3. compact MoE route summary on rank 0 only:
   `cpu_route_ratio`,
   `cpu_weight_mass_ratio`,
   `realized_cpu_expert_count`.

### Integration Point F: after `speculative_step`

File: [nanovllm/engine/speculative/spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:57)

At the end of `speculative_step`, emit a single debug summary if `logger.isEnabledFor(DEBUG)`.

Example:

```text
DEBUG ... | nanovllm.engine.speculative.spec_engine | spec_step step=42 seqs=4 draft_steps=4 accepted_total=9 verify_tokens=16 step_ms=8.774
```

---

## 8.4 Periodic Stats Logs

Periodic stats logs are `INFO` and intentionally less frequent than per-step logs.

### Integration Point G: `LLMEngine.generate`

File: [nanovllm/engine/llm_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/llm_engine.py:154)

Add `self.stats_logger` during initialization:

```python
self.stats_logger = create_stats_logger(logger)
```

Inside the generate loop, after each `step()`:

1. maintain rolling counters for:
   prefill throughput,
   decode throughput,
   completed requests,
   running requests if cheaply available,
   graph hit rate from `get_profile(reset=False)` only if instrumentation is enabled.
2. call `self.stats_logger.maybe_log(snapshot)`.

Recommended snapshot fields:

```python
{
    "mode": self.config.inference_mode,
    "prefill_toks_per_s": round(prefill_throughput, 2),
    "decode_toks_per_s": round(decode_throughput, 2),
    "finished": len(outputs),
    "scheduled": len(seqs) if available,
    "graph_hit_rate": profile.get("graph_hit_rate"),
    "cpu_route_ratio": profile.get("cpu_route_ratio"),
    "accepted_ratio": accepted_ratio_if_available,
}
```

Recommended emitted format:

```text
INFO ... | nanovllm.engine.llm_engine | stats mode=spec prefill_toks_per_s=5412.8 decode_toks_per_s=327.4 finished=3 graph_hit_rate=0.75 cpu_route_ratio=0.62 accepted_ratio=0.71
```

Important: periodic stats logging must be independently disableable from ordinary logging.

---

## 9. Per-Step Timing Logs

This section specifies the additional capability requested by the user.

## 9.1 Requirements

The logger must be able to emit elapsed time after every:

1. prefill step,
2. draft step,
3. verify step.

The design must:

1. define exact timing boundaries,
2. define exact integration points,
3. define emitted format,
4. avoid turning the hot path into logging spaghetti.

## 9.2 Timing Policy

Per-step timing logs are emitted only when:

1. `timing_logging_enabled=True`, and
2. `log_instrumentation_enabled=True`.

If timing logging is enabled but instrumentation is disabled, `warning_once` should log a configuration inconsistency and no timing should be emitted.

## 9.3 Prefill Timing

### Capture Point

File: [nanovllm/engine/llm_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/llm_engine.py:114)

Timing boundary:

1. start immediately before `self.model_runner.call("run", seqs, True)`,
2. stop immediately after `self.scheduler.postprocess(seqs, token_ids)`.

This captures the full prefill step cost visible to the engine, not just model execution.

Pseudo-integration:

```python
prefill_t0 = perf_counter() if self._timing_enabled else None
token_ids = self.model_runner.call("run", seqs, True)
self.scheduler.postprocess(seqs, token_ids)
if self._timing_enabled:
    log_step_timing(
        logger,
        phase="prefill",
        elapsed_ms=(perf_counter() - prefill_t0) * 1000.0,
        seq_count=len(seqs),
        token_count=num_tokens,
        extra={"mode": self.config.inference_mode},
    )
```

### Emitted Format

```text
DEBUG ... | nanovllm.engine.llm_engine | step_timing phase=prefill elapsed_ms=12.481 seqs=8 tokens=1024 mode=heter
```

## 9.4 Draft Timing

Draft timing in this repo should mean each draft iteration inside the speculative loop, not only the whole draft loop total.

### Capture Point

File: [nanovllm/engine/speculative/spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:101)

Inside:

```python
for step_idx in range(draft_steps):
```

Timing boundary:

1. start immediately before `self.model_runner.call("run_draft", seqs)`,
2. stop immediately after the returned token ids have been appended to all sequences for that draft iteration.

This defines "draft step" as one draft infer-and-apply iteration.

Pseudo-integration:

```python
draft_iter_t0 = perf_counter() if self._timing_enabled else None
draft_result = self.model_runner.call("run_draft", seqs)
...
for seq, token_id in zip(seqs, token_ids):
    seq.append_draft_token(token_id)
...
if self._timing_enabled:
    log_step_timing(
        logger,
        phase="draft",
        elapsed_ms=(perf_counter() - draft_iter_t0) * 1000.0,
        seq_count=len(seqs),
        token_count=len(token_ids),
        extra={"draft_iter": step_idx, "draft_budget": draft_steps},
    )
```

### Emitted Format

```text
DEBUG ... | nanovllm.engine.speculative.spec_engine | step_timing phase=draft elapsed_ms=1.932 seqs=4 tokens=4 draft_iter=2 draft_budget=4
```

## 9.5 Verify Timing

### Capture Point

File: [nanovllm/engine/speculative/spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:147)

Timing boundary:

1. start immediately before `self.model_runner.call("run_verify", seqs, verify_lengths)`,
2. stop immediately after `verify_traces` is returned and incorporated into `verify_tokens_map`.

Pseudo-integration:

```python
verify_t0 = perf_counter() if self._timing_enabled else None
verify_traces = self.model_runner.call("run_verify", seqs, verify_lengths)
...
for seq, trace in zip(seqs, verify_traces):
    verify_tokens_map[seq.seq_id] = trace
...
if self._timing_enabled:
    log_step_timing(
        logger,
        phase="verify",
        elapsed_ms=(perf_counter() - verify_t0) * 1000.0,
        seq_count=len(seqs),
        token_count=sum(len(trace) for trace in verify_traces),
        extra={"verify_calls": 1},
    )
```

### Emitted Format

```text
DEBUG ... | nanovllm.engine.speculative.spec_engine | step_timing phase=verify elapsed_ms=4.201 seqs=4 tokens=20 verify_calls=1
```

## 9.6 Avoiding Hot-Path Mess

To avoid messy instrumentation:

1. store `self._timing_enabled = is_timing_logging_enabled() and is_instrumentation_enabled()` once during object construction,
2. keep one short `if self._timing_enabled` block around each measured region,
3. centralize formatting in `log_step_timing`,
4. do not inline string construction in the hot path,
5. do not emit per-step logs from lower-level tensor kernels or MoE inner loops.

The timing design intentionally instruments engine/spec control boundaries, not every sub-operator.

---

## 10. Benchmark-Friendly Disable Strategy

This section specifies the additional runtime switch requirements.

## 10.1 Three Distinct Controls

The implementation must distinguish the following.

### A. Disable visible log emission

This means ordinary logger messages should not be emitted to stderr/stdout.

Control:

1. `logging_enabled=False`
2. env: `NANOVLLM_LOGGING_ENABLED=0`

Effect:

1. startup/operational/debug logs suppressed,
2. timing logs suppressed because they are logger emissions,
3. stats logs may still be independently enabled if desired.

### B. Disable periodic stats logging

Control:

1. `stats_logging_enabled=False`
2. env: `NANOVLLM_STATS_LOGGING_ENABLED=0`

Effect:

1. no periodic `stats ...` logs,
2. ordinary startup and warning logs may still appear,
3. timing logs may still appear if enabled.

### C. Minimize instrumentation overhead itself

Control:

1. `log_instrumentation_enabled=False`
2. env: `NANOVLLM_LOG_INSTRUMENTATION_ENABLED=0`
3. usually implied by `benchmark_quiet=True`

Effect:

1. skip `perf_counter()` calls that exist only for logging/timing emission,
2. skip expensive stats snapshot assembly if it exists only for logging,
3. do not call `get_profile(reset=False)` inside `generate()` purely for periodic logging.

Important distinction:

1. existing `engine_profile` and `spec_profile` remain separate from log instrumentation,
2. benchmark scripts may still enable those structured profiles while keeping visible logging off.

## 10.2 Proposed Convenience Switch

Expose `benchmark_quiet` in `Config` and scripts.

Recommended script behavior:

1. benchmark scripts default `benchmark_quiet=True`,
2. benchmark scripts may optionally re-enable:
   `stats_logging_enabled=True` for long exploratory runs,
   or `timing_logging_enabled=True` for diagnosis.

### Example benchmark combinations

Production-like quiet benchmark:

```python
LLM(..., benchmark_quiet=True, engine_profile=True, spec_profile=True)
```

Timing diagnosis benchmark:

```python
LLM(
    ...,
    benchmark_quiet=True,
    timing_logging_enabled=True,
    log_instrumentation_enabled=True,
)
```

This should be allowed, but the benchmark script should opt into it explicitly.

## 10.3 Fast-Path Rules

When all of the following are false:

1. `logging_enabled`
2. `stats_logging_enabled`
3. `timing_logging_enabled`

then:

1. `self.stats_logger` should be a `NoOpStatsLogger`,
2. `self._timing_enabled` should be false,
3. no logging-only snapshots should be built,
4. no logging-only `perf_counter()` calls should happen.

This is the core low-overhead guarantee.

---

## 11. Exact Code Integration Plan

## 11.1 New File

Add:

`nanovllm/logger.py`

Implementation contents:

1. env parsing,
2. `LoggingConfig`,
3. default config install,
4. formatter,
5. `init_logger`,
6. once-only helpers,
7. state helper predicates,
8. stats logger,
9. timing helper.

## 11.2 Update `nanovllm/config.py`

File: [nanovllm/config.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/config.py:6)

Add fields from Section 5.1 and validate:

1. `stats_logging_interval > 0`,
2. `logging_level` valid,
3. `timing_logging_level` valid.

Also add benchmark derivation logic in `__post_init__`.

## 11.3 Update `nanovllm/__init__.py`

File: [nanovllm/__init__.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/__init__.py:1)

Option A:

1. call `configure_nanovllm_logging()` on import.

Option B:

1. leave import clean,
2. call bootstrap from `LLMEngine`.

Preferred implementation: Option A if testing confirms it does not surprise scripts.

## 11.4 Update `LLMEngine`

File: [nanovllm/engine/llm_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/llm_engine.py:17)

Add members:

```python
self.logger = init_logger(__name__)
self.stats_logger = create_stats_logger(self.logger, config)
self._timing_enabled = should_capture_timing(config)
```

Add startup logs in `__init__`.

Add prefill timing log in `step()`.

Add periodic stats emission in `generate()`.

Add optional debug summary after each step:

```text
DEBUG ... | step_summary mode=spec is_prefill=false num_tokens=-4 finished=0
```

## 11.5 Update `ModelRunner`

File: [nanovllm/engine/model_runner.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/model_runner.py:20)

Add:

```python
self.logger = init_logger(__name__)
```

Emit:

1. rank init,
2. heterogeneous loader enabled,
3. warmup done,
4. KV cache allocation,
5. graph capture summary,
6. graph fallback warnings.

Do not add per-forward timing logs here in the first phase; those belong at engine/spec boundaries.

## 11.6 Update `SpeculativeEngine`

File: [nanovllm/engine/speculative/spec_engine.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:11)

Add:

```python
self.logger = init_logger(__name__)
self._timing_enabled = should_capture_timing(config)
```

Emit:

1. startup summary,
2. temperature fallback warning,
3. draft iteration timing logs,
4. verify timing logs,
5. optional per-step speculative summary.

## 11.7 Update Benchmark and Example Scripts

Likely files:

1. [examples/three_mode_speed_compare.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/examples/three_mode_speed_compare.py:1)
2. [examples/heterogeneous_speed_compare.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/examples/heterogeneous_speed_compare.py:1)
3. [examples/benchmarks/draft_standard_decode_forward_bench.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/examples/benchmarks/draft_standard_decode_forward_bench.py:1)

Add optional CLI flags:

```text
--benchmark-quiet
--logging-enabled
--stats-logging-enabled
--timing-logging-enabled
```

Recommended defaults for benchmark scripts:

1. `benchmark_quiet=true`
2. `timing_logging_enabled=false`
3. `stats_logging_enabled=false`

Recommended defaults for example scripts:

1. `benchmark_quiet=false`
2. `logging_enabled=true`
3. `stats_logging_enabled=true` only for long-running demos, otherwise false.

---

## 12. Testing Plan

Add tests in `tests/`.

## 12.1 Logger Unit Tests

New file:

`tests/test_logger.py`

Test cases:

1. default config installs once and is idempotent,
2. `init_logger("nanovllm.foo")` returns logger under the `nanovllm` tree,
3. `warning_once` emits only once,
4. `info_once` emits only once,
5. `NANOVLLM_CONFIGURE_LOGGING=0` skips bootstrap,
6. custom JSON config path is loaded,
7. `benchmark_quiet` disables logging, stats, timing, instrumentation by default.

## 12.2 Stats Logger Tests

New file:

`tests/test_stats_logger.py`

Test cases:

1. `NoOpStatsLogger` does nothing,
2. `StatsLogger` emits only after interval,
3. stats logging obeys `stats_logging_enabled`,
4. snapshot formatting is stable and compact.

## 12.3 Step Timing Tests

New file:

`tests/test_step_timing_logging.py`

Test cases:

1. prefill timing helper emits one line with `phase=prefill`,
2. draft timing helper emits one line per draft iteration,
3. verify timing helper emits one line with `phase=verify`,
4. when instrumentation is disabled, no timing log is emitted,
5. when timing logging is disabled, no timing log is emitted.

These can be implemented with mocked loggers and fake `perf_counter` values.

## 12.4 Engine Integration Tests

Extend existing tests:

1. [tests/test_spec_engine_basic.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/tests/test_spec_engine_basic.py:1)
2. [tests/test_spec_engine_flow.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/tests/test_spec_engine_flow.py:1)
3. [tests/test_llm_engine_mode_dispatch.py](/Users/zhangxuan/Desktop/MoE_Sd/nano-vllm-moe/tests/test_llm_engine_mode_dispatch.py:1)

Add assertions that:

1. timing helper is called at the correct boundaries,
2. benchmark quiet mode produces `NoOpStatsLogger`,
3. config derivation behaves correctly,
4. fallback warning paths use once-only behavior.

## 12.5 Benchmark Safety Tests

New file:

`tests/test_benchmark_quiet_config.py`

Test cases:

1. `benchmark_quiet=True` disables visible logs,
2. `benchmark_quiet=True` disables periodic stats,
3. `benchmark_quiet=True` disables timing instrumentation unless explicitly overridden,
4. caller can explicitly re-enable timing instrumentation for diagnostic runs.

---

## 13. Staged Rollout Plan

The implementation should be split into small, reviewable PRs.

## PR1: Logging Core

Scope:

1. add `nanovllm/logger.py`,
2. add `Config` fields and env parsing,
3. add bootstrap behavior,
4. add `init_logger`, `*_once`, state helpers,
5. add unit tests for logger bootstrap and config.

Success criteria:

1. any `nanovllm` module can import and use a central logger,
2. benchmark quiet/config derivation is tested,
3. no runtime behavior changes yet besides startup logging readiness.

## PR2: Startup and Operational Logs

Scope:

1. integrate loggers into `LLMEngine`, `ModelRunner`, `SpeculativeEngine`,
2. add startup logs and one-time fallback warnings,
3. replace internal runtime `print(...)` in library code if any are added later.

Success criteria:

1. engine/model/spec startup state is visible,
2. graph/fallback paths emit useful warnings,
3. no periodic stats or timing logs yet.

## PR3: Periodic Stats Logger

Scope:

1. add `StatsLogger` and `NoOpStatsLogger`,
2. integrate into `LLMEngine.generate`,
3. define compact snapshot formatting,
4. add tests for interval behavior and disable switches.

Success criteria:

1. long runs get concise periodic stats,
2. stats can be disabled independently,
3. no benchmark regression when stats logging is off.

## PR4: Per-Step Timing Logs

Scope:

1. add timing helper,
2. instrument prefill in `LLMEngine.step`,
3. instrument each draft iteration in `SpeculativeEngine.speculative_step`,
4. instrument verify in `SpeculativeEngine.speculative_step`,
5. add tests for emitted phase names and disable behavior.

Success criteria:

1. prefill/draft/verify timings are available on demand,
2. hot path remains readable,
3. timing emission is off by default.

## PR5: Benchmark Script Wiring

Scope:

1. expose benchmark quiet and logging CLI flags in benchmark/example scripts,
2. set benchmark defaults appropriately,
3. document recommended combinations.

Success criteria:

1. benchmark users can disable logging and instrumentation explicitly,
2. debug runs can selectively re-enable timing.

## PR6: Cleanup and Coverage Expansion

Scope:

1. review remaining `examples/` / `benchmarks/` output boundaries,
2. add missing tests,
3. refine wording and field names based on early usage.

---

## 14. Implementation Notes and Guardrails

1. Do not log from inside tensor inner loops or fused MoE helper internals in the first implementation.
2. Prefer rank-0 emission for most logs. Non-zero ranks should log only startup/failure information unless explicitly enabled later.
3. Keep timing logs at `DEBUG` by default even when enabled.
4. Keep periodic stats compact. One line per interval, not multi-line dumps.
5. Keep benchmark JSON/profile outputs independent from logger formatting.
6. Avoid calling `get_profile(reset=False)` on every step when stats and timing instrumentation are disabled.
7. Prefer `warning_once` for path degradation notices that can otherwise spam.

---

## 15. Recommended Default Behavior

Library default behavior:

1. `logging_enabled=True`
2. `stats_logging_enabled=True`
3. `stats_logging_interval=10.0`
4. `timing_logging_enabled=False`
5. `benchmark_quiet=False`
6. `log_instrumentation_enabled=True` only when stats or timing logging is active

Benchmark script default behavior:

1. `benchmark_quiet=True`
2. `logging_enabled=False`
3. `stats_logging_enabled=False`
4. `timing_logging_enabled=False`
5. explicit opt-in for diagnostic timing runs

This gives normal users useful runtime visibility while keeping benchmark runs clean and low overhead.

---

## 16. Summary

The proposed logging system for `nano-vllm-moe` is:

1. centralized like vLLM,
2. simpler than vLLM where this repo does not need service-grade complexity,
3. explicitly separated into startup logs, operational logs, debug logs, periodic stats logs, and per-step timing logs,
4. benchmark-friendly through three independent controls:
   visible emission,
   periodic stats,
   instrumentation overhead.

The most important implementation choices are:

1. add `nanovllm/logger.py` as the single logging subsystem,
2. instrument prefill/draft/verify only at engine/spec boundaries,
3. centralize timing emission through `log_step_timing`,
4. derive `benchmark_quiet` into a no-op logging fast path,
5. roll out in staged PRs so logging infrastructure lands cleanly before deeper instrumentation.
