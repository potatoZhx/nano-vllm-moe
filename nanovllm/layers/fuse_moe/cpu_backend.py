from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch
import torch.nn.functional as F

from nanovllm.layers.fuse_moe.cpu_workspace import CpuMoeWorkspace
from nanovllm.layers.fuse_moe.cuda_host_callback import (
    launch_host_callback,
    register_host_callback,
)


@dataclass
class CpuMoeResult:
    token_indices: torch.Tensor
    outputs_cpu: torch.Tensor
    prep_ms: float
    compute_ms: float


@dataclass
class _FusedGraphState:
    batch_size: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype
    device: torch.device
    hidden_cpu: torch.Tensor
    topk_ids_cpu: torch.Tensor
    weights_cpu: torch.Tensor
    route_mask_cpu: torch.Tensor
    outputs_cpu: torch.Tensor
    outputs_gpu: torch.Tensor
    route_indices_gpu: torch.Tensor
    gate_up_buf: torch.Tensor
    act_buf: torch.Tensor
    out_buf: torch.Tensor
    submit_callback_id: int = 0
    async_submit_callback_id: int = 0
    sync_callback_id: int = 0
    future: Future | None = None
    error: BaseException | None = None


def _resolve_dynamic_parallel_mode(
    *,
    requested_mode: str,
    num_tasks: int,
    num_threads: int,
    num_routes: int,
) -> tuple[str, int, bool]:
    """Select effective parallel mode and OMP thread count.

    When average routes per expert is <= 2, expert-parallel with
    single-threaded MKL can be faster than serial with full MKL
    because small-M matmuls don't saturate multi-core BLAS.
    """
    if num_tasks <= 1 or num_threads <= 1:
        return "serial", num_threads, False

    avg_routes = num_routes / num_tasks if num_tasks > 0 else 0.0
    if requested_mode == "expert_parallel":
        if avg_routes <= 2.0:
            return "expert_parallel", num_threads, True
        return "serial", num_threads, False

    # requested_mode == "auto": pick based on workload shape
    if avg_routes <= 1.5:
        return "expert_parallel", max(2, num_threads), True
    return "serial", num_threads, False


def _get_omp_num_threads() -> int:
    n = os.cpu_count()
    default = n if n is not None else 1
    return int(os.environ.get("OMP_NUM_THREADS", default))


def _set_omp_context(target: int) -> str:
    """Temporarily set OMP_NUM_THREADS for the current thread."""
    prev = os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    os.environ["OMP_NUM_THREADS"] = str(target)
    return prev


def get_cpu_expert_weights(
    cpu_expert_pool: dict[int, dict[str, object]],
    expert_idx: int,
    compute_dtype: torch.dtype,
    *,
    strict_packed_dtype: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = cpu_expert_pool.get(int(expert_idx))
    if params is None:
        raise RuntimeError(f"Missing CPU expert weights for expert {expert_idx}")

    packed = params.get("packed")
    if packed is not None:
        gate_up = packed.gate_up
        down = packed.down
        if gate_up.dtype != compute_dtype or down.dtype != compute_dtype:
            raise RuntimeError(
                f"CPU expert {expert_idx} dtype mismatch: "
                f"gate_up={gate_up.dtype}, down={down.dtype}, expected={compute_dtype}"
            )
        return gate_up, down

    gate_up = params["gate_up"]
    down = params["down"]
    if strict_packed_dtype:
        if gate_up.dtype != compute_dtype or down.dtype != compute_dtype:
            raise RuntimeError(
                f"CPU expert {expert_idx} dtype mismatch: "
                f"gate_up={gate_up.dtype}, down={down.dtype}, expected={compute_dtype}"
            )
    else:
        if gate_up.dtype != compute_dtype:
            gate_up = gate_up.to(dtype=compute_dtype)
        if down.dtype != compute_dtype:
            down = down.to(dtype=compute_dtype)
    return gate_up, down


class TorchPackedCpuMoeBackend:
    def __init__(
        self,
        *,
        layer_idx: int,
        cpu_expert_pool: dict[int, dict[str, object]],
        max_routes: int,
        strict_dtype: bool = True,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.cpu_expert_pool = cpu_expert_pool
        self.max_routes = int(max_routes)
        self.strict_dtype = bool(strict_dtype)
        self.workspace: CpuMoeWorkspace | None = None
        self._thread_pool: ThreadPoolExecutor | None = None
        self._num_workers = 0

    def _ensure_thread_pool(self, num_workers: int) -> ThreadPoolExecutor:
        if self._thread_pool is None or self._num_workers != num_workers:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=False)
            self._thread_pool = ThreadPoolExecutor(max_workers=num_workers)
            self._num_workers = num_workers
        return self._thread_pool

    def _get_workspace(self, hidden_states: torch.Tensor) -> CpuMoeWorkspace:
        hidden_size = int(hidden_states.shape[-1])
        hidden_pin_memory = False
        output_pin_memory = False
        workspace = self.workspace
        if (
            workspace is None
            or workspace.max_routes != self.max_routes
            or workspace.hidden_size != hidden_size
            or workspace.dtype != hidden_states.dtype
            or workspace.hidden_pin_memory != hidden_pin_memory
            or workspace.output_pin_memory != output_pin_memory
        ):
            workspace = CpuMoeWorkspace.create(
                max_routes=self.max_routes,
                hidden_size=hidden_size,
                dtype=hidden_states.dtype,
                hidden_pin_memory=hidden_pin_memory,
                output_pin_memory=output_pin_memory,
            )
            self.workspace = workspace
        return workspace

    @torch.no_grad()
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        flat_weights: torch.Tensor,
        top_k: int,
        cpu_indices: torch.Tensor,
        cpu_task_expert_ids: torch.Tensor,
        cpu_task_offsets: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        parallel_mode: str = "serial",
        num_threads: int = 4,
        cpu_task_expert_ids_host: list[int] | None = None,
        cpu_task_offsets_host: list[int] | None = None,
        selected_experts: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> CpuMoeResult:
        prep_t0 = perf_counter()
        compute_dtype = hidden_states.dtype
        num_routes = int(cpu_indices.numel())
        if num_routes > self.max_routes:
            raise RuntimeError(f"CPU MoE routes {num_routes} exceed max_routes={self.max_routes}")

        token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")
        workspace = self._get_workspace(hidden_states)
        hidden_cpu = workspace.hidden_cpu[:num_routes]
        weights_cpu = workspace.weights_cpu[:num_routes]
        outputs_cpu = workspace.outputs_cpu[:num_routes]

        if hidden_states.is_cuda:
            route_token_indices = token_indices.to(device=hidden_states.device, non_blocking=True)
            route_indices = cpu_indices.to(device=flat_weights.device, non_blocking=True)
            hidden_gpu = hidden_states.index_select(0, route_token_indices)
            weights_gpu = flat_weights.index_select(0, route_indices)
            hidden_cpu.copy_(hidden_gpu, non_blocking=workspace.hidden_pin_memory)
            weights_cpu.copy_(weights_gpu, non_blocking=workspace.hidden_pin_memory)
            torch.cuda.current_stream(hidden_states.device).synchronize()
        else:
            route_token_indices = token_indices.to(device=hidden_states.device)
            route_indices = cpu_indices.to(device=flat_weights.device)
            hidden_cpu.copy_(hidden_states.index_select(0, route_token_indices))
            weights_cpu.copy_(flat_weights.index_select(0, route_indices))

        if cpu_task_expert_ids_host is not None and cpu_task_offsets_host is not None:
            task_expert_ids_host = cpu_task_expert_ids_host
            task_offsets_host = cpu_task_offsets_host
        else:
            task_offsets_host = [int(x) for x in cpu_task_offsets.detach().to("cpu", non_blocking=False).tolist()]
            task_expert_ids_host = [int(x) for x in cpu_task_expert_ids.detach().to("cpu", non_blocking=False).tolist()]
        prep_ms = (perf_counter() - prep_t0) * 1000.0

        num_tasks = len(task_expert_ids_host)
        eff_mode, eff_threads, set_omp1 = _resolve_dynamic_parallel_mode(
            requested_mode=parallel_mode,
            num_tasks=num_tasks,
            num_threads=num_threads,
            num_routes=num_routes,
        )

        def run_task(task_idx: int) -> None:
            start = task_offsets_host[task_idx]
            end = task_offsets_host[task_idx + 1]
            if start >= end:
                return

            expert_idx = task_expert_ids_host[task_idx]
            gate_up_weight, down_weight = get_cpu_expert_weights(
                self.cpu_expert_pool,
                expert_idx,
                compute_dtype,
                strict_packed_dtype=self.strict_dtype,
            )
            gate_up = F.linear(hidden_cpu[start:end], gate_up_weight)
            out = F.linear(act_fn(gate_up), down_weight)
            out.mul_(weights_cpu[start:end].unsqueeze(-1))
            outputs_cpu[start:end].copy_(out)

        compute_t0 = perf_counter()
        # ThreadPoolExecutor threads don't inherit InferenceMode.
        in_inference_mode = torch.is_inference_mode_enabled()
        can_parallel = (
            eff_mode == "expert_parallel"
            and eff_threads > 1
            and num_tasks > 1
            and not in_inference_mode
        )
        if can_parallel:
            if set_omp1:
                prev_omp = _get_omp_num_threads()
                _set_omp_context(1)
            try:
                pool = self._ensure_thread_pool(min(eff_threads, num_tasks))
                list(pool.map(run_task, range(num_tasks)))
            finally:
                if set_omp1:
                    _set_omp_context(prev_omp)
        else:
            for task_idx in range(num_tasks):
                run_task(task_idx)
        compute_ms = (perf_counter() - compute_t0) * 1000.0

        return CpuMoeResult(
            token_indices=token_indices,
            outputs_cpu=outputs_cpu,
            prep_ms=prep_ms,
            compute_ms=compute_ms,
        )


class FusedTorchCpuMoeBackend:
    """CPU MoE backend using pre-allocated intermediate buffers.

    This keeps the same dtype as the model weights and avoids allocating
    gate/up, activation, and output temporaries for each expert task.
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        cpu_expert_pool: dict[int, dict[str, object]],
        max_routes: int,
        moe_intermediate_size: int,
        strict_dtype: bool = True,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.cpu_expert_pool = cpu_expert_pool
        self.max_routes = int(max_routes)
        self.intermediate_size = int(moe_intermediate_size)
        self.strict_dtype = bool(strict_dtype)
        self.workspace: CpuMoeWorkspace | None = None
        self._thread_pool: ThreadPoolExecutor | None = None
        self._graph_worker: ThreadPoolExecutor | None = None
        self._num_workers = 0
        self._graph_states: dict[tuple[int, torch.dtype, torch.device], _FusedGraphState] = {}

    def _ensure_thread_pool(self, num_workers: int) -> ThreadPoolExecutor:
        if self._thread_pool is None or self._num_workers != num_workers:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=False)
            self._thread_pool = ThreadPoolExecutor(max_workers=num_workers)
            self._num_workers = num_workers
        return self._thread_pool

    def _get_workspace(self, hidden_states: torch.Tensor) -> CpuMoeWorkspace:
        hidden_size = int(hidden_states.shape[-1])
        workspace = self.workspace
        if (
            workspace is None
            or workspace.max_routes != self.max_routes
            or workspace.hidden_size != hidden_size
            or workspace.dtype != hidden_states.dtype
            or workspace.intermediate_size != self.intermediate_size
        ):
            workspace = CpuMoeWorkspace.create(
                max_routes=self.max_routes,
                hidden_size=hidden_size,
                dtype=hidden_states.dtype,
                hidden_pin_memory=False,
                output_pin_memory=False,
                intermediate_size=self.intermediate_size,
            )
            self.workspace = workspace
        return workspace

    def _ensure_graph_worker(self) -> ThreadPoolExecutor:
        if self._graph_worker is None:
            self._graph_worker = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"fused-graph-cpu-moe-l{self.layer_idx}",
            )
        return self._graph_worker

    def _get_graph_state(
        self,
        *,
        batch_size: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _FusedGraphState:
        key = (int(batch_size), dtype, device)
        state = self._graph_states.get(key)
        if state is not None and state.top_k == int(top_k) and state.hidden_size == int(hidden_size):
            return state

        num_routes = int(batch_size) * int(top_k)
        state = _FusedGraphState(
            batch_size=int(batch_size),
            top_k=int(top_k),
            hidden_size=int(hidden_size),
            intermediate_size=self.intermediate_size,
            dtype=dtype,
            device=device,
            hidden_cpu=torch.empty(
                (batch_size, hidden_size),
                device="cpu",
                dtype=dtype,
                pin_memory=True,
            ),
            topk_ids_cpu=torch.empty(
                (num_routes,),
                device="cpu",
                dtype=torch.int64,
                pin_memory=True,
            ),
            weights_cpu=torch.empty(
                (num_routes,),
                device="cpu",
                dtype=dtype,
                pin_memory=True,
            ),
            route_mask_cpu=torch.empty(
                (num_routes,),
                device="cpu",
                dtype=torch.bool,
                pin_memory=True,
            ),
            outputs_cpu=torch.empty(
                (num_routes, hidden_size),
                device="cpu",
                dtype=dtype,
                pin_memory=True,
            ),
            outputs_gpu=torch.empty(
                (num_routes, hidden_size),
                device=device,
                dtype=dtype,
            ),
            route_indices_gpu=torch.arange(
                num_routes,
                device=device,
                dtype=torch.int64,
            ),
            gate_up_buf=torch.empty(
                (num_routes, self.intermediate_size * 2),
                device="cpu",
                dtype=dtype,
                pin_memory=False,
            ),
            act_buf=torch.empty(
                (num_routes, self.intermediate_size),
                device="cpu",
                dtype=dtype,
                pin_memory=False,
            ),
            out_buf=torch.empty(
                (num_routes, hidden_size),
                device="cpu",
                dtype=dtype,
                pin_memory=False,
            ),
        )
        state.submit_callback_id = register_host_callback(lambda s=state: self._submit_graph_state(s))
        state.async_submit_callback_id = register_host_callback(lambda s=state: self._submit_graph_state_async_sidecar(s))
        state.sync_callback_id = register_host_callback(lambda s=state: self._sync_graph_state(s))
        self._graph_states[key] = state
        return state

    def _submit_graph_state(self, state: _FusedGraphState) -> None:
        try:
            if state.future is not None and not state.future.done():
                state.future.result()
            state.error = None
            state.future = self._ensure_graph_worker().submit(self._compute_graph_state, state)
        except BaseException as exc:
            state.error = exc

    def _submit_graph_state_async_sidecar(self, state: _FusedGraphState) -> None:
        try:
            if state.future is not None:
                if not state.future.done():
                    return
                state.future.result()
            state.error = None
            hidden_cpu = state.hidden_cpu.clone()
            topk_ids_cpu = state.topk_ids_cpu.clone()
            weights_cpu = state.weights_cpu.clone()
            route_mask_cpu = state.route_mask_cpu.clone()
            state.future = self._ensure_graph_worker().submit(
                self._compute_graph_sidecar,
                hidden_cpu,
                topk_ids_cpu,
                weights_cpu,
                route_mask_cpu,
                state.top_k,
                state.hidden_size,
                state.intermediate_size,
                state.dtype,
            )
        except BaseException as exc:
            state.error = exc

    def _sync_graph_state(self, state: _FusedGraphState) -> None:
        try:
            if state.future is not None:
                state.future.result()
        except BaseException as exc:
            state.error = exc

    def _compute_graph_sidecar(
        self,
        hidden_cpu: torch.Tensor,
        topk_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        route_mask_cpu: torch.Tensor,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        compute_dtype: torch.dtype,
    ) -> None:
        with torch.inference_mode():
            active = torch.nonzero(route_mask_cpu, as_tuple=False).flatten()
            if active.numel() == 0:
                return

            max_routes = int(topk_ids_cpu.numel())
            gate_up_buf = torch.empty(
                (max_routes, intermediate_size * 2),
                device="cpu",
                dtype=compute_dtype,
            )
            act_buf = torch.empty(
                (max_routes, intermediate_size),
                device="cpu",
                dtype=compute_dtype,
            )
            out_buf = torch.empty(
                (max_routes, hidden_size),
                device="cpu",
                dtype=compute_dtype,
            )

            active_experts = topk_ids_cpu.index_select(0, active).unique(sorted=True)
            for expert_tensor in active_experts:
                expert_idx = int(expert_tensor.item())
                expert_mask = route_mask_cpu & topk_ids_cpu.eq(expert_idx)
                route_pos = torch.nonzero(expert_mask, as_tuple=False).flatten()
                if route_pos.numel() == 0:
                    continue

                gate_up_w, down_w = get_cpu_expert_weights(
                    self.cpu_expert_pool,
                    expert_idx,
                    compute_dtype,
                    strict_packed_dtype=self.strict_dtype,
                )
                m = int(route_pos.numel())
                token_pos = torch.div(route_pos, int(top_k), rounding_mode="floor")
                hidden = hidden_cpu.index_select(0, token_pos)
                route_weights = weights_cpu.index_select(0, route_pos)

                gate_up = F.linear(hidden, gate_up_w)
                act_out = F.silu(gate_up[:, :intermediate_size]) * gate_up[:, intermediate_size:]
                expert_out = F.linear(act_out, down_w)
                expert_out.mul_(route_weights.unsqueeze(-1))

    def _compute_graph_state(self, state: _FusedGraphState) -> None:
        compute_dtype = state.dtype
        with torch.inference_mode():
            state.outputs_cpu.zero_()
            active = torch.nonzero(state.route_mask_cpu, as_tuple=False).flatten()
            if active.numel() == 0:
                return

            active_experts = state.topk_ids_cpu.index_select(0, active).unique(sorted=True)
            for expert_tensor in active_experts:
                expert_idx = int(expert_tensor.item())
                expert_mask = state.route_mask_cpu & state.topk_ids_cpu.eq(expert_idx)
                route_pos = torch.nonzero(expert_mask, as_tuple=False).flatten()
                if route_pos.numel() == 0:
                    continue

                gate_up_w, down_w = get_cpu_expert_weights(
                    self.cpu_expert_pool,
                    expert_idx,
                    compute_dtype,
                    strict_packed_dtype=self.strict_dtype,
                )
                m = int(route_pos.numel())
                token_pos = torch.div(route_pos, state.top_k, rounding_mode="floor")
                hidden = state.hidden_cpu.index_select(0, token_pos)
                route_weights = state.weights_cpu.index_select(0, route_pos)

                gate_up = F.linear(hidden, gate_up_w)
                interm = state.intermediate_size
                act_out = F.silu(gate_up[:, :interm]) * gate_up[:, interm:]
                expert_out = F.linear(act_out, down_w)
                expert_out.mul_(route_weights.unsqueeze(-1))
                state.outputs_cpu.index_copy_(0, route_pos, expert_out)

    def check_graph_errors(self) -> None:
        for state in self._graph_states.values():
            if state.future is not None and state.future.done():
                try:
                    state.future.result()
                except BaseException as exc:
                    state.error = exc
            if state.error is not None:
                error = state.error
                state.error = None
                raise RuntimeError(
                    f"Fused graph CPU MoE backend failed on layer {self.layer_idx}"
                ) from error

    @torch.no_grad()
    def begin_forward_graph(
        self,
        *,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cpu_route_mask: torch.Tensor,
        top_k: int,
        async_sidecar: bool = False,
    ) -> CpuMoeResult:
        if not hidden_states.is_cuda:
            raise RuntimeError("Fused graph CPU MoE backend requires CUDA hidden states")

        batch_size = int(hidden_states.shape[0])
        hidden_size = int(hidden_states.shape[-1])
        state = self._get_graph_state(
            batch_size=batch_size,
            top_k=int(top_k),
            hidden_size=hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        flat_ids = selected_experts.reshape(-1).to(torch.int64)
        flat_weights = routing_weights.reshape(-1).to(dtype=hidden_states.dtype)
        flat_mask = cpu_route_mask.reshape(-1).to(torch.bool)
        state.hidden_cpu.copy_(hidden_states.contiguous(), non_blocking=True)
        state.topk_ids_cpu.copy_(flat_ids, non_blocking=True)
        state.weights_cpu.copy_(flat_weights, non_blocking=True)
        state.route_mask_cpu.copy_(flat_mask, non_blocking=True)

        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        callback_id = state.async_submit_callback_id if async_sidecar else state.submit_callback_id
        launch_host_callback(stream, callback_id)
        return state

    def finish_forward_graph(self, state: _FusedGraphState) -> CpuMoeResult:
        stream = torch.cuda.current_stream(state.device).cuda_stream
        launch_host_callback(stream, state.sync_callback_id)
        state.outputs_gpu.copy_(state.outputs_cpu, non_blocking=True)

        return CpuMoeResult(
            token_indices=state.route_indices_gpu,
            outputs_cpu=state.outputs_gpu,
            prep_ms=0.0,
            compute_ms=0.0,
        )

    @torch.no_grad()
    def forward_graph(
        self,
        *,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cpu_route_mask: torch.Tensor,
        top_k: int,
    ) -> CpuMoeResult:
        state = self.begin_forward_graph(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            cpu_route_mask=cpu_route_mask,
            top_k=top_k,
        )
        return self.finish_forward_graph(state)

    @torch.no_grad()
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        flat_weights: torch.Tensor,
        top_k: int,
        cpu_indices: torch.Tensor,
        cpu_task_expert_ids: torch.Tensor,
        cpu_task_offsets: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        parallel_mode: str = "serial",
        num_threads: int = 4,
        cpu_task_expert_ids_host: list[int] | None = None,
        cpu_task_offsets_host: list[int] | None = None,
        selected_experts: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> CpuMoeResult:
        prep_t0 = perf_counter()
        compute_dtype = hidden_states.dtype
        num_routes = int(cpu_indices.numel())
        if num_routes > self.max_routes:
            raise RuntimeError(f"CPU MoE routes {num_routes} exceed max_routes={self.max_routes}")

        token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")
        workspace = self._get_workspace(hidden_states)
        hidden_cpu = workspace.hidden_cpu[:num_routes]
        weights_cpu = workspace.weights_cpu[:num_routes]
        outputs_cpu = workspace.outputs_cpu[:num_routes]
        gate_up_buf = workspace.gate_up_buf
        act_buf = workspace.act_buf
        out_buf = workspace.out_fp32_buf
        if gate_up_buf is None or act_buf is None or out_buf is None:
            raise RuntimeError("Fused CPU MoE workspace is missing intermediate buffers")

        if hidden_states.is_cuda:
            route_token_indices = token_indices.to(device=hidden_states.device, non_blocking=True)
            route_indices = cpu_indices.to(device=flat_weights.device, non_blocking=True)
            hidden_gpu = hidden_states.index_select(0, route_token_indices)
            weights_gpu = flat_weights.index_select(0, route_indices)
            hidden_cpu.copy_(hidden_gpu, non_blocking=False)
            weights_cpu.copy_(weights_gpu, non_blocking=False)
            torch.cuda.current_stream(hidden_states.device).synchronize()
        else:
            route_token_indices = token_indices.to(device=hidden_states.device)
            route_indices = cpu_indices.to(device=flat_weights.device)
            hidden_cpu.copy_(hidden_states.index_select(0, route_token_indices))
            weights_cpu.copy_(flat_weights.index_select(0, route_indices))

        if cpu_task_expert_ids_host is not None and cpu_task_offsets_host is not None:
            task_expert_ids_host = cpu_task_expert_ids_host
            task_offsets_host = cpu_task_offsets_host
        else:
            task_offsets_host = [int(x) for x in cpu_task_offsets.detach().to("cpu", non_blocking=False).tolist()]
            task_expert_ids_host = [int(x) for x in cpu_task_expert_ids.detach().to("cpu", non_blocking=False).tolist()]
        prep_ms = (perf_counter() - prep_t0) * 1000.0

        num_tasks = len(task_expert_ids_host)
        eff_mode, eff_threads, set_omp1 = _resolve_dynamic_parallel_mode(
            requested_mode=parallel_mode,
            num_tasks=num_tasks,
            num_threads=num_threads,
            num_routes=num_routes,
        )

        def run_fused_task(task_idx: int) -> None:
            start = task_offsets_host[task_idx]
            end = task_offsets_host[task_idx + 1]
            if start >= end:
                return

            expert_idx = task_expert_ids_host[task_idx]
            gate_up_w, down_w = get_cpu_expert_weights(
                self.cpu_expert_pool,
                expert_idx,
                compute_dtype,
                strict_packed_dtype=self.strict_dtype,
            )

            M = end - start
            h_chunk = hidden_cpu[start:end]
            w_chunk = weights_cpu[start:end]
            o_slice = outputs_cpu[start:end]
            I = self.intermediate_size

            gate_up = F.linear(h_chunk, gate_up_w)
            expert_out = F.linear(act_fn(gate_up), down_w)
            expert_out.mul_(w_chunk.unsqueeze(-1))
            o_slice.copy_(expert_out)

        compute_t0 = perf_counter()
        # ThreadPoolExecutor threads don't inherit InferenceMode.
        in_inference_mode = torch.is_inference_mode_enabled()
        can_parallel = (
            eff_mode == "expert_parallel"
            and eff_threads > 1
            and num_tasks > 1
            and not in_inference_mode
        )
        if can_parallel:
            if set_omp1:
                prev_omp = _get_omp_num_threads()
                _set_omp_context(1)
            try:
                pool = self._ensure_thread_pool(min(eff_threads, num_tasks))
                list(pool.map(run_fused_task, range(num_tasks)))
            finally:
                if set_omp1:
                    _set_omp_context(prev_omp)
        else:
            for task_idx in range(num_tasks):
                run_fused_task(task_idx)
        compute_ms = (perf_counter() - compute_t0) * 1000.0

        return CpuMoeResult(
            token_indices=token_indices,
            outputs_cpu=outputs_cpu,
            prep_ms=prep_ms,
            compute_ms=compute_ms,
        )
