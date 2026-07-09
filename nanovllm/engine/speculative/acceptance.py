from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import os

import torch


class AcceptanceStrategy(ABC):
    @abstractmethod
    def accept(
        self,
        draft_tokens: list[int],
        verify_data: torch.Tensor | Sequence[int],
        temperature: float,
        draft_data: torch.Tensor | None = None,
    ) -> dict:
        raise NotImplementedError


def _to_verify_trace(verify_data: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(verify_data, torch.Tensor):
        return verify_data.argmax(dim=-1).tolist()
    return [int(x) for x in verify_data]


class GreedyAcceptance(AcceptanceStrategy):
    def accept(
        self,
        draft_tokens: list[int],
        verify_data: torch.Tensor | Sequence[int],
        temperature: float,
        draft_data: torch.Tensor | None = None,
    ) -> dict:
        _ = draft_data
        verify_argmax = _to_verify_trace(verify_data)
        num_accepted = 0
        for i, tok in enumerate(draft_tokens):
            if i >= len(verify_argmax):
                break
            if tok != verify_argmax[i]:
                break
            num_accepted += 1

        next_pos = min(num_accepted, len(verify_argmax) - 1)
        next_token = verify_argmax[next_pos]
        return {"num_accepted": num_accepted, "next_token": next_token}


def _sample_from_probs(probs: torch.Tensor) -> int:
    probs = probs.float()
    total = probs.sum()
    total_value = float(total.item())
    if not torch.isfinite(total).item() or total_value <= 0.0:
        return int(torch.argmax(probs).item())
    probs = probs / total
    return int(torch.multinomial(probs, num_samples=1).item())


def _temperature_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-10)
    return torch.softmax(logits.float() / safe_temperature, dim=-1)


class StandardAcceptance(AcceptanceStrategy):
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def accept(
        self,
        draft_tokens: list[int],
        verify_data: torch.Tensor | Sequence[int],
        temperature: float,
        draft_data: torch.Tensor | None = None,
    ) -> dict:
        _ = draft_data
        if not isinstance(verify_data, torch.Tensor):
            trace = _to_verify_trace(verify_data)
            num_accepted = 0
            for i, tok in enumerate(draft_tokens):
                if i >= len(trace) or tok != trace[i]:
                    break
                num_accepted += 1
            next_pos = min(num_accepted, len(trace) - 1)
            next_token = trace[next_pos]
            return {
                "num_accepted": num_accepted,
                "next_token": next_token,
                "mode": "standard_deterministic_trace",
            }

        probs = torch.softmax(verify_data.float(), dim=-1)
        num_accepted = 0
        for i, tok in enumerate(draft_tokens):
            if i >= probs.size(0):
                break
            if probs[i, tok].item() < self.threshold:
                break
            num_accepted += 1

        next_pos = min(num_accepted, probs.size(0) - 1)
        next_token = int(torch.argmax(probs[next_pos]).item())
        return {
            "num_accepted": num_accepted,
            "next_token": next_token,
            "mode": "standard_deterministic_threshold",
        }


class StandardSamplingAcceptance(AcceptanceStrategy):
    """Speculative sampling accept/reject with residual resampling."""

    def accept(
        self,
        draft_tokens: list[int],
        verify_data: torch.Tensor | Sequence[int],
        temperature: float,
        draft_data: torch.Tensor | None = None,
    ) -> dict:
        if float(temperature) <= 1e-10 or not isinstance(verify_data, torch.Tensor):
            return GreedyAcceptance().accept(draft_tokens, verify_data, temperature, draft_data)

        if verify_data.dim() != 2:
            raise ValueError("verify_data must be a [positions, vocab] tensor")
        if verify_data.size(0) < 1:
            raise ValueError("verify_data must include at least the next-token position")
        target_probs = _temperature_probs(verify_data, temperature)

        if not draft_tokens:
            return {
                "num_accepted": 0,
                "next_token": _sample_from_probs(target_probs[0]),
                "mode": "standard_sampling",
                "rejected": False,
                "reject_position": None,
            }

        if draft_data is None or not isinstance(draft_data, torch.Tensor):
            raise ValueError("standard_sampling speculative acceptance requires draft logits")
        if draft_data.dim() != 2:
            raise ValueError("draft_data must be a [positions, vocab] tensor")
        if draft_data.size(0) < len(draft_tokens):
            raise ValueError("draft_data must include one logits row per draft token")

        draft_probs = _temperature_probs(draft_data[: len(draft_tokens)], temperature)
        trace_probs = bool(os.getenv("NANOVLLM_ACCEPTANCE_TRACE_PROBS", "").strip())
        accept_probs_trace: list[float] | None = None
        if trace_probs:
            accept_probs_trace = []
            for i, tok in enumerate(draft_tokens):
                if i >= target_probs.size(0):
                    break
                target_p = target_probs[i, tok].clamp_min(0.0)
                draft_q = draft_probs[i, tok].clamp_min(1e-20)
                accept_prob = torch.minimum(target_p / draft_q, torch.ones_like(target_p))
                accept_probs_trace.append(float(accept_prob.item()))

        num_accepted = 0
        for i, tok in enumerate(draft_tokens):
            if i >= target_probs.size(0):
                break
            target_p = target_probs[i, tok].clamp_min(0.0)
            draft_q = draft_probs[i, tok].clamp_min(1e-20)
            accept_prob = torch.minimum(target_p / draft_q, torch.ones_like(target_p))
            if torch.rand((), device=target_probs.device) > accept_prob:
                residual = (target_probs[i] - draft_probs[i]).clamp_min_(0.0)
                out = {
                    "num_accepted": num_accepted,
                    "next_token": _sample_from_probs(residual),
                    "mode": "standard_sampling",
                    "rejected": True,
                    "reject_position": i,
                    "accept_prob": float(accept_prob.item()),
                }
                if accept_probs_trace is not None:
                    out["accept_probs"] = accept_probs_trace
                return out
            num_accepted += 1

        if num_accepted >= len(draft_tokens):
            next_pos = min(num_accepted, target_probs.size(0) - 1)
            out = {
                "num_accepted": num_accepted,
                "next_token": _sample_from_probs(target_probs[next_pos]),
                "mode": "standard_sampling",
                "rejected": False,
                "reject_position": None,
            }
            if accept_probs_trace is not None:
                out["accept_probs"] = accept_probs_trace
            return out

        next_pos = min(num_accepted, target_probs.size(0) - 1)
        out = {
            "num_accepted": num_accepted,
            "next_token": _sample_from_probs(target_probs[next_pos]),
            "mode": "standard_sampling",
            "rejected": False,
            "reject_position": None,
        }
        if accept_probs_trace is not None:
            out["accept_probs"] = accept_probs_trace
        return out


def create_acceptance_strategy(name: str, threshold: float = 0.7) -> AcceptanceStrategy:
    normalized = name.strip().lower()
    if normalized == "greedy":
        return GreedyAcceptance()
    if normalized == "standard":
        return StandardAcceptance(threshold=threshold)
    if normalized in {"standard_sampling", "sampling", "spec_sampling"}:
        return StandardSamplingAcceptance()
    raise ValueError(f"Unsupported acceptance strategy: {name}")
