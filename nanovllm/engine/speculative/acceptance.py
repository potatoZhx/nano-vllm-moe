from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class AcceptanceStrategy(ABC):
    @abstractmethod
    def accept(self, draft_tokens: list[int], verify_data: torch.Tensor | Sequence[int], temperature: float) -> dict:
        raise NotImplementedError


def _to_verify_trace(verify_data: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(verify_data, torch.Tensor):
        return verify_data.argmax(dim=-1).tolist()
    return [int(x) for x in verify_data]


class GreedyAcceptance(AcceptanceStrategy):
    def accept(self, draft_tokens: list[int], verify_data: torch.Tensor | Sequence[int], temperature: float) -> dict:
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


class StandardAcceptance(AcceptanceStrategy):
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def accept(self, draft_tokens: list[int], verify_data: torch.Tensor | Sequence[int], temperature: float) -> dict:
        # Current verify path in SpecEngine passes argmax traces. Keep behavior stable
        # by falling back to greedy acceptance when logits are unavailable.
        if not isinstance(verify_data, torch.Tensor):
            trace = _to_verify_trace(verify_data)
            num_accepted = 0
            for i, tok in enumerate(draft_tokens):
                if i >= len(trace) or tok != trace[i]:
                    break
                num_accepted += 1
            next_pos = min(num_accepted, len(trace) - 1)
            next_token = trace[next_pos]
            return {"num_accepted": num_accepted, "next_token": next_token}

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
        return {"num_accepted": num_accepted, "next_token": next_token}


def create_acceptance_strategy(name: str, threshold: float = 0.7) -> AcceptanceStrategy:
    normalized = name.strip().lower()
    if normalized == "greedy":
        return GreedyAcceptance()
    if normalized == "standard":
        return StandardAcceptance(threshold=threshold)
    raise ValueError(f"Unsupported acceptance strategy: {name}")
