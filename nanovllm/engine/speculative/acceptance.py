from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AcceptanceStrategy(ABC):
    @abstractmethod
    def accept(self, draft_tokens: list[int], verify_logits: torch.Tensor, temperature: float) -> dict:
        raise NotImplementedError


class GreedyAcceptance(AcceptanceStrategy):
    def accept(self, draft_tokens: list[int], verify_logits: torch.Tensor, temperature: float) -> dict:
        verify_argmax = verify_logits.argmax(dim=-1).tolist()
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

    def accept(self, draft_tokens: list[int], verify_logits: torch.Tensor, temperature: float) -> dict:
        probs = torch.softmax(verify_logits.float(), dim=-1)
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
