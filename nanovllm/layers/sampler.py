import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits_fp32 = logits.float()
        greedy_mask = temperatures <= 1e-10
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)

        scaled_logits = logits_fp32.div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = logits_fp32.argmax(dim=-1)
        sample_tokens = torch.where(greedy_mask, greedy_tokens, sample_tokens)
        return sample_tokens
