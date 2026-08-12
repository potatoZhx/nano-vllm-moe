import torch
from torch import nn


def filtered_sampling_probs(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    *,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Return the exact temperature/top-k/top-p distribution over full vocab."""
    if isinstance(temperature, torch.Tensor):
        denominator = temperature.float().clamp_min(1e-10)
    else:
        denominator = max(float(temperature), 1e-10)
    scaled = logits.float() / denominator
    if int(top_k) <= 0 and float(top_p) >= 1.0:
        return torch.softmax(scaled, dim=-1)
    vocab_size = int(scaled.shape[-1])
    effective_k = vocab_size if int(top_k) <= 0 else min(int(top_k), vocab_size)
    if effective_k < vocab_size:
        values, indices = torch.topk(scaled, effective_k, dim=-1, sorted=True)
    else:
        values, indices = torch.sort(scaled, dim=-1, descending=True)
    probs = torch.softmax(values, dim=-1)
    if float(top_p) < 1.0:
        remove = probs.cumsum(dim=-1) > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probs = probs.masked_fill(remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    full = torch.zeros_like(scaled)
    full.scatter_(-1, indices, probs)
    return full


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def _sample_unfiltered(
        self,
        logits_fp32: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        greedy_tokens: torch.Tensor,
    ) -> torch.Tensor:
        safe_temperatures = torch.where(
            greedy_mask, torch.ones_like(temperatures), temperatures
        )
        probs = torch.softmax(
            logits_fp32.div(safe_temperatures.unsqueeze(dim=1)), dim=-1
        )
        sampled = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled)

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: list[int] | None = None,
        top_ps: list[float] | None = None,
    ):
        logits_fp32 = logits.float()
        greedy_mask = temperatures <= 1e-10
        greedy_tokens = logits_fp32.argmax(dim=-1)
        if (
            (top_ks is None or all(int(value) <= 0 for value in top_ks))
            and (top_ps is None or all(float(value) >= 1.0 for value in top_ps))
        ):
            return self._sample_unfiltered(
                logits_fp32, temperatures, greedy_mask, greedy_tokens
            )
        if bool(greedy_mask.all().item()):
            return greedy_tokens

        sampled_rows: list[torch.Tensor] = []
        all_sampling = bool((~greedy_mask).all().item())
        for row in range(int(logits_fp32.shape[0])):
            if not all_sampling and bool(greedy_mask[row].item()):
                sampled_rows.append(greedy_tokens[row])
                continue
            top_k = 0 if top_ks is None else int(top_ks[row])
            top_p = 1.0 if top_ps is None else float(top_ps[row])
            probs = filtered_sampling_probs(
                logits_fp32[row],
                temperatures[row],
                top_k=top_k,
                top_p=top_p,
            )
            sampled_rows.append(
                probs.div(torch.empty_like(probs).exponential_().clamp_min_(1e-10)).argmax()
            )
        sample_tokens = torch.stack(sampled_rows)
        sample_tokens = torch.where(greedy_mask, greedy_tokens, sample_tokens)
        return sample_tokens
