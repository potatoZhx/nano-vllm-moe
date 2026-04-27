import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _rms_forward_2d(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def _rms_forward_3d(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def _add_rms_forward_2d(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def _add_rms_forward_3d(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim == 2:
            return self._rms_forward_2d(x)
        if x.ndim == 3:
            return self._rms_forward_3d(x)
        orig_shape = x.shape
        out_2d = self._rms_forward_2d(x.reshape(-1, orig_shape[-1]))
        return out_2d.reshape(orig_shape)

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2 and residual.ndim == 2:
            return self._add_rms_forward_2d(x, residual)
        if x.ndim == 3 and residual.ndim == 3:
            return self._add_rms_forward_3d(x, residual)
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        residual_2d = residual.reshape(-1, orig_shape[-1])
        out_2d, residual_2d = self._add_rms_forward_2d(x_2d, residual_2d)
        return out_2d.reshape(orig_shape), residual_2d.reshape(orig_shape)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
