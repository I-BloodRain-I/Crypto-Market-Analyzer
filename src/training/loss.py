from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossMulticlass(nn.Module):
    """
    Focal loss for multiclass classification.

    Implements the focal loss as described in the paper "Focal Loss for Dense Object Detection" 
    by Lin et al., adapted for multiclass classification. The loss helps address class imbalance 
    by down-weighting easy examples and focusing on hard ones.

    The loss is computed as: alpha * (1 - p_t)^gamma * cross_entropy_loss, 
    where p_t is the predicted probability for the true class.

    Args (for forward method):
        logits (torch.Tensor): Logits of shape [N, C, ...], where N is batch size, C is number of classes.
        target (torch.Tensor): Ground truth class indices of shape [N, ...], dtype long, values in {0, ..., C-1}.
        sample_weights (Optional[torch.Tensor]): Per-sample weights of shape [N, ...] or broadcastable to target.

    Returns:
        torch.Tensor: The computed focal loss, reduced according to the 'reduction' parameter.

    Attributes:
        gamma (float): Focusing parameter, default 2.0.
        class_weights (Optional[torch.Tensor]): Per-class weights (alpha), shape [C] or scalar.
        ignore_index (Optional[int]): Class index to ignore in loss computation.
        reduction (str): Reduction method, one of 'none', 'mean', 'sum'.
        eps (float): Small epsilon to avoid numerical issues.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,  # per-class alpha
        ignore_index: Optional[int] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        eps: float = 1e-12,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = float(eps)
        # deferred initialization based on number of classes/device
        self.register_buffer("_alpha", None, persistent=False)
        self._alpha_init = class_weights

    def _init_alpha(self, device, num_classes: int):
        if getattr(self, "_alpha_ready", False):
            return
        a = self._alpha_init
        if a is None:
            self._alpha = None
        else:
            if not torch.is_tensor(a):
                a = torch.tensor(float(a))
            if a.ndim == 1 and a.numel() != num_classes:
                raise ValueError(f"class_weights shape must be [C]={num_classes}")
            self._alpha = a.to(device)
        self._alpha_ready = True

    def forward(
        self,
        logits: torch.Tensor,          # [N, C, ...]
        target: torch.Tensor,          # [N, ...] (long)
        sample_weights: Optional[torch.Tensor] = None,  # [N, ...]
    ) -> torch.Tensor:
        if logits.ndim < 2:
            raise ValueError("logits must be [N, C, ...]")
        if target.dtype != torch.long:
            raise TypeError("target must be LongTensor class indices")

        N, C = logits.shape[:2]
        self._init_alpha(logits.device, C)

        # flatten additional dimensions into one
        if logits.ndim > 2:
            logits_f = logits.view(N, C, -1).transpose(1, 2).reshape(-1, C)  # [M, C]
            target_f = target.view(N, -1).reshape(-1)                         # [M]
            sw_f = None if sample_weights is None else sample_weights.view(N, -1).reshape(-1)
        else:
            logits_f, target_f, sw_f = logits, target, sample_weights

        # mask of valid samples
        valid = torch.ones_like(target_f, dtype=torch.bool)
        if self.ignore_index is not None:
            valid &= (target_f != self.ignore_index)
        if not valid.any():
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        logits_v = logits_f[valid]
        target_v = target_f[valid]
        sw_v = None if sw_f is None else sw_f[valid]

        logp = F.log_softmax(logits_v, dim=1)                    # [K, C]
        logpt = logp.gather(1, target_v.unsqueeze(1)).squeeze(1) # [K]
        pt = logpt.exp().clamp_min(self.eps)                     # [K]

        focal = (1.0 - pt) ** self.gamma                         # [K]
        ce = -logpt                                              # [K]

        if self._alpha is None:
            alpha_t = 1.0
        elif self._alpha.ndim == 0:
            alpha_t = self._alpha
        else:
            alpha_t = self._alpha[target_v]                      # [K]

        loss = alpha_t * focal * ce                              # [K]
        if sw_v is not None:
            loss = loss * sw_v

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            if sw_v is not None:
                denom = sw_v.sum().clamp_min(self.eps)
                return loss.sum() / denom
            return loss.mean()
        # none -> return in target shape
        out = torch.zeros_like(target_f, dtype=logits.dtype)
        out[valid] = loss
        if logits.ndim > 2:
            return out.view(N, *target.shape[1:])
        return out


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.

    Computes the pinball loss (also known as quantile loss) for quantile regression tasks.
    The pinball loss for a quantile q is defined as: (q - 1_{y < yhat}) * (y - yhat),
    where y is the true value, yhat is the prediction, and 1_{y < yhat} is the indicator function.

    This loss is asymmetric and penalizes underestimation and overestimation differently based on the quantile.

    Args (for forward method):
        preds (torch.Tensor): Predictions of shape [N, ..., Q], where Q is the number of quantiles.
        target (torch.Tensor): Target values of shape [N, ...].
        sample_weights (Optional[torch.Tensor]): Per-sample weights, broadcastable to target.

    Returns:
        torch.Tensor: The computed quantile loss, reduced according to the 'reduction' parameter.

    Attributes:
        quantiles: Tensor of quantiles, values in (0, 1).
        reduction (str): Reduction method, one of 'none', 'mean', 'sum'.
        quantile_weights (Optional[torch.Tensor]): Weights for each quantile, shape [Q].
        eps (float): Small epsilon to avoid division by zero.
    """
    def __init__(
        self,
        quantiles,
        reduction: Literal["none", "mean", "sum"] = "mean",
        quantile_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        q = torch.as_tensor(quantiles, dtype=torch.float32)
        if q.ndim != 1 or not torch.all((q > 0) & (q < 1)):
            raise ValueError("quantiles must be 1D with values in (0,1)")
        self.register_buffer("q", q)
        if quantile_weights is not None:
            qw = torch.as_tensor(quantile_weights, dtype=torch.float32)
            if qw.numel() != q.numel():
                raise ValueError("quantile_weights must have length Q")
            self.register_buffer("qw", qw / (qw.sum() + eps))
        else:
            self.register_buffer("qw", None)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        preds: torch.Tensor,        # [N, ..., Q]
        target: torch.Tensor,       # [N, ...]
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self.q.numel()
        if preds.shape[-1] != Q:
            raise ValueError(f"last dim of preds must be Q={Q}")
        if preds.shape[:-1] != target.shape:
            raise ValueError("preds.shape[:-1] must match target.shape")
        if target.dtype != torch.float32 and target.dtype != torch.float64:
            raise TypeError("target must be FloatTensor")

        # mask of valid samples (ignore NaN in target)
        valid = torch.isfinite(target)
        if not valid.any():
            return torch.zeros((), dtype=preds.dtype, device=preds.device)

        y = target[valid]                                     # [K]
        yhat = preds.reshape(-1, Q)[valid.reshape(-1)]        # [K, Q]
        e = y.unsqueeze(-1) - yhat                            # [K, Q]

        # pinball: (q - 1_{e<0}) * e
        loss_q = torch.maximum(self.q.view(1, -1) * e, (self.q.view(1, -1) - 1) * e)  # [K, Q]
        # equivalent: torch.where(e>=0, q*e, (q-1)*e)

        if self.qw is not None:
            loss_v = (loss_q * self.qw.view(1, -1)).sum(dim=-1)   # [K]
        else:
            loss_v = loss_q.mean(dim=-1)                          # [K]

        if sample_weights is not None:
            sw = sample_weights[valid]
            loss_v = loss_v * sw

        if self.reduction == "sum":
            return loss_v.sum()
        if self.reduction == "mean":
            if sample_weights is not None:
                denom = sw.sum().clamp_min(self.eps)
                return loss_v.sum() / denom
            return loss_v.mean()
        # none -> return in target shape
        out = torch.zeros_like(target, dtype=preds.dtype)
        out[valid] = loss_v
        return out
