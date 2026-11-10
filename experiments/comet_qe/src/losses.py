from __future__ import annotations

"""
losses.py

Custom loss functions for regression in the BEA 2026 shared task.

Supported losses:
  - mse         standard mean squared error  (default, same as before)
  - huber       Huber / smooth-L1 loss       (robust to outliers)
  - log_cosh    log-cosh loss                 (smooth approximation of MAE)
  - ccc         1 − concordance correlation   (directly optimises agreement)

Usage in a YAML config:

    loss_function: huber
    loss_kwargs:
      delta: 1.0
"""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss Implementations
# ---------------------------------------------------------------------------

def mse_loss(preds: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard mean squared error (drop-in for the HuggingFace default)."""
    return F.mse_loss(preds.view(-1), labels.view(-1))


def huber_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    delta: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Huber loss — less sensitive to outliers than MSE.

    Args:
        delta: threshold at which the loss transitions from quadratic to linear.
    """
    return F.huber_loss(preds.view(-1), labels.view(-1), delta=delta)


def log_cosh_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Log-cosh loss — smooth approximation of MAE, twice differentiable."""
    diff = preds.view(-1) - labels.view(-1)
    # log(cosh(x)) = abs(x) + softplus(-2*abs(x)) - log(2)  (symmetric numerically stable form)
    abs_diff = torch.abs(diff)
    return torch.mean(
        abs_diff + F.softplus(-2.0 * abs_diff) - torch.log(torch.tensor(2.0, device=diff.device))
    )


def ccc_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """1 − Concordance Correlation Coefficient.

    Directly optimises agreement between predictions and labels.
    CCC = 1 means perfect concordance, so we minimise 1 − CCC.
    """
    p = preds.view(-1)
    l = labels.view(-1)

    # Avoid NaN if the batch has 0 variance (early training or batch_size < 2)
    if p.shape[0] < 2:
        return torch.tensor(0.0, device=p.device, requires_grad=True)

    mean_p = p.mean()
    mean_l = l.mean()
    var_p = p.var(unbiased=False)
    var_l = l.var(unbiased=False)
    covariance = ((p - mean_p) * (l - mean_l)).mean()
    
    ccc = (2.0 * covariance) / (var_p + var_l + (mean_p - mean_l) ** 2 + 1e-8)
    return 1.0 - ccc


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "mse": mse_loss,
    "huber": huber_loss,
    "log_cosh": log_cosh_loss,
    "ccc": ccc_loss,
}


def get_loss_function(
    name: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[Callable[..., torch.Tensor], Dict[str, Any]]:
    """Look up a loss function by name (defaults to ``mse``).

    Returns:
        A (loss_fn, loss_kwargs) tuple ready to be passed to `CustomLossTrainer`.

    Raises:
        ValueError: if the name is not in the registry.
    """
    name = (name or "mse").lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss function '{name}'. "
            f"Available: {sorted(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name], dict(kwargs or {})
