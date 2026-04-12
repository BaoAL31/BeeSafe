"""Metrics for BeeSafe image-level classification."""

from __future__ import annotations

import torch


def infected_recall(
    preds: torch.Tensor,
    labels: torch.Tensor,
    binary_infected: bool,
) -> float:
    """
    Recall for catching infection: among truly infected samples, fraction predicted
    as infected (not healthy). Binary: class 1 = infected. 3-class: labels > 0
    are infected; prediction counts if pred > 0.
    """
    if binary_infected:
        support = (labels == 1).sum().item()
        if support == 0:
            return 1.0
        tp = ((preds == 1) & (labels == 1)).sum().item()
        return tp / support
    support = (labels > 0).sum().item()
    if support == 0:
        return 1.0
    tp = ((labels > 0) & (preds > 0)).sum().item()
    return tp / support
