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

    Returns ``float('nan')`` when there are no infected samples in ``labels`` (recall
    is undefined). Older code incorrectly returned 1.0, which looked like a perfect
    score on splits with only healthy bees.
    """
    if binary_infected:
        support = (labels == 1).sum().item()
        if support == 0:
            return float("nan")
        tp = ((preds == 1) & (labels == 1)).sum().item()
        return tp / support
    support = (labels > 0).sum().item()
    if support == 0:
        return float("nan")
    tp = ((labels > 0) & (preds > 0)).sum().item()
    return tp / support
