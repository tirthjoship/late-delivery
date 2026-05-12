"""Model evaluation adapter.

Computes classification metrics using sklearn and returns
domain MetricsResult. Domain says WHAT (F1 primary), adapter says HOW.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from domain.models import MetricsResult


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> MetricsResult:
    """Compute classification metrics.

    F1 is the primary metric — accuracy is misleading with 55/45 class split.
    """
    cm = confusion_matrix(y_true, y_pred)
    return MetricsResult(
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        auc_roc=float(roc_auc_score(y_true, y_proba)),
        confusion_matrix=tuple(tuple(int(x) for x in row) for row in cm),
    )
