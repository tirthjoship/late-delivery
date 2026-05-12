"""Tests for adapters/ml/evaluation.py."""

import numpy as np
import pytest

from adapters.ml.evaluation import compute_metrics
from domain.models import MetricsResult


class TestComputeMetrics:
    def test_returns_metrics_result(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.4])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert isinstance(result, MetricsResult)

    def test_perfect_predictions_give_f1_one(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_proba = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert result.f1 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)

    def test_all_wrong_predictions_give_f1_zero(self) -> None:
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert result.f1 == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        y_proba = np.array([0.8, 0.2, 0.4, 0.6])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2

    def test_auc_roc_between_zero_and_one(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.6, 0.4, 0.2])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert 0.0 <= result.auc_roc <= 1.0
