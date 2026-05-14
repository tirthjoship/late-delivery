"""Tests for the Plotly charts adapter.

All functions must return go.Figure. Input data uses minimal synthetic fixtures.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from adapters.visualization.plotly_charts import (
    cluster_scatter_2d,
    confusion_matrix_heatmap,
    late_rate_by_region_bar,
    risk_gauge,
    shap_dependence_scatter,
    shap_importance_bar,
    shap_waterfall,
    shipping_mode_bar,
)


# ---------------------------------------------------------------------------
# risk_gauge
# ---------------------------------------------------------------------------
class TestRiskGauge:
    def test_returns_figure(self) -> None:
        fig = risk_gauge(0.75)
        assert isinstance(fig, go.Figure)

    def test_high_risk_probability(self) -> None:
        fig = risk_gauge(0.85)
        assert isinstance(fig, go.Figure)

    def test_low_risk_probability(self) -> None:
        fig = risk_gauge(0.1)
        assert isinstance(fig, go.Figure)

    def test_boundary_probability(self) -> None:
        fig = risk_gauge(0.0)
        assert isinstance(fig, go.Figure)
        fig2 = risk_gauge(1.0)
        assert isinstance(fig2, go.Figure)


# ---------------------------------------------------------------------------
# confusion_matrix_heatmap
# ---------------------------------------------------------------------------
class TestConfusionMatrixHeatmap:
    def test_returns_figure(self) -> None:
        cm = ((50, 10), (5, 35))
        fig = confusion_matrix_heatmap(cm)
        assert isinstance(fig, go.Figure)

    def test_2x2_matrix(self) -> None:
        cm = ((100, 20), (15, 65))
        fig = confusion_matrix_heatmap(cm)
        assert isinstance(fig, go.Figure)

    def test_all_correct_matrix(self) -> None:
        cm = ((80, 0), (0, 120))
        fig = confusion_matrix_heatmap(cm)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# shap_waterfall
# ---------------------------------------------------------------------------
class TestShapWaterfall:
    def test_returns_figure(self) -> None:
        shap_values = {
            "shipping_mode_First Class": 0.45,
            "days_for_shipment_scheduled": -0.12,
            "benefit_per_order": 0.08,
        }
        fig = shap_waterfall(shap_values, expected_value=0.3)
        assert isinstance(fig, go.Figure)

    def test_more_than_10_features(self) -> None:
        shap_values = {f"feature_{i}": float(i) * 0.05 - 0.25 for i in range(15)}
        fig = shap_waterfall(shap_values, expected_value=0.5)
        assert isinstance(fig, go.Figure)

    def test_empty_shap_values(self) -> None:
        fig = shap_waterfall({}, expected_value=0.0)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# shap_importance_bar
# ---------------------------------------------------------------------------
class TestShapImportanceBar:
    def test_returns_figure(self) -> None:
        importances = {
            "shipping_mode_First Class": 0.45,
            "days_for_shipment_scheduled": 0.30,
            "benefit_per_order": 0.15,
        }
        fig = shap_importance_bar(importances)
        assert isinstance(fig, go.Figure)

    def test_single_feature(self) -> None:
        fig = shap_importance_bar({"feature_a": 0.9})
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# shap_dependence_scatter
# ---------------------------------------------------------------------------
class TestShapDependenceScatter:
    def test_returns_figure(self) -> None:
        rng = np.random.default_rng(42)
        feature_values = rng.uniform(1, 7, 50).tolist()
        shap_vals = rng.uniform(-0.5, 0.5, 50).tolist()
        fig = shap_dependence_scatter("days_for_shipment_scheduled", feature_values, shap_vals)
        assert isinstance(fig, go.Figure)

    def test_small_input(self) -> None:
        fig = shap_dependence_scatter("benefit_per_order", [1.0, 2.0], [0.1, -0.1])
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# cluster_scatter_2d
# ---------------------------------------------------------------------------
class TestClusterScatter2D:
    def test_returns_figure(self) -> None:
        rng = np.random.default_rng(42)
        X_pca = rng.standard_normal((50, 2))
        labels = np.array([0, 1, 2, 3] * 12 + [0, 1])
        fig = cluster_scatter_2d(X_pca, labels)
        assert isinstance(fig, go.Figure)

    def test_with_cohort_labels(self) -> None:
        rng = np.random.default_rng(7)
        X_pca = rng.standard_normal((20, 2))
        labels = np.array([0, 1] * 10)
        cohort_labels = {0: "High-Risk First Class", 1: "Low-Risk Same Day"}
        fig = cluster_scatter_2d(X_pca, labels, cohort_labels=cohort_labels)
        assert isinstance(fig, go.Figure)

    def test_without_cohort_labels(self) -> None:
        rng = np.random.default_rng(0)
        X_pca = rng.standard_normal((30, 2))
        labels = np.array([0, 1, 2] * 10)
        fig = cluster_scatter_2d(X_pca, labels, cohort_labels=None)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# shipping_mode_bar
# ---------------------------------------------------------------------------
class TestShippingModeBar:
    def test_returns_figure(self) -> None:
        data = {
            "First Class": 0.95,
            "Second Class": 0.76,
            "Standard Class": 0.38,
            "Same Day": 0.15,
        }
        fig = shipping_mode_bar(data)
        assert isinstance(fig, go.Figure)

    def test_single_mode(self) -> None:
        fig = shipping_mode_bar({"Same Day": 0.10})
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# late_rate_by_region_bar
# ---------------------------------------------------------------------------
class TestLateRateByRegionBar:
    def test_returns_figure(self) -> None:
        data = {
            "North America": 0.62,
            "Europe": 0.55,
            "Asia": 0.48,
            "South America": 0.70,
        }
        fig = late_rate_by_region_bar(data)
        assert isinstance(fig, go.Figure)

    def test_single_region(self) -> None:
        fig = late_rate_by_region_bar({"Asia": 0.50})
        assert isinstance(fig, go.Figure)
