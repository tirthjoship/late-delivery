"""Tests for adapters/ml/kmeans_clusterer.py.

Contract tests: verify KMeansClusterer implements ClusteringPort,
returns valid labels, centroids, and silhouette scores.
"""

import numpy as np
import pytest

from adapters.ml.kmeans_clusterer import KMeansClusterer


@pytest.fixture
def synthetic_feature_matrix() -> np.ndarray:
    """50-row synthetic feature matrix with 3 natural clusters."""
    rng = np.random.default_rng(42)
    cluster_0 = rng.normal(loc=[0, 0, 0], scale=0.5, size=(20, 3))
    cluster_1 = rng.normal(loc=[5, 5, 5], scale=0.5, size=(15, 3))
    cluster_2 = rng.normal(loc=[10, 0, 5], scale=0.5, size=(15, 3))
    return np.vstack([cluster_0, cluster_1, cluster_2])


class TestKMeansClusterer:
    """Contract tests for KMeansClusterer."""

    def test_fit_no_error(self, synthetic_feature_matrix: np.ndarray) -> None:
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(synthetic_feature_matrix)

    def test_predict_returns_labels(self, synthetic_feature_matrix: np.ndarray) -> None:
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(synthetic_feature_matrix)
        labels = clusterer.predict(synthetic_feature_matrix)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 50
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_get_labels_matches_predict(
        self, synthetic_feature_matrix: np.ndarray
    ) -> None:
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(synthetic_feature_matrix)
        fit_labels = clusterer.get_labels()
        predict_labels = clusterer.predict(synthetic_feature_matrix)
        np.testing.assert_array_equal(fit_labels, predict_labels)

    def test_get_centroids_shape(self, synthetic_feature_matrix: np.ndarray) -> None:
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(synthetic_feature_matrix)
        centroids = clusterer.get_centroids()
        assert centroids.shape == (3, 3)  # 3 clusters, 3 features

    def test_silhouette_score_valid_range(
        self, synthetic_feature_matrix: np.ndarray
    ) -> None:
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(synthetic_feature_matrix)
        score = clusterer.compute_silhouette(synthetic_feature_matrix)
        assert -1.0 <= score <= 1.0

    def test_inertia_is_positive(self, synthetic_feature_matrix: np.ndarray) -> None:
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(synthetic_feature_matrix)
        assert clusterer.inertia > 0.0

    def test_fit_without_predict_raises_on_get_labels(self) -> None:
        clusterer = KMeansClusterer(n_clusters=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            clusterer.get_labels()
