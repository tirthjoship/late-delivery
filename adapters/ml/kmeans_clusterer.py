"""K-Means clustering adapter.

Implements ClusteringPort for unsupervised order segmentation.
Wraps sklearn KMeans with silhouette scoring for K selection.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansClusterer:
    """K-Means clustering adapter implementing ClusteringPort.

    Args:
        n_clusters: Number of clusters (K).
        random_state: Random seed for reproducibility.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42) -> None:
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._model: KMeans | None = None

    def fit(self, X: np.ndarray) -> None:
        """Fit K-Means on feature matrix X."""
        self._model = KMeans(
            n_clusters=self._n_clusters,
            random_state=self._random_state,
            n_init=10,
        )
        self._model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels to feature matrix X."""
        if self._model is None:
            raise RuntimeError("KMeansClusterer is not fitted. Call fit() first.")
        result: np.ndarray = self._model.predict(X)
        return result

    def get_centroids(self) -> np.ndarray:
        """Return cluster centroids (K x n_features)."""
        if self._model is None:
            raise RuntimeError("KMeansClusterer is not fitted. Call fit() first.")
        centers: np.ndarray = self._model.cluster_centers_
        return centers

    def get_labels(self) -> np.ndarray:
        """Return cluster labels assigned during fit."""
        if self._model is None:
            raise RuntimeError("KMeansClusterer is not fitted. Call fit() first.")
        labels: np.ndarray = self._model.labels_
        return labels

    def compute_silhouette(self, X: np.ndarray) -> float:
        """Compute silhouette score for current clustering."""
        if self._model is None:
            raise RuntimeError("KMeansClusterer is not fitted. Call fit() first.")
        return float(silhouette_score(X, self._model.labels_))

    @property
    def inertia(self) -> float:
        """Sum of squared distances to nearest cluster center."""
        if self._model is None:
            raise RuntimeError("KMeansClusterer is not fitted. Call fit() first.")
        return float(self._model.inertia_)
