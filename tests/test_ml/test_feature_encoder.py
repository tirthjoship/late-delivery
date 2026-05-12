"""Tests for adapters/ml/feature_encoder.py.

Contract tests: verify encoder transforms raw feature dicts correctly,
preserves shape, and never produces leakage columns.
"""

import pytest

from adapters.data.csv_repository import LEAKAGE_COLUMNS
from adapters.ml.feature_encoder import FeatureEncoder
from domain.services import extract_features


class TestFeatureEncoder:
    """Contract tests for FeatureEncoder."""

    def _raw_features(self, synthetic_orders) -> list[dict]:
        return [extract_features(o) for o in synthetic_orders]

    def test_fit_transform_preserves_row_count(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        assert X.shape[0] == len(raw)

    def test_fit_transform_output_has_columns(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        assert X.shape[1] > 0

    def test_transform_without_fit_raises(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        with pytest.raises(RuntimeError, match="not fitted"):
            encoder.transform(raw)

    def test_get_feature_names_matches_columns(self, synthetic_orders) -> None:
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        X = encoder.fit_transform(raw)
        names = encoder.get_feature_names()
        assert len(names) == X.shape[1]

    def test_no_leakage_column_in_feature_names(self, synthetic_orders) -> None:
        """CRITICAL: encoder output must never contain leakage columns."""
        raw = self._raw_features(synthetic_orders)
        encoder = FeatureEncoder()
        encoder.fit_transform(raw)
        names = encoder.get_feature_names()
        for name in names:
            for leak_col in LEAKAGE_COLUMNS:
                assert leak_col not in name, (
                    f"Leakage column '{leak_col}' found in feature name '{name}'"
                )

    def test_fit_on_train_transform_on_test(self, synthetic_orders) -> None:
        """Encoder fit on subset A can transform subset B without error."""
        raw = self._raw_features(synthetic_orders)
        train_raw = raw[:80]
        test_raw = raw[80:]
        encoder = FeatureEncoder()
        X_train = encoder.fit_transform(train_raw)
        X_test = encoder.transform(test_raw)
        assert X_train.shape[1] == X_test.shape[1]
        assert X_test.shape[0] == 20
