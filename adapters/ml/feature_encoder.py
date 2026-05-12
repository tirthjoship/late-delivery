"""Feature encoding adapter.

Transforms raw feature dicts from domain/services.extract_features()
into model-ready numeric arrays using sklearn preprocessing.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEncoder:
    """Encode raw feature dicts into numeric arrays for ML models.

    Categorical columns are one-hot encoded. Numeric columns are
    standard-scaled. Column order is deterministic.

    Note: order_country is deliberately excluded from CATEGORICAL
    to avoid 164-column sparse explosion (see spec Decision D7).
    """

    CATEGORICAL: list[str] = ["shipping_mode", "order_region"]
    NUMERIC: list[str] = [
        "days_for_shipment_scheduled",
        "order_month",
        "order_day_of_week",
        "benefit_per_order",
        "sales_per_customer",
        "order_profit_per_order",
        "item_count",
        "total_quantity",
        "total_discount",
        "avg_unit_price",
    ]

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] | None = None

    def fit(self, raw_features: list[dict]) -> "FeatureEncoder":
        """Fit encoder on raw feature dicts. Must be called on training data ONLY."""
        df = pd.DataFrame(raw_features)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CATEGORICAL,
                ),
                ("num", StandardScaler(), self.NUMERIC),
            ]
        )
        self._pipeline = Pipeline([("preprocessor", preprocessor)])
        self._pipeline.fit(df)
        self._feature_names = self._build_feature_names()
        return self

    def transform(self, raw_features: list[dict]) -> pd.DataFrame:
        """Transform raw feature dicts using fitted encoder."""
        if self._pipeline is None:
            raise RuntimeError("FeatureEncoder is not fitted. Call fit() first.")
        df = pd.DataFrame(raw_features)
        transformed = self._pipeline.transform(df)
        return pd.DataFrame(transformed, columns=self._feature_names)

    def fit_transform(self, raw_features: list[dict]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(raw_features)
        return self.transform(raw_features)

    def get_feature_names(self) -> list[str]:
        """Return post-encoding feature names. Used by SHAP for labeling."""
        if self._feature_names is None:
            raise RuntimeError("FeatureEncoder is not fitted. Call fit() first.")
        return self._feature_names

    def _build_feature_names(self) -> list[str]:
        """Extract feature names from fitted ColumnTransformer."""
        assert self._pipeline is not None
        ct: ColumnTransformer = self._pipeline.named_steps["preprocessor"]
        cat_names = list(
            ct.named_transformers_["cat"].get_feature_names_out(self.CATEGORICAL)
        )
        num_names = list(self.NUMERIC)
        return cat_names + num_names
